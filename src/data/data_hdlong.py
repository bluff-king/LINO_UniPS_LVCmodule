"""
HDL-long synthetic photometric stereo dataset.

Format (one object folder):
    object_name/
        cams.config            # 10 lines: cam_id <radius azim elev> <4x4 transform>
        dir_lights.config      # 20 lines: name azim elev intensity
        env_lights.config      # 10 lines: name hdri_file euler_x euler_y euler_z
        point_lights.config    # 20 lines: name radius azim elev intensity
        light_means.config     # dir_mean / point_mean / env_mean
        object.config          # mesh + texture
        cam_00001/
            binary_mask.exr
            depth_map.exr
            global_normal.exr      # world space, [-1,1] raw
            local_normal.exr       # camera space, [-1,1] raw  <-- used for PS
            dir_light_*.exr  (x20)
            env_light_*.exr  (x10)
            point_light_*.exr (x20)
        cam_00002/
        ...
        cam_00010/

Sampling strategy (per __getitem__):
    1. Pick 1 random cam from the 10
    2. For each of K input slots:
         - Choose hdl_type ∈ {hdl1=point+env, hdl2=dir+env}
         - alpha ~ Uniform(alpha_lo, alpha_hi)
         - Pick random env_light, and random point/dir light
         - I = alpha * (I_pri / pri_mean) + (1-alpha) * (I_env / env_mean)
"""

import os
import glob
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


# ----------------------------------------------------------------------------
# EXR helpers (use OpenCV — opencv-python-headless ships EXR support)
# ----------------------------------------------------------------------------

# Make sure OpenCV is allowed to read EXR (cv2 sometimes needs this env flag).
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")


def _read_exr_rgb(path: str) -> np.ndarray:
    """Read an EXR as float32 RGB [H, W, 3]."""
    img = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cv2 failed to read EXR: {path}")
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(img, dtype=np.float32)


def _read_exr_mask(path: str) -> np.ndarray:
    """Read binary_mask.exr as float32 [H, W] in {0, 1}."""
    m = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"cv2 failed to read mask EXR: {path}")
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0.5).astype(np.float32)


# ----------------------------------------------------------------------------
# Config parsers
# ----------------------------------------------------------------------------

def _parse_kv_config(path: str) -> dict:
    """light_means.config -> {'dir_mean': 0.04..., 'point_mean': 0.001..., 'env_mean': 0.16...}."""
    out = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return out


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class HDLongDataset(Dataset):
    """
    Synthetic mixed-lighting PS dataset (point/dir + env), per thesis spec.

    Args:
        mode          : 'Train' | 'Val' | 'Test'
        data_root     : path containing many object folders
        numImages     : K input images per sample (default 6)
        image_size    : resize all images to this (default 512)
        hdl_types     : tuple of {'hdl1', 'hdl2'}; sampled uniformly per input slot
                            hdl1 = point + env, hdl2 = dir + env
        alpha_range   : (lo, hi) for the mixing coefficient
        ratio_train   : split fraction
        ratio_val     : split fraction (rest -> Test)
        repeat        : virtual length multiplier; useful when dataset is tiny
                        (e.g. for a smoke test with 1 object set repeat=64 so
                        an epoch has 64 steps instead of 1).
        seed          : RNG seed for the train/val/test partition
    """

    def __init__(
        self,
        mode: str,
        data_root: str,
        numImages: int = 6,
        image_size: int = 256,           # dataset native res; no resize needed
        hdl_types=("hdl1", "hdl2"),
        alpha_range=(0.1, 0.9),
        ratio_train: float = 0.9,
        ratio_val: float = 0.05,
        repeat: int = 1,
        seed: int = 42,
        **kwargs,            # absorb unused kwargs for CLI compatibility
    ):
        self.mode = mode
        self.data_root = data_root
        self.K = int(numImages)
        self.S = int(image_size)
        self.hdl_types = tuple(hdl_types)
        self.alpha_lo, self.alpha_hi = alpha_range
        self.repeat = max(1, int(repeat))

        # --- list object folders -------------------------------------------------
        print(f"[HDLongDataset/{mode}] scanning {data_root} ...")
        all_objs = []
        for d in tqdm(sorted(os.listdir(data_root)), desc="scanning objects"):
            p = os.path.join(data_root, d)
            if os.path.isdir(p):
                all_objs.append(p)
        n = len(all_objs)
        if n == 0:
            raise RuntimeError(f"No object folders under {data_root}")

        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        all_objs = [all_objs[i] for i in idx]

        n_tr = max(1, int(n * ratio_train))
        n_va = max(1, int(n * ratio_val)) if n > 1 else 0

        if mode == "Train":
            self.objs = all_objs[:n_tr]
        elif mode in ("Val", "Validation"):
            self.objs = all_objs[n_tr:n_tr + n_va]
            if not self.objs:                     # tiny-dataset fallback
                self.objs = all_objs[:1]
        else:                                     # Test
            self.objs = all_objs[n_tr + n_va:] or all_objs[:1]

        print(f"[HDLongDataset/{mode}] {len(self.objs)} objects "
              f"(virtual length = {len(self.objs) * self.repeat})")

    # ----------------------------------------------------------------------------

    def __len__(self):
        return len(self.objs) * self.repeat

    def _resize(self, arr: np.ndarray, interp=cv2.INTER_LINEAR) -> np.ndarray:
        if arr.shape[0] == self.S and arr.shape[1] == self.S:
            return arr
        return cv2.resize(arr, (self.S, self.S), interpolation=interp)

    # ----------------------------------------------------------------------------

    def __getitem__(self, index):
        # support repeat: collapse virtual index -> object index
        obj_idx = index % len(self.objs)
        last_err = None
        for _ in range(5):
            try:
                return self._get_one(obj_idx)
            except Exception as e:
                last_err = e
                obj_idx = (obj_idx + 1) % len(self.objs)
        raise RuntimeError(f"HDLongDataset: 5 consecutive failures; last: {last_err}")

    def _get_one(self, obj_idx: int) -> dict:
        obj_path = self.objs[obj_idx]
        means = _parse_kv_config(os.path.join(obj_path, "light_means.config"))
        dir_mean = float(means.get("dir_mean", 1.0)) or 1.0
        pt_mean = float(means.get("point_mean", 1.0)) or 1.0
        env_mean = float(means.get("env_mean", 1.0)) or 1.0

        # --- pick 1 random cam ----------------------------------------------------
        cams = sorted([d for d in os.listdir(obj_path)
                       if d.startswith("cam_") and os.path.isdir(os.path.join(obj_path, d))])
        if not cams:
            raise RuntimeError(f"No cam_* folders in {obj_path}")
        cam = random.choice(cams)
        cam_path = os.path.join(obj_path, cam)

        # --- GT: normal + mask ----------------------------------------------------
        # local_normal.exr is stored as (n+1)/2 in [0,1] — decode back to [-1,1]
        N_raw = _read_exr_rgb(os.path.join(cam_path, "local_normal.exr"))[..., :3]
        N = N_raw * 2.0 - 1.0                                                   # decode -> [-1,1]
        M = _read_exr_mask(os.path.join(cam_path, "binary_mask.exr"))           # [H,W]
        N = self._resize(N, cv2.INTER_NEAREST)
        M = self._resize(M, cv2.INTER_NEAREST)
        M = M[..., None]                                                        # [H,W,1]
        N = N * M                                                               # zero outside mask

        # --- light-image candidate lists -----------------------------------------
        env_paths = sorted(glob.glob(os.path.join(cam_path, "env_light_*.exr")))
        dir_paths = sorted(glob.glob(os.path.join(cam_path, "dir_light_*.exr")))
        pt_paths = sorted(glob.glob(os.path.join(cam_path, "point_light_*.exr")))
        if not env_paths or (not dir_paths and not pt_paths):
            raise RuntimeError(f"Missing light EXRs in {cam_path}")

        # --- assemble K mixed inputs ---------------------------------------------
        imgs = np.zeros((self.S, self.S, 3, self.K), dtype=np.float32)
        for i in range(self.K):
            avail = [t for t in self.hdl_types
                     if (t == "hdl1" and pt_paths) or (t == "hdl2" and dir_paths)]
            hdl = random.choice(avail) if avail else "hdl2"
            alpha = random.uniform(self.alpha_lo, self.alpha_hi)

            I_env = _read_exr_rgb(random.choice(env_paths)) / (env_mean + 1e-8)

            if hdl == "hdl1":
                I_pri = _read_exr_rgb(random.choice(pt_paths)) / (pt_mean + 1e-8)
            else:
                I_pri = _read_exr_rgb(random.choice(dir_paths)) / (dir_mean + 1e-8)

            I_mix = alpha * I_pri + (1.0 - alpha) * I_env
            I_mix = self._resize(I_mix, cv2.INTER_LINEAR)
            # numerical safety
            I_mix = np.nan_to_num(I_mix, nan=0.0, posinf=0.0, neginf=0.0)
            imgs[..., i] = I_mix

        # mask images (encoder expects masked inputs)
        imgs = imgs * M[..., None]                                              # [H,W,3,K]

        # --- dummy auxiliary fields (model expects these keys) -------------------
        # We don't compute env-HDRI / point / area GT yet — feed zeros so the
        # forward pass runs.  The auxiliary losses will be uninformative but
        # the main normal loss is still correct.
        env_light = np.zeros((self.K, 9, 256, 256), dtype=np.float32)
        point_lights = np.zeros((self.K, 12), dtype=np.float32)
        area_light = np.zeros((self.K, 5), dtype=np.float32)

        # --- to tensors with the layout the model expects ------------------------
        # img : [3, H, W, K]
        # nml : [3, H, W, 1]
        # mask: [1, H, W, 1]
        img_t = torch.from_numpy(imgs).permute(2, 0, 1, 3).contiguous()
        nml_t = torch.from_numpy(N).permute(2, 0, 1).unsqueeze(-1).contiguous()
        mask_t = torch.from_numpy(M).permute(2, 0, 1).unsqueeze(-1).contiguous()

        return {
            "img": img_t,
            "nml": nml_t,
            "mask": mask_t,
            "env_light": torch.from_numpy(env_light),
            "point_lights": torch.from_numpy(point_lights),
            "area_light": torch.from_numpy(area_light),
            "numberOfImages": torch.tensor([self.K], dtype=torch.int32),
        }
