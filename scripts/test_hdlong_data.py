"""
Smoke test for HDLongDataset.

Usage:
    python scripts/test_hdlong_data.py --data_root D:\\Ky8\\hdlong-complexv1
    python scripts/test_hdlong_data.py --data_root /content/drive/MyDrive/CodeColab/DataThesis/hdlong-complexv1

Prints shapes, dtypes, value ranges, and saves the first sample's first input
image + normal map to disk so you can eyeball whether decoding is correct.
"""

import os
import sys
import argparse
import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from src.data.data_hdlong import HDLongDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True,
                   help="Folder containing object_xxx/ subfolders")
    p.add_argument("--K", type=int, default=6)
    p.add_argument("--image_size", type=int, default=256)
    p.add_argument("--n_samples", type=int, default=3,
                   help="How many samples to dry-run through the loader")
    p.add_argument("--out_dir", default="scripts/_hdlong_test_out")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ds = HDLongDataset(
        mode="Train",
        data_root=args.data_root,
        numImages=args.K,
        image_size=args.image_size,
        repeat=1,
        ratio_train=1.0, ratio_val=0.0,   # one-folder smoke test: put everything in train
    )

    print(f"\n[dataset] len = {len(ds)}")

    for i in range(min(args.n_samples, len(ds))):
        print(f"\n=== sample {i} ===")
        s = ds[i]
        for k, v in s.items():
            if hasattr(v, "shape"):
                print(f"  {k:14s}  shape={tuple(v.shape)}  dtype={v.dtype}  "
                      f"min={float(v.min()):+.4f}  max={float(v.max()):+.4f}  "
                      f"mean={float(v.float().mean()):+.4f}")
            else:
                print(f"  {k:14s}  {v}")

        if i == 0:
            # Save first image + normal so user can sanity-check visually.
            img0 = s["img"][..., 0].permute(1, 2, 0).cpu().numpy()       # [H,W,3]
            nml = s["nml"][..., 0].permute(1, 2, 0).cpu().numpy()        # [H,W,3]
            mask = s["mask"][0, :, :, 0].cpu().numpy()                   # [H,W]

            # tone-map img for preview
            img_vis = np.clip(img0 / max(img0.max(), 1e-6), 0, 1)
            cv2.imwrite(os.path.join(args.out_dir, "input_0.png"),
                        cv2.cvtColor((img_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            # normal -> 0..255 (encode as (n+1)/2 for display)
            nml_vis = np.clip((nml + 1.0) * 0.5, 0, 1)
            cv2.imwrite(os.path.join(args.out_dir, "normal.png"),
                        cv2.cvtColor((nml_vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

            cv2.imwrite(os.path.join(args.out_dir, "mask.png"),
                        (mask * 255).astype(np.uint8))

            print(f"\n[saved previews to {args.out_dir}]")

    print("\n[OK] dataset loads without crashing.")


if __name__ == "__main__":
    main()
