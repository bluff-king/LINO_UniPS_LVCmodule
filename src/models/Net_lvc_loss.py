"""
Version A — LVC Loss only.

Modifies the training loss of LiNo-UniPS with a confidence-weighted
formulation: per-pixel gradient updates are down-weighted for pixels whose
intensity shows little variation across the K input images (low CV2).
Feature aggregation is unchanged from the baseline.

Ablation role: isolates the contribution of the confidence-weighted loss
               (independent of the feature-scaling mechanism).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2

import pytorch_lightning as pl
from torchmetrics import MeanMetric
from typing import Dict, Any

from .module.utils import (
    ScaleInvariantSpatialLightImageEncoder,
    GLC_Upsample, GLC_Aggregation, Regressor,
    HdriFeatureProj, EnvLightHead, PointLightAlign, AreaAlign,
    make_index_list,
)
from .module.lvc import LVCModule
from .utils import decompose_tensors, gauss_filter
from .hdri_encoder.condmodel_hdri_v2 import HDRICondModel
from src.models.utils.compute_mae import compute_mae_np, compute_mae
from src.models.utils.utils import sobel_edge_map
from datetime import datetime


class NetLVC_Loss(nn.Module):
    """LiNo-UniPS + LVC confidence-weighted loss (Version A)."""

    def __init__(
        self,
        pixel_samples,
        output,
        depth,
        lvc_alpha: float = 10.0,
        lvc_w_min: float = 0.1,
        lvc_per_channel: bool = True,
    ):
        super().__init__()
        self.target = output
        self.pixel_samples = pixel_samples
        self.depth = depth
        self.glc_smoothing = True
        self.input_dim = 4
        self.image_encoder = ScaleInvariantSpatialLightImageEncoder(
            self.input_dim, self.depth, use_efficient_attention=False
        )
        self.mode = None
        self.input_dim = 0
        self.glc_upsample = GLC_Upsample(
            256 + self.input_dim, num_enc_sab=1, dim_hidden=256,
            dim_feedforward=1024, use_efficient_attention=True,
        )
        self.glc_aggregation = GLC_Aggregation(
            256 + self.input_dim, num_agg_transformer=2, dim_aggout=384,
            dim_feedforward=1024, use_efficient_attention=False,
        )
        ckpt_path = os.getenv("HDRI_ENCODER_CKPT")
        state_dict = torch.load(ckpt_path, weights_only=False) if ckpt_path and os.path.exists(ckpt_path) else {}
        state_dict = {k[16:]: v for k, v in state_dict.items() if "hdri_cond_model." in k}
        self.img_embedding = nn.Sequential(
            nn.Linear(3, 32), nn.LeakyReLU(), nn.Linear(32, 256)
        )
        self.hdri_encoder = HDRICondModel()
        self.hdri_encoder.load_state_dict(state_dict, strict=False)
        self.hdri_encoder.requires_grad_(False)
        del state_dict
        self.regressor = Regressor(
            384, num_enc_sab=1, use_efficient_attention=True,
            dim_feedforward=1024, output=self.target,
        )
        self.criterionL2 = nn.MSELoss(reduction='mean')
        self.env_feature_proj = HdriFeatureProj()
        self.env_light_head = EnvLightHead(input_feature_dim=1536, output_dim_per_item=768)
        self.point_light_align = PointLightAlign(input_feature_dim=1536, output_dim_per_item=12)
        self.area_light_align = AreaAlign(input_feature_dim=1536, output_dim_per_item=5)

        # --- LVC module (no learned parameters) ---
        self.lvc = LVCModule(alpha=lvc_alpha, w_min=lvc_w_min, per_channel=lvc_per_channel)

    def forward(self, data, decoder_resolution, canonical_resolution):
        I = data["img"].to(torch.bfloat16)          # [B, C, H, W, Nmax]
        N = data["nml"][:, :, :, :, 0].to(torch.bfloat16)
        M = data["mask"][:, :, :, :, 0].to(torch.bfloat16)
        env_light = data["env_light"].to(torch.bfloat16)
        point_lights = data["point_lights"].to(torch.bfloat16)
        area_light = data["area_light"].to(torch.bfloat16)
        nImgArray = data["numberOfImages"].reshape(-1, 1)

        decoder_resolution = np.int32(decoder_resolution)
        canonical_resolution = np.int32(canonical_resolution)
        B, C, H, W, Nmax = I.shape

        # ------------------------------------------------------------------
        # LVC: compute per-pixel confidence from raw images (float32)
        # ------------------------------------------------------------------
        with torch.no_grad():
            C_map = self.lvc.confidence_map(I.float())  # [B, H, W]
        # Interpolate to decoder resolution for pixel-level weighting
        C_dec = F.interpolate(
            C_map.unsqueeze(1), size=(decoder_resolution, decoder_resolution),
            mode='bilinear', align_corners=False,
        ).squeeze(1)  # [B, H_dec, W_dec]
        lvc_w_dec = self.lvc.loss_weights(C_dec)  # [B, H_dec, W_dec], in [w_min, 1]

        # ------------------------------------------------------------------
        # Image encoder (unchanged from baseline)
        # ------------------------------------------------------------------
        img = I.permute(0, 4, 1, 2, 3)
        img_index = make_index_list(Nmax, nImgArray)
        img = img.reshape(-1, img.shape[2], img.shape[3], img.shape[4])
        M_enc = M.unsqueeze(1).expand(-1, Nmax, -1, -1, -1).reshape(-1, 1, H, W)
        data_enc = img * M_enc
        data_enc = data_enc[img_index == 1, :, :, :]
        glc, light_tokens = self.image_encoder(data_enc, nImgArray, canonical_resolution)

        env_token = light_tokens[:, :, :, :, 0, :]
        point_lights_token = light_tokens[:, :, :, :, 1, :]
        area_lights_token = light_tokens[:, :, :, :, 2, :]

        env_feature = torch.zeros(B, nImgArray[0], 1024, 768).to(torch.bfloat16).to(glc.device)
        for i in range(nImgArray[0]):
            env_feature[:, i, :, :] = self.hdri_encoder(env_light[:, i])
        env_feature_project = self.env_feature_proj(env_feature)
        env_token_predict = self.env_light_head(env_token)
        loss_point_lights = self.point_light_align(point_lights_token, point_lights)
        loss_area_lights = self.area_light_align(area_lights_token, area_light)
        loss_align_env = 1 - F.cosine_similarity(
            F.normalize(env_token_predict, dim=-1),
            F.normalize(env_feature_project, dim=-1), dim=-1,
        ).mean()

        # ------------------------------------------------------------------
        # Pixel sampling
        # ------------------------------------------------------------------
        img_valid = img[img_index == 1, :, :, :]
        I_dec = F.interpolate(img_valid.float(), size=(decoder_resolution, decoder_resolution),
                              mode='bilinear', align_corners=False).to(torch.bfloat16)
        N_dec = F.interpolate(N.float(), size=(decoder_resolution, decoder_resolution),
                              mode='bilinear', align_corners=False).to(torch.bfloat16)
        M_dec = F.interpolate(M.float(), size=(decoder_resolution, decoder_resolution),
                              mode='nearest').to(torch.bfloat16)
        Gradient_dec = sobel_edge_map(N_dec)

        if self.glc_smoothing:
            f_scale = decoder_resolution // canonical_resolution
            smoothing = gauss_filter.gauss_filter(glc.shape[1], 10 * f_scale + 1, 1).to(glc.device)
            glc = smoothing(glc)

        p = 0
        ids_batch = []
        n_true_list, o_ids_list, glc_ids_list, gradient_ids_list = [], [], [], []
        lvc_weights_list = []

        for b in range(B):
            target = range(p, p + nImgArray[b])
            p = p + nImgArray[b]
            m_ = M_dec[b, :, :, :].reshape(-1, decoder_resolution * decoder_resolution).permute(1, 0)
            ids = np.nonzero(m_.cpu().numpy() > 0)[:, 0]
            ids = ids[np.random.permutation(len(ids))]
            idset = [ids[:self.pixel_samples]]
            o_ = I_dec[target, :, :, :].reshape(nImgArray[b], C, decoder_resolution * decoder_resolution).permute(2, 0, 1)
            n_true = F.normalize(N_dec[b, :, :, :].reshape(3, decoder_resolution * decoder_resolution).permute(1, 0),
                                 p=2, dim=-1).to(torch.bfloat16)
            gradient_n = Gradient_dec[b, :, :, :].reshape(1, decoder_resolution * decoder_resolution).permute(1, 0).to(torch.bfloat16)

            # LVC weights at sampled pixels
            lvc_w_flat = lvc_w_dec[b].reshape(-1)  # [H_dec * W_dec]

            for ids in idset:
                o_ids_list.append(o_[ids, :, :])
                glc_ids_list.append(glc[target, :, :, :].permute(2, 3, 0, 1).flatten(0, 1)[ids, :, :])
                n_true_list.append(n_true[ids, :])
                gradient_ids_list.append(gradient_n[ids, :])
                lvc_weights_list.append(lvc_w_flat[ids])  # [pixel_samples]
            ids_batch.append(ids)

        o_ids = torch.cat(o_ids_list, dim=0)
        glc_ids = torch.cat(glc_ids_list, dim=0)
        n_true = torch.stack(n_true_list, dim=0)
        gradient_ids = torch.stack(gradient_ids_list, dim=0)
        lvc_weights = torch.stack(lvc_weights_list, dim=0)  # [B, pixel_samples]

        o_ids = self.img_embedding(o_ids)
        x = o_ids + glc_ids
        glc_ids = self.glc_upsample(x)
        x = o_ids + glc_ids
        x = self.glc_aggregation(x)
        x_n, _, _, conf = self.regressor(x, len(ids_batch[0]))
        x_n = F.normalize(x_n, p=2, dim=-1)

        # ------------------------------------------------------------------
        # LVC-weighted loss (Version A)
        # per-pixel squared error weighted by lvc_weights, then normalised
        # ------------------------------------------------------------------
        mse = self.criterionL2(x_n, n_true)
        loss_gradient = self.criterionL2(conf.exp(), gradient_ids.exp()) * 3

        per_pixel_err = ((x_n - n_true) ** 2) * (1 + conf)   # [B, pixel_samples, 3]
        w = lvc_weights.unsqueeze(-1).to(per_pixel_err.dtype)  # [B, pixel_samples, 1]
        loss_conf = (per_pixel_err * w).sum() / (w.sum() * 3 + 1e-8)

        # NaN guards
        for name, t in [("loss_conf", loss_conf), ("loss_gradient", loss_gradient),
                        ("loss_align_env", loss_align_env), ("loss_point_lights", loss_point_lights),
                        ("loss_area_lights", loss_area_lights)]:
            if torch.isnan(t).any():
                print(f"{name} contains NaN — zeroed")
                t = torch.zeros_like(t) + 1e-6

        loss = (loss_conf
                + 0.1 * loss_gradient / (loss_gradient / loss_conf).detach()
                + 0.1 * loss_align_env / (loss_align_env / loss_conf).detach()
                + 0.1 * loss_point_lights / (loss_point_lights / loss_conf).detach()
                + 0.1 * loss_area_lights / (loss_area_lights / loss_conf).detach())
        return {
            'mse': mse,
            'loss': loss,
            'loss_gradient': loss_gradient,
            'loss_conf': loss_conf,
            'loss_align_env': loss_align_env,
            'loss_point_lights': loss_point_lights,
            'loss_area_lights': loss_area_lights,
        }


class LINO_UniPS_LVC_LossModule(pl.LightningModule):
    """Lightning wrapper for NetLVC_Loss (Version A)."""

    def __init__(self, net, optimizer_class, scheduler_class, canonical_resolution,
                 sample_num, save_dir, learning_rate=1e-4, weight_decay=0.05,
                 max_epochs=100, min_lr=1e-6, step_size=10, gamma=0.8):
        super().__init__()
        self.strict_loading = False
        self.save_hyperparameters(logger=False)
        self.canonical_resolution = canonical_resolution
        self.net = net
        self.sample_num = sample_num
        self.optimizer_class = optimizer_class
        self.scheduler_class = scheduler_class
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.step_size = step_size
        self.gamma = gamma
        self.criterion = nn.MSELoss(reduction='mean')
        self.save_dir = save_dir
        self.train_mae = MeanMetric()
        self.val_mae = MeanMetric()
        self.test_mae = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

    def forward(self, data, decoder_resolution, canonical_resolution):
        return self.net(data, decoder_resolution, canonical_resolution)

    def on_train_start(self):
        self.val_loss.reset()

    def model_step(self, batch):
        img = batch["img"]
        B, C, H, W, N = img.shape
        return self.forward(data=batch, decoder_resolution=H,
                            canonical_resolution=self.canonical_resolution)

    def training_step(self, batch, batch_idx):
        metric_dict = self.model_step(batch)
        loss = metric_dict['loss']
        self.train_loss(loss)
        self.lr = self.optimizers().param_groups[0]['lr']
        self.log("train/lr", self.lr, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss", self.train_loss(loss), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse_loss", metric_dict['mse'], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        metric_dict = self.model_step(batch)
        loss = metric_dict['loss']
        self.loss = loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mse_loss", metric_dict['mse'], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.optimizer_class(
            self.trainer.model.parameters(),
            lr=self.learning_rate, weight_decay=self.weight_decay,
        )
        if self.scheduler_class is not None:
            if self.scheduler_class.__name__ == 'StepLR':
                scheduler = self.scheduler_class(optimizer, step_size=self.step_size, gamma=self.gamma)
            else:
                scheduler = self.scheduler_class(optimizer, T_max=self.max_epochs, eta_min=self.min_lr)
            return {"optimizer": optimizer,
                    "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss",
                                     "interval": "epoch", "frequency": 1}}
        return {"optimizer": optimizer}
