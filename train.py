from torch.utils.data import DataLoader
from src.data.data_train_module import TrainData
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import argparse
from src.models.Net_train_module import Net, LINO_UniPSModule
from src.models.Net_lvc_loss import NetLVC_Loss, LINO_UniPS_LVC_LossModule
from src.models.Net_lvc_feat import NetLVC_Feat, LINO_UniPS_LVC_FeatModule
from src.models.Net_lvc_full import NetLVC_Full, LINO_UniPS_LVC_FullModule
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Map --variant string to (Net class, LightningModule class)
VARIANT_MAP = {
    "baseline": (Net,          LINO_UniPSModule),
    "lvc_loss": (NetLVC_Loss,  LINO_UniPS_LVC_LossModule),
    "lvc_feat": (NetLVC_Feat,  LINO_UniPS_LVC_FeatModule),
    "lvc_full": (NetLVC_Full,  LINO_UniPS_LVC_FullModule),
}

def train_model(args):
    train_data = TrainData(
        mode='Train',
        data_root=args.data_root,
        low_normal=args.low_normal
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_data = TrainData(
        mode='Val',
        data_root=args.data_root,
        low_normal=args.low_normal
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ------------------------------------------------------------------ #
    # Select architecture variant                                          #
    # ------------------------------------------------------------------ #
    if args.variant not in VARIANT_MAP:
        raise ValueError(
            f"Unknown variant '{args.variant}'. "
            f"Choose from: {list(VARIANT_MAP.keys())}"
        )
    NetClass, ModuleClass = VARIANT_MAP[args.variant]

    # LVC-enabled variants accept extra hyper-parameters; baseline ignores them
    lvc_kwargs = {}
    if args.variant != "baseline":
        lvc_kwargs = dict(
            lvc_alpha=args.lvc_alpha,
            lvc_w_min=args.lvc_w_min,
            lvc_per_channel=args.lvc_per_channel,
        )

    net = NetClass(
        pixel_samples=args.pixel_samples,
        output="normal",
        depth=args.depth,
        **lvc_kwargs,
    )

    model = ModuleClass(
        net=net,
        optimizer_class=optim.AdamW,
        scheduler_class=StepLR,
        canonical_resolution=args.canonical_resolution,
        sample_num=args.pixel_samples,
        save_dir=args.save_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        min_lr=args.min_lr,
        step_size=args.step_size,
        gamma=args.gamma
    )

    # Use the variant name in the checkpoint filename so runs don't collide
    ckpt_filename = f"{args.variant}_{{epoch:02d}}_{{val_loss:.4f}}"

    trainer = pl.Trainer(
        accelerator="auto",
        devices=args.devices,
        precision="bf16-mixed",
        max_epochs=args.max_epochs,
        val_check_interval=args.val_check_interval,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor="val/loss",
                dirpath=args.save_dir,
                filename=ckpt_filename,
                save_top_k=3,
                mode="min"
            ),
            pl.callbacks.EarlyStopping(
                monitor="val/loss",
                patience=args.patience,
                mode="min"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        ],
        logger=pl.loggers.TensorBoardLogger(
            save_dir=args.save_dir,
            name=f"lightning_logs/{args.variant}"
        )
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiNO UniPS Training Script")

    # ------------------------------------------------------------------ #
    # Variant selection                                                    #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--variant",
        type=str,
        default="baseline",
        choices=list(VARIANT_MAP.keys()),
        help=(
            "Which model variant to train. "
            "'baseline' = original LiNo-UniPS; "
            "'lvc_loss' = Version A (LVC loss weighting only); "
            "'lvc_feat' = Version B (LVC feature scaling only); "
            "'lvc_full' = Version C (full LVC, proposed method)."
        )
    )

    # ------------------------------------------------------------------ #
    # LVC hyper-parameters (only used when variant != baseline)           #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--lvc_alpha",
        type=float,
        default=10.0,
        help="LVC sensitivity parameter alpha (default 10.0)"
    )
    parser.add_argument(
        "--lvc_w_min",
        type=float,
        default=0.1,
        help="LVC minimum loss weight w_min (default 0.1)"
    )
    parser.add_argument(
        "--lvc_per_channel",
        action="store_true",
        default=True,
        help=(
            "LVC CV² computation mode. "
            "If set (default): take max CV² across RGB channels (per_channel=True). "
            "Use --no-lvc_per_channel to average over channels (grayscale mode)."
        )
    )
    parser.add_argument("--no-lvc_per_channel", dest="lvc_per_channel", action="store_false")

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--low_normal",
        type=bool,
        default=True,
        help="Low normal mode or high normal mode"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="Root directory of the dataset"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=6,
        help="Number of images to process"
    )

    # ------------------------------------------------------------------ #
    # Model                                                                #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--pixel_samples",
        type=int,
        default=2048,
        help="Number of pixel samples for training"
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="Depth of the network, default is 4"
    )
    parser.add_argument(
        "--canonical_resolution",
        type=int,
        default=256,
        help="Canonical resolution for processing"
    )

    # ------------------------------------------------------------------ #
    # Optimiser / scheduler                                                #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="Minimum learning rate"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="Step size for StepLR scheduler"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Gamma for StepLR scheduler"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience"
    )

    # ------------------------------------------------------------------ #
    # Infrastructure                                                       #
    # ------------------------------------------------------------------ #
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints/",
        help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--val_check_interval",
        type=float,
        default=1.0,
        help="Validation check interval (epochs)"
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=50,
        help="Log every n steps"
    )

    args = parser.parse_args()

    seed_everything(seed=args.seed, workers=True)
    train_model(args)
