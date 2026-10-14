import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from .config import MODELS, TrainConfig, SEED, SPLIT_CSV, OUTPUTS_DIR
from .data import MalariaDataModule
from .lit_module import LitClassifier

def train_model(model_key, split_csv=SPLIT_CSV, pretrained=True, fast_dev_run=False,
                run_name=None) -> str:
    """run_name: output folder under outputs/ (defaults to model_key)."""
    L.seed_everything(SEED, workers=True)
    out_dir = OUTPUTS_DIR / (run_name or model_key)
    mcfg, tcfg = MODELS[model_key], TrainConfig()
    nw = 0 if fast_dev_run else tcfg.num_workers
    tcfg = TrainConfig(max_epochs=tcfg.max_epochs, patience=tcfg.patience,
                       num_workers=nw, precision=tcfg.precision)
    dm = MalariaDataModule(split_csv, mcfg, tcfg)
    lit = LitClassifier(mcfg.name, lr=mcfg.lr, pretrained=pretrained)
    ckpt = ModelCheckpoint(dirpath=out_dir / "ckpts",
                           monitor="val_loss", mode="min", save_top_k=1, filename="best")
    early = EarlyStopping(monitor="val_loss", mode="min", patience=tcfg.patience)
    trainer = L.Trainer(
        max_epochs=tcfg.max_epochs, precision=tcfg.precision,
        accumulate_grad_batches=mcfg.accumulate, accelerator="auto", devices=1,
        deterministic="warn", callbacks=[ckpt, early],
        logger=CSVLogger(out_dir, name="logs"),
        fast_dev_run=fast_dev_run,
    )
    trainer.fit(lit, dm)
    return ckpt.best_model_path
