"""Leakage control: retrain DeiT-S on a random image-level split.

Same model, hyperparameters and seed as the main run; only the split changes.
Cells are assigned to train/val/test at random (70/15/15, stratified by label),
so most patients end up on both sides of the split. Comparing this run's test
accuracy with the patient-level result measures how much leakage inflates it.

    uv run python scripts/leakage_control.py

Writes data/split_image_level.csv and outputs/deit_s_image_split/metrics.json.
"""
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from malaria_vit.config import DATA_DIR, MODELS, OUTPUTS_DIR, SEED, SPLIT_CSV, TrainConfig
from malaria_vit.data import MalariaDataModule
from malaria_vit.evaluate import evaluate_checkpoint, save_metrics
from malaria_vit.train import train_model

KEY = "deit_s"
RUN = "deit_s_image_split"
IMAGE_SPLIT_CSV = DATA_DIR / "split_image_level.csv"


def make_image_level_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """The textbook split: 70/15/15 over individual cells, ignoring patients."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=seed)
    out = df.copy()
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out.loc[part.index, "split"] = name
    return out


def overlap(split: pd.DataFrame) -> dict:
    """How much of the test set shares a patient or a source photo with training."""
    split = split.assign(field=split["patient_id"] + "/" + split["filepath"].str.extract(r"(IMG_\d+_\d+)")[0])
    train, test = split[split["split"] == "train"], split[split["split"] == "test"]
    return {
        "test_cells": len(test),
        "test_patients": test["patient_id"].nunique(),
        "test_patients_also_in_train": len(set(test["patient_id"]) & set(train["patient_id"])),
        "test_cells_whose_patient_is_in_train": int(test["patient_id"].isin(train["patient_id"]).sum()),
        "test_cells_whose_photo_is_in_train": int(test["field"].isin(train["field"]).sum()),
    }


def main() -> None:
    cells = pd.read_csv(SPLIT_CSV).drop(columns="split")
    split = make_image_level_split(cells)
    split.to_csv(IMAGE_SPLIT_CSV, index=False)
    leak = overlap(split)
    print("overlap:", json.dumps(leak), flush=True)

    ckpt = train_model(KEY, split_csv=IMAGE_SPLIT_CSV, run_name=RUN)
    metrics = evaluate_checkpoint(ckpt, MalariaDataModule(IMAGE_SPLIT_CSV, MODELS[KEY], TrainConfig()))
    save_metrics({**metrics, "overlap": leak, "checkpoint": str(ckpt)}, OUTPUTS_DIR / RUN / "metrics.json")

    patient = json.loads((OUTPUTS_DIR / KEY / "metrics.json").read_text())
    for k in ("accuracy", "f1", "roc_auc"):
        print(f"{k}: image-level {metrics[k]:.4f} vs patient-level {patient[k]:.4f}", flush=True)
    print("confusion_matrix:", metrics["confusion_matrix"], flush=True)


if __name__ == "__main__":
    main()
