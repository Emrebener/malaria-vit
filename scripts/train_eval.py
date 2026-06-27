"""Train + evaluate one or more ViT models and persist their results.

Usage:
    uv run python scripts/train_eval.py            # all models in MODELS
    uv run python scripts/train_eval.py deit_s     # just DeiT-S (walking skeleton)

For each model: fine-tunes via train_model, evaluates the best checkpoint on the
held-out test set, and writes outputs/<key>/metrics.json + cm.png. Robust to a
single model failing (e.g. OOM) — it logs the error and continues with the rest.
"""
import json
import sys
import traceback

import torch

from malaria_vit.config import MODELS, TrainConfig, SPLIT_CSV, OUTPUTS_DIR
from malaria_vit.data import MalariaDataModule
from malaria_vit.evaluate import evaluate_checkpoint, save_confusion_matrix, save_metrics
from malaria_vit.efficiency import efficiency_report
from malaria_vit.models import create_model
from malaria_vit.train import train_model


def run(key: str) -> None:
    print(f"\n========== {key}: TRAIN ==========", flush=True)
    ckpt = train_model(key)
    print(f"{key}: best checkpoint = {ckpt!r}", flush=True)

    print(f"========== {key}: EVAL ==========", flush=True)
    dm = MalariaDataModule(SPLIT_CSV, MODELS[key], TrainConfig())
    metrics = evaluate_checkpoint(ckpt, dm)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eff = efficiency_report(create_model(MODELS[key].name, pretrained=False), device=dev)

    out = OUTPUTS_DIR / key
    save_metrics({**metrics, "efficiency": eff, "checkpoint": str(ckpt)}, out / "metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], out / "cm.png", title=key)

    perf = {k: round(metrics[k], 4) for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    print(f"{key}: PERF {json.dumps(perf)}", flush=True)
    print(f"{key}: EFFICIENCY {json.dumps(eff)}", flush=True)
    print(f"{key}: confusion_matrix {metrics['confusion_matrix']}", flush=True)


if __name__ == "__main__":
    keys = sys.argv[1:] or list(MODELS)
    for key in keys:
        try:
            run(key)
        except Exception:
            print(f"!!! {key}: FAILED", flush=True)
            traceback.print_exc()
    print("\n=== DONE ===", flush=True)
