import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from .lit_module import LitClassifier
from .metrics import compute_metrics

@torch.no_grad()
def evaluate_checkpoint(ckpt_path, datamodule) -> dict:
    lit = LitClassifier.load_from_checkpoint(ckpt_path, pretrained=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit.eval().to(device)
    datamodule.setup("test")
    probs, preds, labels = [], [], []
    for x, y in datamodule.test_dataloader():
        logits = lit(x.to(device))
        probs += torch.softmax(logits, 1)[:, 1].cpu().tolist()
        preds += logits.argmax(1).cpu().tolist()
        labels += y.tolist()
    return compute_metrics(labels, preds, probs)

def save_metrics(metrics: dict, out_json: Path):
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(metrics, indent=2))

def save_confusion_matrix(cm, out_png: Path, title: str = ""):
    cm = np.array(cm)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["uninfected", "parasitised"])
    ax.set_yticks([0, 1], ["uninfected", "parasitised"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
