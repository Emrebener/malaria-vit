"""Aggregate per-model results into a comparison table + figures.

Reads outputs/<key>/metrics.json (produced by train_eval.py) and writes the
results artefacts into docs/: comparison.csv and figures/*.png.

    uv run python scripts/build_results.py
"""
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from malaria_vit.config import MODELS, OUTPUTS_DIR  # noqa: E402

LABELS = {"deit_s": "DeiT-S", "swin_t": "Swin-T", "vit_b16": "ViT-B/16"}
DOCS = Path("docs")
FIGS = DOCS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

rows = []
for key in MODELS:
    m = json.loads((OUTPUTS_DIR / key / "metrics.json").read_text())
    eff = m["efficiency"]
    rows.append({
        "model": LABELS[key],
        "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "roc_auc": m["roc_auc"],
        "params_M": eff["params_millions"], "latency_ms": eff["latency_ms"], "gflops": eff["gflops"],
    })
    cm = OUTPUTS_DIR / key / "cm.png"
    if cm.exists():
        shutil.copy(cm, FIGS / f"cm_{key}.png")

df = pd.DataFrame(rows)
df.to_csv(DOCS / "comparison.csv", index=False)
print(df.round(4).to_string(index=False))

# Predictive-performance bars
fig, ax = plt.subplots(figsize=(7, 4))
mets = ["accuracy", "f1", "roc_auc"]
w, x = 0.25, range(len(df))
for i, mt in enumerate(mets):
    ax.bar([xi + i * w for xi in x], df[mt], width=w, label=mt)
ax.axhline(0.89, ls="--", color="grey", label="CNN patient-level ~0.89 (Laghari 2025)")
ax.set_xticks([xi + w for xi in x])
ax.set_xticklabels(df["model"])
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("score")
ax.set_title("Predictive performance (held-out, patient-level)")
ax.legend(fontsize=8, loc="lower right")
fig.savefig(FIGS / "performance.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# Accuracy vs efficiency
for xcol, xlabel, fname, color in [
    ("params_M", "Parameters (millions)", "accuracy_vs_params.png", "tab:blue"),
    ("latency_ms", "Inference latency (ms/image)", "accuracy_vs_latency.png", "tab:green"),
]:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df[xcol], df["accuracy"], s=90, color=color)
    for _, r in df.iterrows():
        ax.annotate(r["model"], (r[xcol], r["accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs {xlabel.split(' (')[0].lower()}")
    fig.savefig(FIGS / fname, bbox_inches="tight", dpi=150)
    plt.close(fig)

print("\nWrote docs/comparison.csv and figures to", FIGS)
