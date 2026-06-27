from dataclasses import dataclass
from pathlib import Path

SEED = 42
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
SPLIT_CSV = DATA_DIR / "split.csv"
OUTPUTS_DIR = Path("outputs")

@dataclass(frozen=True)
class ModelConfig:
    name: str          # timm model name
    batch_size: int    # real batch sized for 6 GB VRAM
    accumulate: int    # gradient-accumulation steps
    lr: float = 3e-5
    strong_aug: bool = False

@dataclass(frozen=True)
class TrainConfig:
    max_epochs: int = 15
    patience: int = 3
    num_workers: int = 2      # keep host-RAM footprint low (14 GiB shared laptop w/ earlyoom)
    pin_memory: bool = False  # pinned RAM is non-swappable; off avoids tripping earlyoom
    precision: str = "bf16-mixed"

MODELS = {
    "deit_s":  ModelConfig("deit_small_patch16_224",      batch_size=32, accumulate=2, strong_aug=True),
    "swin_t":  ModelConfig("swin_tiny_patch4_window7_224", batch_size=24, accumulate=3),
    "vit_b16": ModelConfig("vit_base_patch16_224",        batch_size=12, accumulate=5),
}
