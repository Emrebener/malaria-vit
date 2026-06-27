import re
from pathlib import Path

import lightning as L
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import SEED

_PATIENT_RE = re.compile(r"(C\d+)")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def parse_patient_id(filename: str) -> str:
    """Patient identifier = the leading C-number of an NIH malaria filename.

    Filenames vary across the dataset (parasitised 'C39P4thinF...'; uninfected
    'C225ThinF...' with no P part; some 'C51AP12...'), but the leading C<number>
    reliably identifies the patient. This recovers exactly the dataset's 200 patients
    (150 infected + 50 healthy), with 150 contributing cells to BOTH classes, so
    grouping by it is conservative and leakage-safe.
    """
    m = _PATIENT_RE.match(filename)
    if m:
        return m.group(1)
    return filename.split("_")[0]  # fallback: leading token (no C-number present)

def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Patient-grouped, class-stratified 70/15/15 split. No patient spans splits."""
    patients = (df.groupby("patient_id")["label"]
                  .agg(lambda s: s.mode().iloc[0])      # representative class per patient
                  .reset_index())
    train_p, temp_p = train_test_split(
        patients, test_size=0.30, stratify=patients["label"], random_state=seed)
    val_p, test_p = train_test_split(
        temp_p, test_size=0.50, stratify=temp_p["label"], random_state=seed)
    split_of = {}
    for name, frame in [("train", train_p), ("val", val_p), ("test", test_p)]:
        for pid in frame["patient_id"]:
            split_of[pid] = name
    out = df.copy()
    out["split"] = out["patient_id"].map(split_of)
    return out

def build_image_index(raw_dir: Path) -> pd.DataFrame:
    """Scan raw_dir/{Parasitized,Uninfected} for .png images -> dataframe."""
    rows = []
    for label, sub in [(1, "Parasitized"), (0, "Uninfected")]:
        for p in sorted((Path(raw_dir) / sub).glob("*.png")):
            rows.append({"filepath": str(p),
                         "patient_id": parse_patient_id(p.name),
                         "label": label})
    return pd.DataFrame(rows)

def write_split(raw_dir: Path, out_csv: Path, seed: int = SEED) -> pd.DataFrame:
    df = make_split(build_image_index(raw_dir), seed=seed)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df

def build_transforms(train: bool, strong_aug: bool = False):
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return transforms.Compose([transforms.Resize((224, 224)), *norm])
    j = 0.3 if strong_aug else 0.1
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=j, contrast=j, saturation=j),
        *norm,
    ])

class MalariaDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["filepath"]).convert("RGB")
        return self.transform(img), int(row["label"])

class MalariaDataModule(L.LightningDataModule):
    def __init__(self, split_csv, model_cfg, train_cfg):
        super().__init__()
        self.split_csv = split_csv
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
    def setup(self, stage=None):
        df = pd.read_csv(self.split_csv)
        self.train_ds = MalariaDataset(df[df["split"] == "train"],
                                       build_transforms(True, self.model_cfg.strong_aug))
        self.val_ds = MalariaDataset(df[df["split"] == "val"], build_transforms(False))
        self.test_ds = MalariaDataset(df[df["split"] == "test"], build_transforms(False))
    def _loader(self, ds, shuffle):
        return DataLoader(ds, batch_size=self.model_cfg.batch_size, shuffle=shuffle,
                          num_workers=self.train_cfg.num_workers, pin_memory=self.train_cfg.pin_memory)
    def train_dataloader(self): return self._loader(self.train_ds, True)
    def val_dataloader(self):   return self._loader(self.val_ds, False)
    def test_dataloader(self):  return self._loader(self.test_ds, False)
