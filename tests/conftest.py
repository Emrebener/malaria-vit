import pandas as pd
import pytest
from PIL import Image

@pytest.fixture
def index_df():
    """40 patients (20 uninfected, 20 parasitised), 5 cells each."""
    rows = []
    for label in (0, 1):
        for k in range(20):
            pid = f"C{label}{k:02d}P{k}"
            for cell in range(5):
                rows.append({"filepath": f"/fake/{pid}_cell_{cell}.png",
                             "patient_id": pid, "label": label})
    return pd.DataFrame(rows)

@pytest.fixture
def tiny_split_csv(tmp_path):
    """Real tiny PNGs + a split.csv referencing them (>=2 per split, both classes)."""
    rows = []
    plan = [("train", 4), ("val", 2), ("test", 2)]
    for split, n in plan:
        for i in range(n):
            label = i % 2
            pid = f"{split}P{i}"
            fp = tmp_path / f"{split}_{i}.png"
            Image.new("RGB", (64, 64), (label * 200, 0, 0)).save(fp)
            rows.append({"filepath": str(fp), "patient_id": pid,
                         "label": label, "split": split})
    csv = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv
