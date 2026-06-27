from pathlib import Path
from malaria_vit.data import build_image_index

def test_index_labels_and_patients(tmp_path):
    for sub, fname in [("Parasitized", "C1P1thinF_cell_0.png"),
                       ("Uninfected", "C2P2thinF_cell_0.png")]:
        d = tmp_path / sub; d.mkdir()
        (d / fname).write_bytes(b"x")  # content irrelevant for indexing
    df = build_image_index(tmp_path)
    assert len(df) == 2
    assert set(df["label"]) == {0, 1}
    assert set(df["patient_id"]) == {"C1", "C2"}  # parser returns the leading C-number
    # Enforce per-row label↔directory mapping (catch swapped labels)
    assert df[df["patient_id"] == "C1"].iloc[0]["label"] == 1   # Parasitized
    assert df[df["patient_id"] == "C2"].iloc[0]["label"] == 0   # Uninfected
