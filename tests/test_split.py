from malaria_vit.data import make_split

def test_no_patient_leaks_across_splits(index_df):
    out = make_split(index_df, seed=42)
    per_patient_splits = out.groupby("patient_id")["split"].nunique()
    assert (per_patient_splits == 1).all()  # each patient in exactly one split

def test_every_row_assigned(index_df):
    out = make_split(index_df, seed=42)
    assert set(out["split"].unique()) == {"train", "val", "test"}
    assert out["split"].isna().sum() == 0

def test_split_is_deterministic(index_df):
    a = make_split(index_df, seed=42)["split"].tolist()
    b = make_split(index_df, seed=42)["split"].tolist()
    assert a == b

def test_proportions_roughly_70_15_15(index_df):
    out = make_split(index_df, seed=42)
    frac = out["split"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.12
    assert abs(frac["val"] - 0.15) < 0.10
    assert abs(frac["test"] - 0.15) < 0.10
