from malaria_vit.train import train_model

def test_train_runs_one_dev_batch(tiny_split_csv):
    # fast_dev_run runs a single train/val batch end to end without saving.
    path = train_model("deit_s", split_csv=tiny_split_csv,
                       pretrained=False, fast_dev_run=True)
    assert isinstance(path, str)
