from malaria_vit.config import MODELS, TrainConfig
from malaria_vit.data import MalariaDataModule

def test_datamodule_yields_correct_batch(tiny_split_csv):
    dm = MalariaDataModule(tiny_split_csv, MODELS["deit_s"], TrainConfig(num_workers=0))
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape[1:] == (3, 224, 224)
    assert x.shape[0] == y.shape[0]
