from PIL import Image
from malaria_vit.data import build_transforms

def test_eval_transform_shape():
    img = Image.new("RGB", (130, 90))
    out = build_transforms(train=False)(img)
    assert tuple(out.shape) == (3, 224, 224)

def test_train_transform_shape():
    img = Image.new("RGB", (130, 90))
    out = build_transforms(train=True, strong_aug=True)(img)
    assert tuple(out.shape) == (3, 224, 224)
