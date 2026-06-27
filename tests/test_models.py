import torch
from malaria_vit.models import create_model

def test_outputs_two_logits():
    model = create_model("deit_small_patch16_224", pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)
