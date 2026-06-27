import torch
from malaria_vit.lit_module import LitClassifier

def test_training_step_returns_scalar_loss():
    lit = LitClassifier("deit_small_patch16_224", pretrained=False)
    batch = (torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))
    loss = lit.training_step(batch, 0)
    assert loss.ndim == 0 and loss.requires_grad

def test_configure_optimizers_is_adamw():
    lit = LitClassifier("deit_small_patch16_224", pretrained=False)
    assert isinstance(lit.configure_optimizers(), torch.optim.AdamW)
