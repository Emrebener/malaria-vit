import timm
import torch.nn as nn

def create_model(name: str, pretrained: bool = True) -> nn.Module:
    """timm backbone with a fresh 2-class head."""
    return timm.create_model(name, pretrained=pretrained, num_classes=2)
