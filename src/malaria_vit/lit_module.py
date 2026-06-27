import lightning as L
import torch
import torch.nn as nn
import torchmetrics
from .models import create_model

class LitClassifier(L.LightningModule):
    def __init__(self, model_name: str, lr: float = 3e-5, pretrained: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_model(model_name, pretrained=pretrained)
        self.loss_fn = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.loss_fn(logits, y), prog_bar=True)
        self.val_acc(logits, y)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
