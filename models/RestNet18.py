import lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from constants import ALL_CHAR_SET_LEN, MAX_CAPTCHA


class CaptchaModel_RestNet18(pl.LightningModule):
    def __init__(self):
        super(CaptchaModel_RestNet18, self).__init__()
        model_weights = models.ResNet18_Weights.DEFAULT
        self.restnet18 = models.resnet18(weights=model_weights)

        # congelamos los tres primeros bloques
        # dejamos el último bloque sin congelar
        for name, layer in self.restnet18.named_parameters():
            if "layer4" in name:
                layer.require_grad = True
            else:
                layer.require_grad = False

        # self.restnet18.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.restnet18.fc = nn.Linear(in_features=512, out_features=ALL_CHAR_SET_LEN*MAX_CAPTCHA, bias=True)

    def forward(self, x):
        return self.restnet18(x)

    def training_step(self, batch, batch_idx):
        x, y, label = batch
        y_hat = self.restnet18(x)
        loss_func = nn.MultiLabelSoftMarginLoss()
        loss = loss_func(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, label = batch
        y_hat = self.restnet18(x)
        loss_func = nn.MultiLabelSoftMarginLoss()
        loss = loss_func(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
