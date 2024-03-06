import lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from constants import ALL_CHAR_SET_LEN, MAX_CAPTCHA
from accuracy import calcular_accuracy

class CaptchaModel_EfficentNet(pl.LightningModule):
    def __init__(self):
        super(CaptchaModel_EfficentNet, self).__init__()
        model_weights = models.EfficientNet_B0_Weights.DEFAULT
        self.efficent_net = models.efficientnet_b0(weights=model_weights)

        # dejamos el último bloque sin congelar mas la fc
        for name, layer in self.efficent_net.named_parameters():
            if "6" in name or "7" in name or "8" in name or "classifier" in name:
                layer.requires_grad = True
            else:
                layer.requires_grad = False

        self.efficent_net.classifier[1] = nn.Linear(in_features=1280, out_features=ALL_CHAR_SET_LEN*MAX_CAPTCHA, bias=True)

    def forward(self, x):
        return self.efficent_net(x)

    def training_step(self, batch, batch_idx):
        x, y, label = batch
        y_hat = self.efficent_net(x)
        loss_func = nn.MultiLabelSoftMarginLoss()
        loss = loss_func(y_hat, y)

        # Calcular la accuracy
        accuracy = calcular_accuracy(y_hat, label)
        # self.log_dict({'train_loss': loss.item()}) #'train_accuracy': accuracy})
        self.log("train_loss", loss.item())
        self.log("train_accuracy", accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, label = batch
        y_hat = self.efficent_net(x)
        loss_func = nn.MultiLabelSoftMarginLoss()
        loss = loss_func(y_hat, y)

        # Calcular la accuracy
        accuracy = calcular_accuracy(y_hat, label)

        # self.log_dict({'val_loss': loss.item()}) #'val_accuracy': accuracy})
        self.log("val_loss", loss.item())
        self.log("val_accuracy", accuracy)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)