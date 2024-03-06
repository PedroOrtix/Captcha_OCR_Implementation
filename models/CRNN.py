import lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from constants import ALL_CHAR_SET_LEN, MAX_CAPTCHA
from accuracy import calcular_accuracy

class CaptchaModel_ConvLSTM(pl.LightningModule):
    def __init__(self):
        super(CaptchaModel_ConvLSTM, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 6), stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 6), stride=1, padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )
        self.fc1 = nn.Sequential(nn.Linear(in_features=1152, out_features=64, bias=True),
                                nn.ReLU())
        self.drop = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=2, batch_first=True, bidirectional=True, dropout=0.25)
        self.fc2 = nn.Linear(in_features=64, out_features=ALL_CHAR_SET_LEN*MAX_CAPTCHA, bias=True)
        
    def forward(self, x):
        batch, _, _, _ = x.size()
        # use only the conv and the fc
        x = self.conv_block(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(batch, x.size(1), -1)
        x = self.fc1(x)
        x = self.drop(x)
        x, _ = self.lstm(x)
        x = self.fc2(x)
        x = x.permute(1, 0, 2)
        
        return x
        
    
    def training_step(self, batch, batch_idx):
        x, y, label = batch
        y_hat = self(x)
        loss_func = nn.MultiLabelSoftMarginLoss()
        loss = loss_func(y_hat, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y, label = batch
        y_hat = self(x)
        loss_func = nn.MultiLabelSoftMarginLoss()
        loss = loss_func(y_hat, y)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)