import lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from constants import ALL_CHAR_SET_LEN, MAX_CAPTCHA
from accuracy import calcular_accuracy

class CRNN(nn.Module):

    def __init__(self, img_channel, img_height, img_width, num_class,
                map_to_seq_hidden=64, rnn_hidden=256, leaky_relu=False):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1)),

            nn.Conv2d(512, 512, kernel_size=2, stride=1),
            nn.ReLU(inplace=True) if not leaky_relu else nn.LeakyReLU(0.2, inplace=True)
        )

        # Calculating output dimensions after CNN layers
        output_height = img_height // 16 - 1
        output_width = img_width // 4 - 1
        output_channel = 512

        self.map_to_seq = nn.Linear(output_channel * output_height, map_to_seq_hidden)

        self.rnn1 = nn.LSTM(map_to_seq_hidden, rnn_hidden, bidirectional=True)
        self.rnn2 = nn.LSTM(2 * rnn_hidden, rnn_hidden, bidirectional=True)

        self.dense = nn.Linear(2 * rnn_hidden, num_class)

    def forward(self, images):
        # shape of images: (batch, channel, height, width)

        conv = self.cnn(images)
        batch, channel, height, width = conv.size()

        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)  # (width, batch, feature)
        seq = self.map_to_seq(conv)

        recurrent, _ = self.rnn1(seq)
        recurrent, _ = self.rnn2(recurrent)

        output = self.dense(recurrent)
        return output  # shape: (seq_len, batch, num_class)


class CaptchaModel_ConvLSTM(pl.LightningModule):
    def __init__(self):
        super(CaptchaModel_ConvLSTM, self).__init__()
        self.CRNN = CRNN(3, 256, 256, 378)
        
    def forward(self, x):
        x = self.CRNN(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y_oh, label = batch
        logits = self(x)
        # print(x.shape)
        # print(logits.shape)
        # tensor con los indices de los caracteres
        targets = torch.LongTensor([int(i) for i in label[0]]) # label es una tupla del tipo (str, )
        
        # definimos el CTC loss
        loss_fun = nn.CTCLoss(reduction='mean')
        
        # definimos las variables que necesita el CTC loss
        log_probs = nn.functional.log_softmax(logits, dim=2)
        batch_size = x.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.LongTensor([6] * batch_size)
        
        loss = loss_fun(log_probs, targets, input_lengths, target_lengths)
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y_oh, label = batch
        logits = self(x)
        print("Shape of input: ", x.shape)
        print("Shape of logits: ", logits.shape)
        print("Label: ", label, type(label))
        # tensor con los indices de los caracteres
        targets = torch.LongTensor([int(i) for i in label[0]]) # label es una tupla del tipo (str, )
        print("Targets: ", targets)
        # definimos el CTC loss
        loss_fun = nn.CTCLoss(reduction='mean')
        
        # definimos las variables que necesita el CTC loss
        log_probs = nn.functional.log_softmax(logits, dim=2)
        batch_size = x.size(0)
        print("Batch size: ", x.size(0))
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.LongTensor([6] * batch_size)
        print("Input lengths: ", input_lengths)
        print("Target lengths: ", target_lengths)
        
        loss = loss_fun(log_probs, targets, input_lengths, target_lengths)
        self.log('val_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)