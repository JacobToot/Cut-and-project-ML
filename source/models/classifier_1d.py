import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """Conv1d + Norm + Activation + Dropout, with optional residual."""

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1,
                 dropout=0.0, residual=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation // 2  
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              stride=stride, padding=self.pad, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()
        self.drop = nn.Dropout1d(p=dropout)

        self.residual = residual and (stride == 1) and (in_ch == out_ch)

    def forward(self, x):
        out = self.drop(self.act(self.norm(self.conv(x))))
        if self.residual:
            out = out + x 
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.SiLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):                       
        w = x.mean(dim=-1)                     
        w = self.fc(w).unsqueeze(-1)           
        return x * w


class Conv1DClassifier(nn.Module):

    def __init__(self, seq_len, classes, kernel_size=9, dropout_conv=0.1,
                 dropout_linear=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )

        self.encoder = nn.Sequential(
            Conv1DBlock(32, 64,  kernel_size, stride=2, dropout=dropout_conv),
            Conv1DBlock(64, 64,  kernel_size, stride=1, dropout=dropout_conv),
            Conv1DBlock(64, 128, kernel_size, stride=2, dropout=dropout_conv),
            Conv1DBlock(128, 128, kernel_size, stride=1, dropout=dropout_conv),
        )

        self.dilated = nn.Sequential(
            Conv1DBlock(128, 128, kernel_size, dilation=2,  dropout=dropout_conv),
            Conv1DBlock(128, 128, kernel_size, dilation=4,  dropout=dropout_conv),
            Conv1DBlock(128, 128, kernel_size, dilation=8,  dropout=dropout_conv),
            Conv1DBlock(128, 128, kernel_size, dilation=16, dropout=dropout_conv),
            Conv1DBlock(128, 128, kernel_size, dilation=32, dropout=dropout_conv),
        )

        self.se = SEBlock(128)

        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Dropout(p=dropout_linear),
            nn.Linear(128, classes),
        )

    def forward(self, x):                      
        x = self.stem(x)
        x = self.encoder(x)
        x = self.dilated(x)
        x = self.se(x)
        x = x.mean(dim=-1)                     
        x = self.head(x)
        return x