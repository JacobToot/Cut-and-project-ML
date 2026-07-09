import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class ResidualBlock(nn.Module):
    def __init__(self,
                 channels,
                 context_dim = None,
                 activation=F.relu,
                 dropout=0.0,
                 batch_norm=False,
                 zero_init=True,
                 ):
    
        super().__init__()
        self.channels = channels
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.batch_norm = batch_norm

        self.layers = nn.ModuleList([nn.Linear(channels,channels) for _ in range(2)])

        if context_dim is not None:
            self.context_layer = nn.Linear(context_dim, channels)

        if batch_norm:
            self.batch_norm_layer = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(2)])

        if zero_init:
            init.uniform_(self.layers[-1].weight, -1e-3, 1e-3)
            init.uniform_(self.layers[-1].bias, -1e-3, 1e-3)    

    def forward(self, x, context=None):
        t = x

        if self.batch_norm:
            t = self.batch_norm_layer[0](t)
        t = self.activation(t)
        t = self.layers[0](t)
        
        if self.batch_norm:
            t = self.batch_norm_layer[1](t)
        t = self.activation(t)
        t = self.dropout(t)
        t = self.layers[1](t)
        
        if context is not None:
            t = F.glu(torch.cat((t, self.context_layer(context)), dim=1), dim=1)
        return x + t

class ResNet(nn.Module):
    def __init__(self,
                 d_in,
                 d_out,
                 channels,
                 d_context = None,
                 activation=F.relu,
                 num_blocks=2,
                 dropout=0.0,
                 batch_norm=False,
                 ):
        
        super().__init__()

        self.channels = channels
        self.activation = activation

        self.initial_layer = nn.Sequential(nn.Linear(d_in,channels))

        if d_context is not None:
            self.context_layer = nn.Sequential(
                nn.Linear(d_in + d_context, channels)
            )
        else:
            self.context_layer = None

        self.final_layer = nn.Sequential(nn.Linear(channels, d_out))
        self.blocks = nn.ModuleList([
            ResidualBlock(channels,
                          d_context, 
                          activation, 
                          dropout, 
                          batch_norm)
                            for _ in range(num_blocks)])
        
    def forward(self, x, context=None):
        t = x
        if context is None:
            t = self.initial_layer(t)
        elif context is not None:
            t = self.context_layer(torch.cat((x, context),dim=1))
        for block in self.blocks:
            t = block(t, context)
        out = self.final_layer(t)
        return out

def build_rn(d_in,
             d_out,
             channels,
             d_context = None,
             activation=F.relu,
             num_blocks=2,
             dropout=0.0,
             batch_norm=False):
    
    return ResNet(d_in=d_in,
             d_out=d_out,
             channels=channels,
             d_context=d_context,
             activation=activation,
             num_blocks=num_blocks,
             dropout=dropout,
             batch_norm=batch_norm)