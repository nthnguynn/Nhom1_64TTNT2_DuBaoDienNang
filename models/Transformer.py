import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math

# Positional encoding for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Map input features to d_model dimension
        self.input_fc = nn.Linear(input_size, d_model)
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        # Final projection to scalar output
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # 1) project to d_model
        x = self.input_fc(x)            # (batch, seq_len, d_model)
        # 2) prepare for transformer: (seq_len, batch, d_model)
        x = x.permute(1, 0, 2)
        # 3) add positional encoding
        x = self.pos_encoder(x)
        # 4) transformer encode
        output = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        # 5) take last time step's representation
        last = output[-1, :, :]               # (batch, d_model)
        # 6) decode to scalar
        y = self.decoder(last)                # (batch, 1)
        return y.squeeze(-1)                  # (batch,)