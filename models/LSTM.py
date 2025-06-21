import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM dự báo univariate từ multivariate input.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)  # output univariate

    def forward(self, x):
        # x: (batch, seq_length, input_size)
        out, _ = self.lstm(x)              # out: (batch, seq_length, hidden_size)
        last = out[:, -1, :]               # lấy hidden state của bước cuối
        y = self.fc(last)                  # y: (batch, 1)
        return y.squeeze(-1)               # (batch,)