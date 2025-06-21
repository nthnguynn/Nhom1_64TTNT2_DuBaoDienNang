import torch
from torch.utils.data import Dataset
import pandas as pd
class TimeSeriesDataset(Dataset):
    """
    Dataset cho multivariate input và univariate target.
    Input: cửa sổ seq_length của tất cả features.
    Target: giá trị target_feature tại bước tiếp theo.
    """
    def __init__(self, df: pd.DataFrame, seq_length: int, target_col: str):
        self.features = df.values.astype('float32')  # toàn bộ cột làm input
        self.seq_length = seq_length
        self.target_idx = df.columns.get_loc(target_col)

    def __len__(self):
        # số mẫu: tổng độ dài - seq_length
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        # x: seq_length bước cho tất cả features    
        x = self.features[idx : idx + self.seq_length]
        # y: giá trị của target_col ngay sau cửa sổ x
        y = self.features[idx + self.seq_length, self.target_idx]
        # đưa về tensor
        return (
            torch.from_numpy(x),             # shape: (seq_length, num_features)
            torch.tensor(y, dtype=torch.float32)  # shape: () scalar
        )
