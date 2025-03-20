from torch.utils.data import Dataset
import torch
import pandas as pd

class TimeSeriesDataset(Dataset):

    def __init__(self, data, seq_len=48, label_len=24, pred_len=24):

        
        if isinstance(data, pd.DataFrame):
            self.df_columns = data.columns
            data = data.values
        else:
            self.df_columns = None
        
        self.data = data  # shape [N, D]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        
        self.n_samples = len(self.data)      # N
        self.n_features = self.data.shape[1] # D

    def __len__(self):
        return self.n_samples - (self.seq_len + self.pred_len) + 1

    def __getitem__(self, idx):
        enc_begin = idx
        enc_end = idx + self.seq_len  # Left-closed, right-open
        x_enc = self.data[enc_begin:enc_end, :]  # Shape: [seq_len, n_features]

        # Decoder input (past `label_len` + future `pred_len` time steps)
        dec_begin = enc_end - self.label_len
        dec_end = enc_end + self.pred_len
        x_dec = self.data[dec_begin:dec_end, :]  # Shape: [label_len + pred_len, n_features]

        # Target output (future `pred_len` time steps for all sensors)
        y_start = dec_end - self.pred_len
        y_end = dec_end
        y = self.data[y_start:y_end, :] 

        # Convert to PyTorch tensors
        x_enc = torch.tensor(x_enc, dtype=torch.float)
        x_dec = torch.tensor(x_dec, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)  

        return x_enc, x_dec, y

