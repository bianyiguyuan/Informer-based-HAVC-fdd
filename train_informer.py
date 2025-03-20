import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, random_split
from timeseries_dataset import TimeSeriesDataset
from model import Informer

BATCH_SIZE = 2700

DATASET_DIR = "LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/"
FAULT_PATH = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) 
               if f.endswith(".csv") and "AHU_annual" not in f]

def load_and_preprocess(csv_file, features):
    df = pd.read_csv(csv_file)
    df = df[features]
    df['Timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='T')
    df.set_index('Timestamp', inplace=True)

    df['month_of_year'] = (df.index.month - 1) / 11.0  
    df['day_of_month'] = (df.index.day - 1) / 30.0  
    df['weekday'] = df.index.dayofweek / 6.0  
    df['hour_of_day'] = df.index.hour / 23.0  
    df['minute_of_hour'] = df.index.minute / 59.0  

    df = df.fillna(method='ffill').fillna(method='bfill')

    # Normalize data
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    return df_scaled, scaler


def train_informer_main():
    # 1) 指定数据文件和特征列
    csv_file = 'LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/AHU_annual.csv'  
    def get_features_from_csv(file_path):

        df = pd.read_csv(file_path, nrows=5)
        features = list(df.columns)
        time_columns = ["timestamp", "time", "datetime", "date"] 
        features = [col for col in features if col.lower() not in time_columns]

        return features

    features = get_features_from_csv(csv_file)

    # 2) 加载并预处理数据
    df_scaled, _ = load_and_preprocess(csv_file, features)
    print("Data shape after scaling:", df_scaled.shape)
    
    # 3) 构建 Dataset
    seq_len = 48
    label_len = 24
    pred_len = 24
    
    dataset = TimeSeriesDataset(
        data=df_scaled, 
        seq_len=seq_len,
        label_len=label_len,
        pred_len=pred_len
    )
    
    # 4) 划分训练集、验证集 (80% 训练, 20% 验证)
    total_size = len(dataset)
    train_size = int(total_size * 0.8)
    val_size = total_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

    
    # 5) 实例化 Informer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Device Count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current CUDA Device:", torch.cuda.current_device())
        print("CUDA Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    n_features = len(features) + 5  # 预测所有传感器
    
    model = Informer(
        enc_in=n_features,  # 输入特征数
        dec_in=n_features,  # Decoder 也使用相同特征数
        c_out=n_features,   # 预测所有传感器
        seq_len=seq_len,
        label_len=label_len,
        out_len=pred_len,   # 预测步长
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_layers=2,
        d_ff=512,
        dropout=0.05,
        attn='prob', 
        embed='fixed', 
        freq='h',   
        activation='gelu',
        output_attention=False,
        distil=True,
        mix=True,
        device=device
    ).to(device)
    
    # 6) 定义损失与优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 7) 训练循环
    epochs = 5
    for epoch in range(epochs):
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
        print(train_loader)

        if not check_none_values(train_loader):
            print("🚨 Training aborted due to None values in DataLoader!")
            return  # 终止训练，避免错误传播

        model.train()
        train_loss_sum = 0.0
        for i, (x_enc, x_dec, y) in enumerate(train_loader):
            x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
            # 提取时间信息（假设最后两列是时间特征）
            x_mark_enc = x_enc[:, :, -5:].to(device)  # 取 5 维时间特征
            x_mark_dec = x_dec[:, :, -5:].to(device)  # 取 5 维时间特征


            preds = model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # 传入时间信息
            
            loss = criterion(preds, y)  # 计算所有传感器的预测误差
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()

            print(f"Epoch {epoch+1}, Batch {i+1}: Loss={loss.item():.6f}")
        
        train_loss_avg = train_loss_sum / len(train_loader)
        print(f"Epoch {epoch+1} - Total Train Loss: {train_loss_sum:.4f}, Average: {train_loss_avg:.4f}")
        
        # 验证
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for j, (x_enc, x_dec, y) in enumerate(val_loader):
                x_enc, x_dec, y = x_enc.to(device), x_dec.to(device), y.to(device)
                x_mark_enc = x_enc[:, :, -5:]
                x_mark_dec = x_dec[:, :, -5:]
                preds = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss_val = criterion(preds, y)
                val_loss_sum += loss_val.item()
                print(f"Validation Batch {j+1}: Loss={loss_val.item():.6f}")

        val_loss_avg = val_loss_sum / len(val_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss={train_loss_avg:.4f}, Val Loss={val_loss_avg:.4f}")
    
    # 8) 训练结束，可保存模型
    torch.save(model.state_dict(), "informer_sdahu_"+ str(BATCH_SIZE) +".pth")
    print("Model saved: informer_sdahu_"+ str(BATCH_SIZE) +".pth")

def check_none_values(data_loader):
    print("Checking DataLoader for None values...")
    for batch_idx, (x_enc, x_dec, y) in enumerate(data_loader):
        if x_enc is None or x_dec is None or y is None:
            print(f"None value detected in batch {batch_idx}!")
            return False
    print("DataLoader check passed, no None values detected.")
    return True

if __name__ == "__main__":
    train_informer_main()
