from model import Informer
import pandas as pd
import os

NORMAL_PATH = "LBNL_FDD_Dataset_SDAHU_all\LBNL_FDD_Dataset_SDAHU\AHU_annual.csv"
DATASET_DIR = "LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/"
FAULT_PATH = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) 
               if f.endswith(".csv") and "AHU_annual" not in f]

def get_features_from_csv(file_path):

    df = pd.read_csv(file_path, nrows=5)
    features = list(df.columns)
    time_columns = ["timestamp", "time", "datetime", "date"] 
    features = [col for col in features if col.lower() not in time_columns]

    return features

features = get_features_from_csv(NORMAL_PATH)

