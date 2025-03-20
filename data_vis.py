import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load normal data
normal_path = "LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/AHU_annual.csv"
df_normal = pd.read_csv(normal_path)

# Fault files grouped by type
fault_files = {
    "coi_bias": [
        "coi_bias_-2_annual.csv", "coi_bias_-4_annual.csv", "coi_bias_2_annual.csv", "coi_bias_4_annual.csv"
    ],
    "coi_leakage": [
        "coi_leakage_010_annual.csv", "coi_leakage_025_annual.csv", "coi_leakage_040_annual.csv", "coi_leakage_050_annual.csv"
    ],
    "coi_stuck": [
        "coi_stuck_010_annual.csv", "coi_stuck_025_annual.csv", "coi_stuck_050_annual.csv", "coi_stuck_075_annual.csv"
    ],
    "damper_stuck": [
        "damper_stuck_010_annual.csv", "damper_stuck_025_annual.csv", "damper_stuck_075_annual.csv", "damper_stuck_100_annual_short.csv"
    ],
    "oa_bias": [
        "oa_bias_-2_annual.csv", "oa_bias_-4_annual.csv", "oa_bias_2_annual.csv", "oa_bias_4_annual.csv"
    ]
}

# Select key sensor features
features = [
    "SA_TEMP", "OA_TEMP", "MA_TEMP", "RA_TEMP",  # Temperature-related
    "SF_SPD", "RF_SPD", "SF_CS", "RF_CS",  # Fan speed & control
    "OA_CFM", "RA_CFM", "SA_CFM",  # Airflow-related
    "OA_DMPR", "RA_DMPR", "CHWC_VLV",  # Damper & cooling valve
    "SYS_CTL"  # System control
]

df_normal = df_normal[features]

# Set timestamp index (assuming data is recorded per minute)
df_normal["Timestamp"] = pd.date_range(start="2023-01-01", periods=len(df_normal), freq="T")
df_normal.set_index("Timestamp", inplace=True)

scaler = MinMaxScaler()
df_normal_scaled = pd.DataFrame(scaler.fit_transform(df_normal), columns=df_normal.columns, index=df_normal.index)

# Load faulty data
df_fault = pd.read_csv(f"LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/{fault_files['coi_bias'][0]}")
df_fault = df_fault[features]

# Set timestamp index
df_fault["Timestamp"] = pd.date_range(start="2023-01-01", periods=len(df_fault), freq="T")
df_fault.set_index("Timestamp", inplace=True)

# Normalize
df_fault_scaled = pd.DataFrame(scaler.transform(df_fault), columns=df_fault.columns, index=df_fault.index)

# Plot temperature sensors
plt.figure(figsize=(15, 6))
plt.plot(df_normal_scaled.index, df_normal_scaled["SA_TEMP"], label="Normal - SA_TEMP")
plt.plot(df_fault_scaled.index, df_fault_scaled["SA_TEMP"], label="Faulty - SA_TEMP", linestyle="dashed")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalized Temperature")
plt.title("Coil Bias - Supply Air Temperature")
plt.show()

df_fault = pd.read_csv(f"LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/{fault_files['coi_leakage'][0]}")
df_fault = df_fault[features]
df_fault["Timestamp"] = pd.date_range(start="2023-01-01", periods=len(df_fault), freq="T")
df_fault.set_index("Timestamp", inplace=True)
df_fault_scaled = pd.DataFrame(scaler.transform(df_fault), columns=df_fault.columns, index=df_fault.index)

plt.figure(figsize=(15, 6))
plt.plot(df_normal_scaled.index, df_normal_scaled["MA_TEMP"], label="Normal - Mixed Air Temperature")
plt.plot(df_fault_scaled.index, df_fault_scaled["MA_TEMP"], label="Faulty - Mixed Air Temperature", linestyle="dashed")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalized Temperature")
plt.title("Coil Leakage - Mixed Air Temperature")
plt.show()

df_fault = pd.read_csv(f"LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/{fault_files['coi_stuck'][0]}")
df_fault = df_fault[features]
df_fault["Timestamp"] = pd.date_range(start="2023-01-01", periods=len(df_fault), freq="T")
df_fault.set_index("Timestamp", inplace=True)
df_fault_scaled = pd.DataFrame(scaler.transform(df_fault), columns=df_fault.columns, index=df_fault.index)

plt.figure(figsize=(15, 6))
plt.plot(df_normal_scaled.index, df_normal_scaled["CHWC_VLV"], label="Normal - Cooling Valve")
plt.plot(df_fault_scaled.index, df_fault_scaled["CHWC_VLV"], label="Faulty - Cooling Valve", linestyle="dashed")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalized Valve Position")
plt.title("Coil Stuck - Cooling Valve Control")
plt.show()

df_fault = pd.read_csv(f"LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/{fault_files['damper_stuck'][0]}")
df_fault = df_fault[features]
df_fault["Timestamp"] = pd.date_range(start="2023-01-01", periods=len(df_fault), freq="T")
df_fault.set_index("Timestamp", inplace=True)
df_fault_scaled = pd.DataFrame(scaler.transform(df_fault), columns=df_fault.columns, index=df_fault.index)

plt.figure(figsize=(15, 6))
plt.plot(df_normal_scaled.index, df_normal_scaled["OA_DMPR"], label="Normal - Outdoor Air Damper")
plt.plot(df_fault_scaled.index, df_fault_scaled["OA_DMPR"], label="Faulty - Outdoor Air Damper", linestyle="dashed")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalized Damper Position")
plt.title("Damper Stuck - Outdoor Air Damper Control")
plt.show()

df_fault = pd.read_csv(f"LBNL_FDD_Dataset_SDAHU_all/LBNL_FDD_Dataset_SDAHU/{fault_files['oa_bias'][0]}")
df_fault = df_fault[features]
df_fault["Timestamp"] = pd.date_range(start="2023-01-01", periods=len(df_fault), freq="T")
df_fault.set_index("Timestamp", inplace=True)
df_fault_scaled = pd.DataFrame(scaler.transform(df_fault), columns=df_fault.columns, index=df_fault.index)

plt.figure(figsize=(15, 6))
plt.plot(df_normal_scaled.index, df_normal_scaled["OA_CFM"], label="Normal - Outdoor Airflow")
plt.plot(df_fault_scaled.index, df_fault_scaled["OA_CFM"], label="Faulty - Outdoor Airflow", linestyle="dashed")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Normalized Airflow")
plt.title("Outdoor Air Bias - Outdoor Airflow")
plt.show()

