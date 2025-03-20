import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

normal_path = "LBNL_FDD_Dataset_SDAHU_all\LBNL_FDD_Dataset_SDAHU\AHU_annual.csv"
df_normal = pd.read_csv(normal_path)

fault_files = [
    'damper_stuck_075_annual.csv'
]

df_faults = [pd.read_csv(f'LBNL_FDD_Dataset_SDAHU_all\LBNL_FDD_Dataset_SDAHU\{file}') for file in fault_files]
df_faults = pd.concat(df_faults, ignore_index=True)

features = [
    "SA_TEMP", "SA_TEMPSPT", "OA_TEMP", "MA_TEMP", "RA_TEMP",
    "SF_SPD_DM", "RF_SPD_DM", "OA_CFM", "RA_CFM", "SA_CFM",
    "SF_CS", "SF_SPD", "RF_CS", "RF_SPD", "SF_WAT",
    "RF_WAT", "OA_DMPR_DM", "OA_DMPR", "RA_DMPR_DM", "RA_DMPR",
    "CHWC_VLV_DM", "CHWC_VLV", "SA_SP", "SA_SPSPT", "SYS_CTL",
    "ZONE_TEMP_1", "ZONE_TEMP_2", "ZONE_TEMP_3", "ZONE_TEMP_4", "ZONE_TEMP_5"
]

df = df_faults[features] 

df["Timestamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="T")
df.set_index("Timestamp", inplace=True)  # 设定时间索引

print(df.info())  # 查看数据规模

df_hourly = df.resample("H").mean()  # 每小时取均值
df_daily = df.resample("D").mean()  # 每天取均值

plt.figure(figsize=(15, 6))
plt.plot(df_hourly.index, df_hourly["SA_TEMP"], label="送风温度 (SA_TEMP)")
plt.plot(df_hourly.index, df_hourly["OA_TEMP"], label="室外温度 (OA_TEMP)")
plt.plot(df_hourly.index, df_hourly["MA_TEMP"], label="混合空气温度 (MA_TEMP)")
plt.legend()
plt.xlabel("时间")
plt.ylabel("温度 (°F)")
plt.title("温度传感器变化趋势（每小时均值）")
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(df_hourly.index, df_hourly["SF_SPD"], label="送风风机转速 (SF_SPD)")
plt.plot(df_hourly.index, df_hourly["RF_SPD"], label="回风风机转速 (RF_SPD)")
plt.legend()
plt.xlabel("时间")
plt.ylabel("风速")
plt.title("风机状态变化趋势（每小时均值）")
plt.show()

plt.figure(figsize=(15, 6))
plt.plot(df_hourly.index, df_hourly["OA_DMPR"], label="室外空气风门 (OA_DMPR)")
plt.plot(df_hourly.index, df_hourly["RA_DMPR"], label="回风风门 (RA_DMPR)")
plt.legend()
plt.xlabel("时间")
plt.ylabel("风门开度 (0-1)")
plt.title("风门状态变化趋势（每小时均值）")
plt.show()

