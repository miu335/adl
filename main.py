import pandas as pd

path = "D:\\miu\\adl\\data\\raw\\test_dataset_sim.csv"   # 你的临时 CSV

# 1. 看首行原始内容
with open(path, "rb") as f:
    print(repr(f.readline()))   # 看是否有 BOM、是否含 ';'

# 2. 直接用 ';' 读
df = pd.read_csv(path, sep=";", engine="python")
print(df.shape)          # 应为 (19, 12)
print(df.columns.tolist())