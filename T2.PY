import pandas as pd
import os

# 改成你的文件夹
rec_dir = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\BackupData\recording_20260705_192855"

# 读清洗后的CSV
clean_csv = [f for f in os.listdir(rec_dir) if f.endswith("_clean.csv")][0]
df = pd.read_csv(os.path.join(rec_dir, clean_csv))

# 清洗后CSV里保留的文件名
keep_frames = set(df["frame_file"].dropna())
keep_emg    = set(df["emg_file"].dropna())

# 删除frames里没用到的
frames_dir = os.path.join(rec_dir, "frames")
deleted = 0
for f in os.listdir(frames_dir):
    if f not in keep_frames:
        os.remove(os.path.join(frames_dir, f))
        deleted += 1
print(f"删除frames: {deleted}")

# 删除emg里没用到的
emg_dir = os.path.join(rec_dir, "emg")
deleted = 0
for f in os.listdir(emg_dir):
    if f not in keep_emg:
        os.remove(os.path.join(emg_dir, f))
        deleted += 1
print(f"删除emg: {deleted}")

print("清理完成")