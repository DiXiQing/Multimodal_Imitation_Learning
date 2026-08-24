import numpy as np
import pandas as pd
from pathlib import Path

base = r"C:\MineApp\Code\Multimodal_Imitation_Learning\Data\TrainingData"
high, low = 0, 0

for rec_dir in Path(base).glob("recording_*"):
    csv = list(rec_dir.glob("*_clean.csv"))
    if not csv: continue
    df = pd.read_csv(csv[0])
    for _, row in df.iterrows():
        emg_path = rec_dir / "emg" / row["emg_file"]
        if emg_path.exists():
            emg = np.load(str(emg_path))
            rms = np.sqrt(np.mean(emg**2))
            if rms > 0.05:
                high += 1
            else:
                low += 1

print(f"高EMG（用力）: {high}")
print(f"低EMG（放松）: {low}")
print(f"比例: {high/(high+low)*100:.1f}% 高")