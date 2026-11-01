Function / MLP 5f / LSTM Full. Input: index_mcp_object_distance_mm / first-frame value (l/l0, dimensionless; Euclidean landmark 5 to object). Output: finger_width_mm.
400 fixed neural epochs; final epoch saved. Function single Train-only fit; no epoch selection or Test tuning.
Distance x-axis uses l/l0, aperture and error metrics use mm (R2 unitless). Actual elapsed seconds and original frame IDs retained. Input l/l0 is fed directly (no secondary input scaling); output numerical conditioning fitted on Train only. No time or aperture normalization.
Main metrics use common valid 5-frame targets for all models. Separate LSTM full-history exports start at first valid frame, resetting on missing/invalid rows and frame/time gaps.
Train and Test reports include overall and per-experiment scores. Train MAE history is measured during training with dropout active; final Train metrics use evaluation mode.
Original data preserved; split_manifest.json records whole-experiment partition.
