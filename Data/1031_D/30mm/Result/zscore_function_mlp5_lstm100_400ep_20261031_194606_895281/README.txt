Function / MLP 5f / LSTM 100f. Input: index_mcp_object_distance_mm (Euclidean landmark 5 to object). Output: finger_width_mm.
400 fixed neural epochs; final epoch saved. Function single Train-only fit; no epoch selection or Test tuning.
All plots and metrics use mm (R2 unitless), actual elapsed seconds, original frame IDs. LSTM: independent rolling window up to 100 frames, zero state each window; short prefixes use available frames. Shared Train-only Z-score: x=(distance_mm-x_mean)/x_std; scaled_y=(aperture_mm-y_mean)/y_std. Predict mm=scaled_output*y_std+y_mean. No division by initial distance and no per-trial time/amplitude normalization.
Main metrics use common valid 5-frame targets for all models. Separate LSTM rolling-100 exports start at first valid frame, resetting on missing/invalid rows and frame/time gaps.
Train and Test reports include overall and per-experiment scores. Train MAE history is measured during training with dropout active; final Train metrics use evaluation mode.
Original data preserved; split_manifest.json records whole-experiment partition.
