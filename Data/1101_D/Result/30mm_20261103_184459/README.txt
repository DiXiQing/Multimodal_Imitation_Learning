Trial-level 5-fold CV. Each trial is tested exactly once with fresh models trained on the other four groups.
Function: current distance only. MLP 5f: last five distances. LSTM Full: causal complete history; reset at trial start, invalid frame and time/frame gaps.
Input: index_mcp_object_distance_mm / its first recorded frame value (dimensionless). Output: finger_width_mm. No velocity, smoothing or time/amplitude normalization.
Output numerical scaling is fitted independently on each fold Train only. All models use the same fold. 400 fixed neural epochs; save final epoch; no Test model/epoch/parameter selection.
LSTM loss includes first valid frame of each sequence. Main comparisons use common 5-frame-valid targets. Full-from-start LSTM predictions exported separately.
Top-level comparison: equal-trial mean ± sample SD (ddof=1) across all out-of-fold trial metrics; not SD over five fold means. R2 unitless; MAE/RMSE mm. Frame-pooled metrics are separate.
Original folders read directly; no copies/moves. Five model sets saved; no automatic best fold or all-data deployment fit.
