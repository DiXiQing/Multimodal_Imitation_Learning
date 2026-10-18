Input: hand_object_distance (px); output: finger_width (px).
Split is frozen in split.json. Only training-trial leave-one-trial-out validation selects the model.
Spline is a cubic B-spline with second-difference coefficient penalty. Polynomial uses a scaled input.
model.json stores coefficients, scaling, and spline knots where applicable; predict() in the script loads this format.
No prediction clipping; no exported source measurements changed.
Test trials share this session and participant; this does not establish cross-session/participant generalization.
Model mode: median; validation selection metric: mae.
Median spline estimates quantile 0.5 using weighted absolute error plus an L1 second-difference smoothness penalty.
Mean and median references are each tuned on training folds, then compared on the identical test split.
Model-family test comparison is descriptive and does not change the selected model.
