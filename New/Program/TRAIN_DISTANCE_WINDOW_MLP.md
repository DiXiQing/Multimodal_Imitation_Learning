# Fixed Train / Validation folders

Run in the gripper environment:

    python New/Program/train_distance_window_mlp.py

SOURCE defaults to C:\MineApp\Code\Multimodal_Imitation_Learning\New\Data\D1\3.
Change SOURCE or pass --source "absolute dataset root".

The root must contain Train/ and Validation/. Each holds complete grasp folders with camera.csv, segment.json and settings.json. Counts are automatic; no random split or train/test count arguments. backup/ and Result/ are ignored. Duplicate names or identical CSVs across the two groups are rejected.

Compare current distance with the past four plus current distance (5 frames), using the same 32,32 hidden layers and eligible validation target frames. No future frames or historical true aperture inputs. Discontinuous or missing data breaks the window.

Fit input MinMax and output standardization using Train only. Validation macro-MAE selects the best checkpoint and controls early stopping. Restore that checkpoint as the final model; do not refit on Validation or merge Validation into Train. These are validation results, not independent test results.

EPOCHS: maximum epochs. PATIENCE: early-stopping patience. LOSS='mae' by default.

Each run writes a new Result/distance_mlp_window5_timestamp/ directory:

- models/distance_current.pth and models/distance_window5.pth: best checkpoint, compatible with window inference.
- validation_predictions/: per-trial curves and CSV.
- validation_model_comparison.png: pooled-frame MAE, RMSE, R2.
- validation_metrics_chart.png: per-trial metrics.
- validation_phase_errors.png and PDF: Window 5 only; Initial, Movement, Hold; mean per-grasp phase MAE with sample SD error bars.
- validation_trial_phase_metrics.csv and validation_phase_summary.csv: phase details.
- training_curves.png and training_history.csv: learning curves and history.
- split.json and data_split.csv: explicit folder split.
- training_log.txt and RESULT.txt: logs and output paths.

Update the model path in realtime inference to use the newly trained checkpoint. No training is started by documentation or data-reading checks.
