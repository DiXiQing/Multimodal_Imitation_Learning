# Crop verification

Source: TRIAL_Size_4cm_20261015_195026

Accepted: 31 of 36 major FW cycles. Each segment contains original JPEG frames, grasp_video.mp4, camera.csv, imu.csv, grasp_data.csv and grasp_data_smoothed.csv. Names use the actual first camera timestamp in Asia/Tokyo, to seconds, retaining size 4 cm.

Detection: centered 5-frame median used only for boundary finding; FW peak prominence 80 px, distance 70 frames. Onset is the final crossing of baseline + 10% peak excursion within 65 frames before the peak. Pre-roll is up to 25 frames, shortened if it overlaps the preceding return. Stable tail requires 16 samples (stable through stable+15), FW range <=20 px, median FW between 15% and 80% of the excursion, OA relative range <=22%, and OA gain >=1.35. Boundary images were reviewed. Four cycles (peaks 954, 2738, 3109, 3197) failed the stable-tail requirement; peak 1121 was excluded for FW loss at frame 1096. Initial small hand adjustment and return movements are not exported.

Pre-roll distribution: {25: 25, 23: 3, 22: 1, 16: 1, 19: 1}. Post-stability padding: 15 frames.

FW/OA smoothing: causal trailing 3-frame median then EMA span 5. Acceleration: interpolate original IMU onto a 50 Hz grid, causal second-order 3 Hz Butterworth with first-sample steady-state initialization, then interpolate to camera timestamps. No curve shifts or magnitude substitution. forward_raw remains identical in the synchronized raw and smoothed CSV; forward_acceleration is filtered signed acceleration.

Original IMU: 6346 rows; 60 duplicate timestamps and 105 backward timestamp transitions. Sorting/deduplication affects computation only; saved imu.csv retains original rows and ordering within each selected time interval. Global maximum sorted gap: 0.100578 s. Exported-interval maximum gap including interpolation support: 0.073547 s. Exported intervals contain 31 gaps >60 ms. Timing is irregular and short missing/delayed intervals exist; true sensor packet loss cannot be distinguished from notification timestamp effects using these files alone. No accepted interval exceeds 250 ms; all have IMU samples bracketing camera timestamps.

All 1995 frames were copied with SHA-256 checks; all segment videos were fully decoded and matched frame and CSV counts. Frame IDs and camera timestamps agree across the three camera-aligned CSV files. Signed acceleration and smoothing were independently recomputed and verified. Original recording files including source frames were hash-checked unchanged by the exporter. No JSON generated.
