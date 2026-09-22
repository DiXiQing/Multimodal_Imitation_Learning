# Reach-to-grasp trial comparison

Seven `grasp_data.csv` files were analyzed. Source files were read only. All values use the CSV columns as recorded; no trial was removed or adjusted.

## Summary

| Trial | Frames | Duration (s) | Area mean | Width mean | Mag mean | Velocity mean | Missing/zero | Spikes |
|---|---|---|---|---|---|---|---|---|
| TRIAL_20260916_112804 | 70 | 2.288 | 12834.814 | 148.609 | 0.848 | 0.1038 | 0.71% | 0 |
| TRIAL_20260916_114311 | 76 | 2.496 | 12825.158 | 162.623 | 0.844 | -0.1084 | 0.66% | 1 |
| TRIAL_20260916_114955 | 101 | 3.329 | 9433.574 | 136.495 | 3.002 | 0.0971 | 0.50% | 0 |
| TRIAL_20260916_140846 | 67 | 2.192 | 14777.567 | 141.255 | 1.332 | 0.1744 | 0.75% | 2 |
| TRIAL_20260918_143818 | 41 | 1.328 | 8315.878 | 163.773 | 3.017 | -0.0129 | 1.22% | 0 |
| TRIAL_20260919_133440 | 84 | 2.768 | 16874.405 | 183.668 | 2.166 | 0.0116 | 0.60% | 3 |
| TRIAL_20260919_133940 | 51 | 1.665 | 14398.431 | 183.622 | 1.356 | -0.0289 | 0.98% | 0 |

## Methods

- Frames are CSV row counts; duration is last minus first `t_camera` in seconds. Event times are seconds since the first row. If a maximum ties, its first occurrence is reported. Feature units are unspecified in the source CSV and are left unchanged.
- Each first/last 10% mean uses ceil(0.10 × row count) rows. Standard deviations within trials and CV across trials use sample standard deviation (ddof=1).
- Missing means blank, nonnumeric, or nonfinite. Zero means an exact numeric zero. The overall missing/zero ratio divides their combined count across the four features by four times the row count. Zeros are retained in all statistics.
- Spike flags are isolated one-frame reversals: the point differs from a centered 5-frame median, and both adjacent steps exceed max(6 × 1.4826 × MAD of absolute local residuals, 3 × median absolute adjacent step). Endpoints are not flagged. These are descriptive flags, not confirmed errors.
- Smoothed curves use a centered 5-frame moving mean (`min_periods=1`) for display only. Statistics use raw values. Curves use elapsed seconds; no time normalization or alignment was imposed.
- Pairwise differences are trial B minus trial A in every feature mean. Possible outliers use modified z = 0.6745 × (value − across-trial median) / MAD, with |z| > 3.5. When MAD is zero, no flag is computed for that metric. Seven trials give limited evidence for outliers.
- CV is across trial means. Mean absolute velocity is also reported because signed velocity can cancel, making its CV hard to interpret.

## Observations

- Trial lengths range from 41 to 101 frames; durations range from 1.328 to 3.329 s.
- object_area trial means range from 8316 (TRIAL_20260918_143818) to 1.687e+04 (TRIAL_20260919_133440); across-trial CV = 0.236.
  - Closest pair of trial means: TRIAL_20260916_112804 and TRIAL_20260916_114311 (absolute difference 9.656); largest pairwise difference: TRIAL_20260918_143818 and TRIAL_20260919_133440 (8559).
  - Last 10% mean exceeds first 10% mean in 7 trials, is lower in 0, and is equal in 0.
- finger_width trial means range from 136.5 (TRIAL_20260916_114955) to 183.7 (TRIAL_20260919_133440); across-trial CV = 0.119.
  - Closest pair of trial means: TRIAL_20260919_133440 and TRIAL_20260919_133940 (absolute difference 0.04567); largest pairwise difference: TRIAL_20260916_114955 and TRIAL_20260919_133440 (47.17).
  - Last 10% mean exceeds first 10% mean in 7 trials, is lower in 0, and is equal in 0.
- mag trial means range from 0.8436 (TRIAL_20260916_114311) to 3.017 (TRIAL_20260918_143818); across-trial CV = 0.523.
  - Closest pair of trial means: TRIAL_20260916_112804 and TRIAL_20260916_114311 (absolute difference 0.004389); largest pairwise difference: TRIAL_20260916_114311 and TRIAL_20260918_143818 (2.173).
  - Last 10% mean exceeds first 10% mean in 7 trials, is lower in 0, and is equal in 0.
- velocity trial means range from -0.1084 (TRIAL_20260916_114311) to 0.1744 (TRIAL_20260916_140846); across-trial CV = 2.844.
  - Closest pair of trial means: TRIAL_20260916_112804 and TRIAL_20260916_114955 (absolute difference 0.006657); largest pairwise difference: TRIAL_20260916_114311 and TRIAL_20260916_140846 (0.2828).
  - Last 10% mean exceeds first 10% mean in 4 trials, is lower in 3, and is equal in 0.
- Mean absolute velocity varies across trials with CV = 0.591. The signed velocity CV is sensitive to cancellation around zero.
- The four feature columns contain 0 missing values and 14 exact zeros across 1960 cells. The spike rule flags 6 points in total.
- Frame gaps: none.
- No trial mean exceeded the modified-z threshold on the tested metrics. No rows or trials were excluded.

## Files

`summary.csv` contains all per-trial metrics and event locations. `pairwise_differences.csv` contains all 21 trial pairs. `between_trial_variability.csv` contains the CVs. `spike_flags.csv` and `possible_outliers.csv` contain inspection flags. Four comparison PNGs show raw and smoothed curves.
