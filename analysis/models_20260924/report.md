# Three regression models

Data: 490 valid frames from 7 trials.
Validation: leave one complete trial out. Models are linear ridge regressions.

| Model | Inputs | RMSE (px) | MAE (px) | R² |
|---|---|---:|---:|---:|
| area_velocity | object_area+velocity | 63.378 | 53.747 | 0.027 |
| area_only | object_area | 63.536 | 52.212 | 0.022 |
| area_acceleration | object_area+ax+ay+az | 86.873 | 69.115 | -0.829 |

The velocity values are the existing single-axis integrated values and inherit their known drift limitations.
The acceleration model uses ax, ay, and az together.