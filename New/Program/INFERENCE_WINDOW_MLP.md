# 5 帧距离 MLP 实时实验

在 gripper 环境运行：

```powershell
python New/Program/inference_window_mlp.py
```

默认加载 `New/Data/2/Result/distance_mlp_window5_20261017_213924_764497/models/distance_window5.pth`。
更换模型时修改脚本顶部 `MODEL_PATH`，或者传入 `--model "模型绝对路径"`。也支持传入训练结果文件夹。
使用最终的 `distance_window5.pth`，不要选择内部验证阶段的 checkpoint。

## 操作

- N：开始新试次，清空旧窗口，从该试次重新积累 5 帧。
- S：停止并保存当前试次。
- L：未录制时锁定/解锁物体中心，同时清空历史窗口。
- M：查看颜色检测掩膜。
- Q / Esc：保存并退出。

默认使用模型中记录的摄像头编号和颜色检测参数；可用 `--camera 0` 更改摄像头。
保持训练时的相机位置、分辨率、帧率、物体和操作要求。当前模型仍由你手动选择。

输入按 `[d(t-4), d(t-3), d(t-2), d(t-1), d(t)]` 排列，使用 checkpoint 中的输入归一化和输出反标准化参数。
不读取未来帧，不输入真实指间距，也不额外平滑预测。
漏检或帧间隔超过模型限制（这次为 0.1 秒）会清空缓存。预热时记录真实开度，预测留空，不把旧预测当新预测。
画面显示窗口实际覆盖时长和单次 MLP 计算耗时；MLP 耗时不包含摄像头、检测与绘图耗时。

## 保存

每次运行新建 `Data/TopView/Inference/run_时间戳/`。每次按 N 后，在 `trial_001` 等目录保存：

- `raw_video.mp4`：原始画面。
- `prediction_video.mp4`：带真实值、预测值及误差的画面。
- `predictions.csv`：逐帧时间、距离、真实开度、预测开度、误差、窗口内容和状态。
- `prediction_plot.png`：真实与预测曲线。
- `summary.json`：MAE、RMSE、R²、漏检和预热帧数；只在有预测的帧上计算误差。
- `config.json`：模型路径、归一化、检测参数和物体中心设置。

运行目录也保存整场 CSV 和汇总。任意窗口距离超出训练范围时会标记；不截断距离或输出。视频以固定 FPS 写入，真实实验时间以 CSV 时间戳为准。

可回放原始视频（应与训练分辨率一致）：

```powershell
python New/Program/inference_window_mlp.py --video "原始视频绝对路径" --auto-record
```

回放用视频时间判断窗口连续性；实时摄像头用实际读取时间。现有 `inference.py` 保持原样，新脚本复用其中的录制类，因此两份文件需要放在同一目录。
