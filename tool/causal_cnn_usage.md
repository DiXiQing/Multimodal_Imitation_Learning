运行：

```powershell
python tool/train_window_causal_cnn.py
```

默认沿用 train_window_mlp.py 中的数据路径、训练模式、训练轮数、优化器和窗口设置。在 Khairy/3 的单尺寸实验中，默认复用 temporalLoss_20261016_140009_768410/data_split.csv；引用的实验缺失时明确报错，不重新抽取测试集。

输出目录：数据目录/Result/causalCNN_时间/。

四个模型使用同一划分：MLP（OA）、MLP（OA＋有符号加速度）、因果 CNN（OA）、因果 CNN（OA＋有符号加速度）。全部使用 MSE，不加入变化量损失、不额外平滑预测、不修改原始 CSV。

CNN 默认 16 通道，卷积核 3，15 帧窗口对应膨胀率 1、2、4，只在左侧填充，以最后时间步特征回归当前 FW。约 2200 个参数，无双向操作、无未来数据、无跨时间归一化。更改窗口后自动增加卷积层直到感受野覆盖窗口。

指定另一份已有划分：

```powershell
python tool/train_window_causal_cnn.py --split-csv "已有结果/data_split.csv"
```

重新划分：

```powershell
python tool/train_window_causal_cnn.py --new-split
```

test_predictions 中每段有四模型对照 PNG 和逐帧 CSV；test_predictions_metrics.csv 含 RMSE、MAE、R²、相邻帧变化误差和峰值误差。模型分别保存于 MLP/models 和 CNN/models，checkpoint 包含结构、训练数据归一化参数及特征顺序。新 CNN checkpoint 需由本脚本的 predict_trial 加载，不能交给原 MLP 专用加载函数。

这是固定轮数的结构对照实验。测试集不用于早停或选参数；单次运行的改善不能代表其他参与者、尺寸或实时场景的表现。
