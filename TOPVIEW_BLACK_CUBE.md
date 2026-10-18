# 黑色方块：连续预览、分次采集（无 IMU）

本版按照 `Record_mediapipe_continuous.py` 的 Tk 窗口、MediaPipe 指尖识别、N/S 连续试次及保存结构改写。为避免导入旧配置时加载蓝牙和 IMU，所需相机与检测逻辑独立放在新文件中。

```powershell
python New/Program/record_topview_black_cube.py --camera 1
```

请使用能运行 `Record_mediapipe_continuous.py` 的 Python 环境。需要 OpenCV、NumPy、Pillow 和提供 `mp.solutions.hands` 的 MediaPipe。无需蓝色标记。

1. 相机垂直固定在上方，使用浅色背景。启动后自动全画面检测黑色物体并显示绿框，不需要先框选。
2. 未出现绿框时，勾选“显示黑色检测掩膜”。白色为检测候选：方块不够白时提高亮度上限 V；色偏导致漏检时提高饱和度上限 S；小物体被过滤时降低最小面积。默认 V=60、S=120、最小面积=500，与连续版共享检测配置一致。
3. 此检测按黑色区域筛选，选择最大候选，不能语义区分黑色衣服、桌边和阴影。开始前确认绿框确实覆盖目标方块，移开其他黑色物品。相比上一版已取消严格正方形筛选。
4. MediaPipe 自动识别拇指尖 4、食指尖 8；C 为两点中点，B 为间距。尺寸栏填写方块尺寸用于标识试次。
5. 按 N 开始，S 结束；相机预览始终保持。再次 N 录制下一条；Q 保存并退出。
6. 默认每帧检测物体中心。如果遮挡前需锁定中心，按 L，再按 L 解除；锁定状态下物体必须静止。移动方块后务必解除锁定重新检测。录制期间不允许切换锁定状态或阈值。

输出位于项目 `Data/TopView/TRIAL_Size_尺寸cm_时间/`：`frames/` 保存原始 JPG、`recorded_video.mp4` 保存带标注视频、`camera.csv` 保存测量、`settings.json` 保存本次配置。不生成 IMU 文件。

CSV 保留 `frame,t_camera,finger_width,object_area`，新增指尖坐标、物体中心、手中点和 `hand_object_distance`。单位为像素；未检测值留空，`valid=0`；中心来源见 `object_center_source`（detected/locked/missing）。物体面积沿用旧版包围框面积定义。锁定中心只是假定物体不动，不代表遮挡时仍有视觉检测结果。

先采集少量试次检查原始画面和曲线，再正式采集。训练时仅使用 valid=1 的帧，按完整试次划分训练和测试数据。视频使用固定帧率，实际时间以 CSV 时间戳为准。此脚本不含模型训练。
