# 算法验证方案

## 目标

本阶段先不急着做完整 Android 游戏，而是验证核心闭环：

```text
标准舞蹈视频 -> 姿态关键点序列 -> 玩家视频/摄像头姿态 -> 时间对齐 -> 相似度评分
```

## 标准动作采集模块

入口：

```powershell
python -m nowdance.cli collect --video assets/teacher.mp4 --out charts/teacher.json --sample-fps 15
```

处理流程：

1. 使用 OpenCV 逐帧读取视频。
2. 按 `sample-fps` 抽样，避免序列过密导致 DTW 计算量过大。
3. 使用 MediaPipe Pose Solutions 或 MediaPipe Tasks Pose Landmarker 提取 33 个人体关键点。
4. 保存为 JSON，包含时间戳、关键点名称、二维坐标、深度近似值和可见度。

当前 JSON 以 MediaPipe 的归一化图像坐标为基础，即 `x`、`y` 通常在 `0..1` 范围内。评分时会再次做身体尺度归一化。

如果本地 MediaPipe 版本使用 Tasks API，需要准备 `models/pose_landmarker_lite.task` 模型文件。这个模型也更接近后续 Android 端 MediaPipe Tasks 的迁移方式。

## 姿态归一化

为了减少摄像头距离、人物站位和视频分辨率差异的影响，每一帧都会单独归一化：

1. 以左右髋部中心作为骨架原点。
2. 使用肩宽、髋宽、左右躯干长度的均值作为身体尺度。
3. 所有关键点坐标转换为相对于身体尺度的坐标。

这样玩家站得偏左、偏右、远一些或近一些时，评分不会明显漂移。

## 帧级评分

帧级相似度由两部分组成：

- 关键点距离分数：比较标准姿态和玩家姿态中可见关键点的平均距离。
- 关节角度分数：比较肩、肘、髋、膝等主要关节夹角。

默认权重：

```text
总分 = 0.7 * 关键点距离分数 + 0.3 * 关节角度分数
```

关键点距离负责整体姿态位置，角度分数负责动作形态，可以降低手脚整体偏移造成的误判。

## 时间对齐

玩家动作很难和标准视频完全同速，所以使用 DTW 动态时间规整：

1. 先计算标准序列每一帧和玩家序列每一帧之间的代价。
2. 在代价矩阵中寻找总代价最低的单调路径。
3. 按路径上的匹配帧计算平均分。

这允许玩家动作略快、略慢，或者某个动作多停顿一两帧。

## 评级规则

```text
90-100  Perfect
75-89   Great
60-74   Good
40-59   Okay
0-39    Miss
```

课程设计里可以把总分映射为游戏内的连击、命中判定和最终评级。

## Android 迁移建议

Python 原型验证通过后，Android 端建议复用同一套数据和算法边界：

- 摄像头：CameraX
- 姿态识别：MediaPipe Tasks Vision Pose Landmarker 或 TFLite MoveNet
- 标准动作：读取本原型生成的 JSON，或转换为更紧凑的二进制资源
- 评分逻辑：将 `normalize.py` 和 `scoring.py` 的数学逻辑迁移到 Kotlin

模块边界建议：

```text
PoseExtractor     摄像头帧 -> 关键点
PoseSequenceStore 标准动作 JSON 读写
PoseNormalizer    单帧骨架归一化
DanceScorer       DTW 对齐和相似度评分
GameJudge         分数 -> Perfect/Great/Good/Miss/连击
```
