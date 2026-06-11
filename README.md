# NowDance 算法验证原型

这是一个面向“舞力全开”类课程设计的 Python 算法原型，目标是先验证：

- 从舞蹈视频采集标准动作序列
- 将人体姿态关键点归一化为可比较的骨架
- 使用 DTW 对齐玩家动作和标准动作
- 输出相似度评分、评级和帧级明细

## Conda 环境安装

推荐使用 Conda 创建独立虚拟环境，避免和系统 Python 或后续 Android 工具链互相影响。

```powershell
conda env create -f environment.yml
conda activate nowdance
```

如果后续修改了依赖，可以更新环境：

```powershell
conda env update -f environment.yml --prune
```

如果使用的是新版 MediaPipe Tasks API，需要准备姿态模型文件：

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task" `
  -OutFile "models/pose_landmarker_lite.task"
```

当前仓库已下载了 `models/pose_landmarker_lite.task`，可以直接运行采集命令。

## 从舞蹈视频采集标准动作

```powershell
python -m nowdance.cli collect `
  --video assets/teacher_dance.mp4 `
  --out charts/teacher_dance.json `
  --sample-fps 15 `
  --inference-width 720
```

输出的 JSON 就是“标准动作序列”，后续 Android 端可以按同样结构读取或转换。

## 用玩家视频评分

```powershell
python -m nowdance.cli score `
  --standard charts/teacher_dance.json `
  --player-video assets/player_dance.mp4 `
  --out reports/player_score.json
```

也可以直接对两个已采集好的动作序列评分：

```powershell
python -m nowdance.cli score `
  --standard charts/teacher_dance.json `
  --player-sequence charts/player_dance.json
```

## 摄像头实时评分

实时模式会打开一个 OpenCV 画布：左侧播放标准舞蹈视频，右侧显示摄像头画面和实时骨架，顶部显示当前分、平均分、评级和连击。
每局开始前会先进入校准阶段：玩家需要举起右手、左手放低，并保持完整入镜 3 秒，校准通过后才开始播放音乐和正式评分。

```powershell
python -m nowdance.cli live `
  --standard charts/sample_dance.json `
  --video assets/sample_dance.mp4 `
  --camera-index 0 `
  --bpm 120 `
  --judge-window 0.35 `
  --hit-threshold 45 `
  --pose-tolerance 0.75
```

如果实时画面偏卡，推荐先使用上半身轻量模式：

```powershell
python -m nowdance.cli live `
  --standard charts/sample_dance.json `
  --video assets/sample_dance.mp4 `
  --camera-index 0 `
  --body-mode upper `
  --inference-width 360 `
  --process-every 2 `
  --bpm 120 `
  --score-every-beats 1 `
  --judge-window 0.45 `
  --hit-threshold 40 `
  --pose-tolerance 0.85
```

其中：

- `--body-mode upper`：只用肩、肘、腕、髋部做评分和绘制
- `--inference-width 360`：降低摄像头推理分辨率
- `--process-every 2`：每 2 帧推理一次，中间帧复用上一帧姿态
- `--bpm 120`：每分钟 120 拍，也就是每 0.5 秒结算一次
- `--score-every-beats 1`：每隔多少拍结算一次评分；例如 BPM 149 且每 4 拍一次，用 `--score-every-beats 4`
- `--judge-window 0.45`：节拍前后 0.45 秒内的动作都可以参与判定
- `--hit-threshold 40`：节拍最高分达到 40 就延续 Combo
- `--pose-tolerance 0.85`：姿态距离容忍度，越大越宽松
- `--calibration-seconds 3`：开局校准动作需要保持的秒数
- `--skip-calibration`：跳过校准，直接开始游戏

实时模式现在不是每帧断 Combo，而是按节拍结算。系统会在每个节拍点取“节拍附近玩家动作”和“该节拍标准动作”的最高相似度，得到一次 Perfect、Great、Good、Close 或 Miss。

快捷键：

- `Q` 或 `Esc`：退出
- `R`：从头重新开始标准视频和评分

如果摄像头画面打不开，可以尝试把 `--camera-index` 改成 `1` 或 `2`。

实时模式的画面由 OpenCV 绘制，OpenCV 本身不会播放 MP4 音轨。项目会自动尝试调用 `ffplay` 播放同一个视频的声音；如果没有声音，请先安装 ffmpeg：

```powershell
conda install -n nowdance -c conda-forge ffmpeg
```

如果只想静音运行：

```powershell
python -m nowdance.cli live `
  --standard charts/xbd.json `
  --video assets/xbd.mp4 `
  --no-audio
```

如果异常退出后声音没有停止，可以手动结束音频进程：

```powershell
taskkill /F /IM ffplay.exe /T
```

## 当前算法

1. MediaPipe Pose 从视频中提取 33 个人体关键点。
2. 每帧以髋部中心为原点，按肩宽、髋宽和躯干尺度归一化，降低人物远近、画面位置影响。
3. 使用关键点距离分数和关节角度分数混合得到帧级相似度。
4. 使用 DTW 动态时间规整对齐标准序列和玩家序列，允许玩家动作略快或略慢。

## 测试

核心评分算法不依赖 OpenCV 或 MediaPipe，可以直接运行：

```powershell
python -m unittest discover -s tests
```
