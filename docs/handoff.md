# NowDance 项目进度总结

## 一、项目定位

一个基于 MediaPipe 的人体姿态评估舞蹈评分系统，类似"舞力全开"（Just Dance）。当前聚焦于**上半身**的"极乐净土"舞蹈 demo。

## 二、已完成的工作

### 1. 核心模块（`nowdance/` 包）

| 模块 | 功能 | 完成度 |
|------|------|--------|
| `schema.py` | PoseFrame / PoseSequence 数据结构，33 个关键点定义 | ✅ |
| `collect.py` | 从视频提取 MediaPipe 姿态序列（支持 legacy solutions 和新版 tasks API） | ✅ |
| `io.py` | JSON 读写（序列/谱面） | ✅ |
| `normalize.py` | 骨架归一化（髋部中心为原点，体尺缩放） | ✅ |
| `scoring.py` | DTW 动态时间规整 + 帧级评分（位置 + 角度），含 Grading（Perfect/Great/Good/Miss） | ✅ |
| `chart.py` | ChartStep / Chart 谱面数据结构，支持 `pose` 和 `circle` 两种动作类型 | ✅ |
| `quality.py` | 质量评估引擎：位置偏差(45%) / 左右对称(25%) / 流畅性(30%) + 画圈评估（圆度/半径/节奏） | ✅ |
| `visualize.py` | 骨架序列渲染为视频 | ✅ |
| `live.py` | 实时摄像头打分（双画面：参考视频 + 摄像头，含节拍判定、Combo、校准） | ✅ |
| `cli.py` | 命令行入口：`collect` / `score` / `visualize` / `live` | ✅ |

### 2. 极乐净土数据

| 文件 | 内容 |
|------|------|
| `assets/jljt.mp4` | 极乐净土原始舞蹈视频 |
| `charts/jljt.json` | 394 帧姿态序列（从视频提取，34.6s，约 11.4 fps） |
| `charts/jljt_chart.json` | 45 步谱面（131 BPM，4/4 拍），含每步的 template、tolerance、expected_jerk |

### 3. 谱面详情（44 步）

| 步骤 | 时间 | 动作 | 容错度 |
|------|------|------|--------|
| 1 | 0.00-5.04s | 身体朝左，左右手同时向左伸直 | 1.0 |
| 2-3 | 5.04-12.37s | 右手顺时针旋转一周（各约8拍） | 1.2 |
| 4-9 | 12.37-17.86s | 斜线交替（各2拍） | 1.0 |
| 10 | 17.86-18.78s | 手臂收窄垂直向上 | 0.9 |
| 11-19 | 18.78-26.11s | 斜线交替（交替2拍，首尾各1拍） | 1.0 |
| 20 | 26.11-27.26s | 双臂与肩齐平，手肘向外，小臂指向胸前 | 1.0 |
| 21-44 | 27.26-34.29s | 肘部交替（每三个一组，第三个稍长，共8组） | 1.3 |
| 45 | 34.29-34.50s | 左手叉腰右手下垂 | 1.0 |


### 4. 辅助工具

| 文件 | 用途 |
|------|------|
| `scripts/render.py` | 将骨架叠加到原始视频上，显示步骤名 |
| `scripts/template_extract.py` | 从姿态序列提取每步的模板帧 |
| `scripts/test_step.py` | 测试工具，支持 `--camera` 摄像头模式和自评模式 |
| `tests/test_scoring.py` | scoring 模块的单元测试（4 个 test case） |

### 5. 已生成的产物

- `charts/jljt_overlay.mp4` / `charts/jljt_overlay_audio.mp4`：骨架叠加预览视频
- `charts/jljt_chart_preview.mp4`：谱面预览

## 三、待完成的工作

1. **测试工具摄像头模式 bug**：`test_step.py` 中 `--camera` 模式刚修复了变量名 `reference`→`ref` 错误，等待测试验证
2. **容错度调优**：需要收集真人动作评分数据，调整每步的 `tolerance` 值（当前是初步设定）
3. **质量评估集成到 live 模式**：`live.py` 目前用 DTW 逐帧评分，未使用 `quality.py` 的步骤级质量评估
4. **circle 动作类型支持**：`chart.py` 定义了 `circle` 类型，`quality.py` 有 `_evaluate_circle`，但目前谱面全用 `pose`，steps 2-3 标注为画圈但实际存储为 `pose`
5. **错误诊断系统**：课程设计文档中提到的动作错误识别（缺失/多余/延迟/幅度不足等）尚未实现
6. **云端服务 + 前端**：FastAPI 推理服务、移动端界面均未启动
7. **LLM 智能复盘**：课程设计中提到的大语言模型生成训练建议，未实现

## 四、技术栈与依赖

```
Python 3.11 (Conda 环境: nowdance)
├── numpy >= 1.26
├── opencv-python >= 4.9
├── mediapipe >= 0.10
└── Pillow (render.py 用)
```

关键信息：
- Git 仓库：[https://github.com/DRLOU-1029/NowDance-.git](https://github.com/DRLOU-1029/NowDance-.git)
- 最新提交：`c7e4877` "极乐净土完整谱面: 45步, 131BPM 拍子对齐, Step20肘部起点"
- 未提交变更：`chart.py`、`quality.py`、`jljt_chart.json`、`test_step.py`

## 五、给接手工程师的建议

1. 优先跑通 `python scripts/test_step.py --steps 1-5 --repeats 5 --camera` 验证摄像头模式正常
2. `quality.py` 是判分的核心，理解了三个维度的加权逻辑就能接手
3. 谱面 JSON 是 Unicode 转义格式（PowerShell 管道会损坏 UTF-8），`Chart.load()` 方法会自动处理
4. `live.py` 目前走的是节拍判定 + DTW 评分，与 `quality.py` 是两套并行的评分体系，需要决定最终用哪套


### 7. Android ????????

Android ??? `android/` ????19 ????????????

**?????**
1. ? Android Studio ?? `android/` ???
2. ????? Android SDK Platform 35 ? ??????? Gradle
3. ?? Android 15 ?????? ? Run

**????**
- Kotlin + CameraX + MediaPipe Tasks Vision
- ?????????????????
- ?? JSON ???????? assets ?

**???????????**
1. ??? 3-2-1 ? "RAISE HANDS!" ??
2. ???????? 3 ? ? ????
3. CameraX ???? MediaPipe ?? ? ????????
4. ????? `ScoringEngine` ???Python quality.py ? Kotlin ???
5. ???????????? S/A/B/C?????Perfect ???? Combo?? Golden ??

**6 ? Kotlin ????**

| ?? | ?? | ?? |
|------|------|------|
| `MainActivity.kt` | ~270 | ?????? CameraX / ??? / ??? / ???? |
| `chart/ChartStep.kt` | ~20 | ??????? |
| `chart/ChartLoader.kt` | ~60 | ? JSON ???? |
| `scoring/StepResult.kt` | ~10 | ??????? |
| `scoring/ScoringEngine.kt` | ~280 | ?????????? + ??/??/?????? |
| `pose/PoseOverlayView.kt` | ~90 | ?????? View |

**?????/????**
- `imageProxyToBitmap()` ? YUV?Bitmap ???????????????
- ???????????????????
- ????????? MediaPlayer ? ExoPlayer ? jljt.mp4?
- ???? R ???Q ???????????
- ?????? Kotlin String ?????
