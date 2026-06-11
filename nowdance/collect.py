from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import POSE_LANDMARK_NAMES, PoseFrame, PoseSequence


DEFAULT_TASK_MODEL = Path("models/pose_landmarker_lite.task")


def collect_pose_sequence_from_video(
    video_path: str | Path,
    sample_fps: float = 15.0,
    max_frames: int | None = None,
    model_complexity: int = 1,
    task_model: str | Path = DEFAULT_TASK_MODEL,
    inference_width: int = 720,
) -> PoseSequence:
    try:
        import cv2
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "视频采集需要安装 opencv-python 和 mediapipe，请先运行：conda env create -f environment.yml"
        ) from exc

    video_path = Path(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频：{video_path}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_interval = max(1, round(source_fps / sample_fps))
    frames: list[PoseFrame] = []

    try:
        _collect_with_solutions(
            capture=capture,
            mp=mp,
            cv2=cv2,
            source_fps=source_fps,
            frame_interval=frame_interval,
            frames=frames,
            max_frames=max_frames,
            model_complexity=model_complexity,
            inference_width=inference_width,
        )
    except AttributeError:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _collect_with_tasks(
            capture=capture,
            mp=mp,
            cv2=cv2,
            source_fps=source_fps,
            frame_interval=frame_interval,
            frames=frames,
            max_frames=max_frames,
            task_model=Path(task_model),
            inference_width=inference_width,
        )
    finally:
        capture.release()

    return PoseSequence(
        frames=frames,
        source=str(video_path),
        fps=source_fps,
        sampled_fps=source_fps / frame_interval,
    )


def _collect_with_solutions(
    capture: Any,
    mp: Any,
    cv2: Any,
    source_fps: float,
    frame_interval: int,
    frames: list[PoseFrame],
    max_frames: int | None,
    model_complexity: int,
    inference_width: int,
) -> None:
    mp_pose = mp.solutions.pose
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            rgb = cv2.cvtColor(_resize_for_inference(frame, cv2, inference_width), cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if result.pose_landmarks:
                frames.append(
                    PoseFrame(
                        timestamp=frame_index / source_fps,
                        keypoints=_landmarks_to_keypoints(result.pose_landmarks.landmark),
                    )
                )

            if max_frames is not None and len(frames) >= max_frames:
                break
            frame_index += 1


def _collect_with_tasks(
    capture: Any,
    mp: Any,
    cv2: Any,
    source_fps: float,
    frame_interval: int,
    frames: list[PoseFrame],
    max_frames: int | None,
    task_model: Path,
    inference_width: int,
) -> None:
    if not task_model.exists():
        raise FileNotFoundError(
            f"缺少 MediaPipe Tasks 模型：{task_model}。"
            "请下载 pose_landmarker_lite.task 到 models 目录。"
        )

    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python import vision

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_interval != 0:
                frame_index += 1
                continue

            timestamp_ms = int(round(frame_index * 1000 / source_fps))
            rgb = cv2.cvtColor(_resize_for_inference(frame, cv2, inference_width), cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, timestamp_ms)
            if result.pose_landmarks:
                frames.append(
                    PoseFrame(
                        timestamp=frame_index / source_fps,
                        keypoints=_landmarks_to_keypoints(result.pose_landmarks[0]),
                    )
                )

            if max_frames is not None and len(frames) >= max_frames:
                break
            frame_index += 1


def _resize_for_inference(frame: Any, cv2: Any, inference_width: int) -> Any:
    if inference_width <= 0 or frame.shape[1] <= inference_width:
        return frame
    height = round(frame.shape[0] * inference_width / frame.shape[1])
    return cv2.resize(frame, (inference_width, height), interpolation=cv2.INTER_AREA)


def _landmarks_to_keypoints(landmarks: Any) -> list[list[float]]:
    keypoints = []
    for landmark in landmarks[: len(POSE_LANDMARK_NAMES)]:
        visibility = getattr(landmark, "visibility", None)
        if visibility is None:
            visibility = getattr(landmark, "presence", 1.0)
        keypoints.append(
            [
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                float(visibility),
            ]
        )
    return keypoints
