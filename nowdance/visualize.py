from __future__ import annotations

from pathlib import Path

import numpy as np

from .schema import LANDMARK_INDEX, PoseFrame, PoseSequence


POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("left_wrist", "left_index"),
    ("right_wrist", "right_index"),
]

IMPORTANT_POINTS = {
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
}


def render_sequence_video(
    sequence: PoseSequence,
    output_path: str | Path,
    width: int = 720,
    height: int = 1280,
    fps: float | None = None,
    visibility_threshold: float = 0.35,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("可视化需要安装 opencv-python，请先激活 Conda 环境。") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_rate = fps or sequence.sampled_fps or sequence.fps or 15.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(frame_rate),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频文件：{output_path}")

    try:
        for index, frame in enumerate(sequence.frames):
            canvas = _draw_frame(frame, index, len(sequence.frames), width, height, visibility_threshold, cv2)
            writer.write(canvas)
    finally:
        writer.release()


def _draw_frame(
    frame: PoseFrame,
    frame_index: int,
    frame_count: int,
    width: int,
    height: int,
    visibility_threshold: float,
    cv2,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
    points = _frame_points(frame, width, height)

    _draw_grid(canvas, cv2)
    _draw_connections(canvas, points, frame, visibility_threshold, cv2)
    _draw_points(canvas, points, frame, visibility_threshold, cv2)
    _draw_header(canvas, frame, frame_index, frame_count, cv2)
    return canvas


def _frame_points(frame: PoseFrame, width: int, height: int) -> dict[str, tuple[int, int]]:
    usable_width = int(width * 0.88)
    usable_height = int(height * 0.86)
    offset_x = int(width * 0.06)
    offset_y = int(height * 0.08)

    result = {}
    for name, index in LANDMARK_INDEX.items():
        point = frame.keypoints[index]
        x = offset_x + int(np.clip(point[0], 0.0, 1.0) * usable_width)
        y = offset_y + int(np.clip(point[1], 0.0, 1.0) * usable_height)
        result[name] = (x, y)
    return result


def _draw_grid(canvas: np.ndarray, cv2) -> None:
    height, width = canvas.shape[:2]
    color = (36, 43, 52)
    for x in range(0, width, max(1, width // 8)):
        cv2.line(canvas, (x, 0), (x, height), color, 1)
    for y in range(0, height, max(1, height // 10)):
        cv2.line(canvas, (0, y), (width, y), color, 1)


def _draw_connections(
    canvas: np.ndarray,
    points: dict[str, tuple[int, int]],
    frame: PoseFrame,
    visibility_threshold: float,
    cv2,
) -> None:
    for start, end in POSE_CONNECTIONS:
        start_point = frame.keypoints[LANDMARK_INDEX[start]]
        end_point = frame.keypoints[LANDMARK_INDEX[end]]
        if start_point[3] < visibility_threshold or end_point[3] < visibility_threshold:
            continue
        cv2.line(canvas, points[start], points[end], (77, 214, 181), 5, cv2.LINE_AA)


def _draw_points(
    canvas: np.ndarray,
    points: dict[str, tuple[int, int]],
    frame: PoseFrame,
    visibility_threshold: float,
    cv2,
) -> None:
    for name, point in points.items():
        landmark = frame.keypoints[LANDMARK_INDEX[name]]
        if landmark[3] < visibility_threshold:
            continue
        radius = 8 if name in IMPORTANT_POINTS else 5
        color = (246, 196, 83) if name in IMPORTANT_POINTS else (149, 166, 183)
        cv2.circle(canvas, point, radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, radius + 2, (18, 22, 28), 1, cv2.LINE_AA)


def _draw_header(
    canvas: np.ndarray,
    frame: PoseFrame,
    frame_index: int,
    frame_count: int,
    cv2,
) -> None:
    height, width = canvas.shape[:2]
    progress = 0.0 if frame_count <= 1 else frame_index / (frame_count - 1)
    cv2.rectangle(canvas, (0, 0), (width, 58), (24, 30, 38), -1)
    cv2.rectangle(canvas, (28, height - 38), (width - 28, height - 24), (55, 64, 76), -1)
    cv2.rectangle(canvas, (28, height - 38), (28 + int((width - 56) * progress), height - 24), (77, 214, 181), -1)
    cv2.putText(
        canvas,
        f"NowDance Pose Sequence  t={frame.timestamp:.2f}s  frame={frame_index + 1}/{frame_count}",
        (28, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 241, 247),
        2,
        cv2.LINE_AA,
    )
