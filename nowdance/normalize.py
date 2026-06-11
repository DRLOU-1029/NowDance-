from __future__ import annotations

import math

import numpy as np

from .schema import LANDMARK_INDEX, PoseFrame


ANGLE_JOINTS = [
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_elbow", "left_shoulder", "left_hip"),
    ("right_elbow", "right_shoulder", "right_hip"),
    ("left_shoulder", "left_hip", "left_knee"),
    ("right_shoulder", "right_hip", "right_knee"),
    ("left_hip", "left_knee", "left_ankle"),
    ("right_hip", "right_knee", "right_ankle"),
]


def frame_array(frame: PoseFrame) -> np.ndarray:
    return np.asarray(frame.keypoints, dtype=np.float32)


def normalize_frame(frame: PoseFrame) -> np.ndarray:
    points = frame_array(frame).copy()
    xy = points[:, :2]
    visibility = points[:, 3]

    origin = _center(points, "left_hip", "right_hip")
    if origin is None:
        visible_xy = xy[visibility > 0.2]
        origin = visible_xy.mean(axis=0) if len(visible_xy) else np.zeros(2, dtype=np.float32)

    scale = _body_scale(points)
    normalized = points.copy()
    normalized[:, 0] = (xy[:, 0] - origin[0]) / scale
    normalized[:, 1] = (xy[:, 1] - origin[1]) / scale
    normalized[:, 2] = points[:, 2] / scale
    return normalized


def visible_mask(a: np.ndarray, b: np.ndarray, threshold: float) -> np.ndarray:
    return (a[:, 3] >= threshold) & (b[:, 3] >= threshold)


def angle_features(normalized: np.ndarray, joints: list[tuple[str, str, str]] | None = None) -> dict[str, float]:
    features: dict[str, float] = {}
    for a_name, b_name, c_name in joints or ANGLE_JOINTS:
        a = normalized[LANDMARK_INDEX[a_name], :2]
        b = normalized[LANDMARK_INDEX[b_name], :2]
        c = normalized[LANDMARK_INDEX[c_name], :2]
        angle = _angle(a, b, c)
        if not math.isnan(angle):
            features[f"{a_name}:{b_name}:{c_name}"] = angle
    return features


def _center(points: np.ndarray, left_name: str, right_name: str) -> np.ndarray | None:
    left = points[LANDMARK_INDEX[left_name]]
    right = points[LANDMARK_INDEX[right_name]]
    if left[3] < 0.2 or right[3] < 0.2:
        return None
    return (left[:2] + right[:2]) / 2.0


def _body_scale(points: np.ndarray) -> float:
    candidates = [
        _distance(points, "left_shoulder", "right_shoulder"),
        _distance(points, "left_hip", "right_hip"),
        _distance(points, "left_shoulder", "left_hip"),
        _distance(points, "right_shoulder", "right_hip"),
    ]
    valid = [value for value in candidates if value > 1e-6]
    if not valid:
        return 1.0
    return float(np.mean(valid))


def _distance(points: np.ndarray, left_name: str, right_name: str) -> float:
    left = points[LANDMARK_INDEX[left_name]]
    right = points[LANDMARK_INDEX[right_name]]
    if left[3] < 0.2 or right[3] < 0.2:
        return 0.0
    return float(np.linalg.norm(left[:2] - right[:2]))


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    denom = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denom < 1e-6:
        return float("nan")
    cosine = float(np.clip(np.dot(ab, cb) / denom, -1.0, 1.0))
    return math.acos(cosine)
