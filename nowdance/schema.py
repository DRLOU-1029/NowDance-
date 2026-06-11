from __future__ import annotations

from dataclasses import dataclass
from typing import Any


POSE_LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

LANDMARK_INDEX = {name: index for index, name in enumerate(POSE_LANDMARK_NAMES)}
SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class PoseFrame:
    timestamp: float
    keypoints: list[list[float]]

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PoseFrame":
        points_by_name = {point["name"]: point for point in payload["keypoints"]}
        keypoints: list[list[float]] = []
        for name in POSE_LANDMARK_NAMES:
            point = points_by_name.get(name)
            if point is None:
                keypoints.append([0.0, 0.0, 0.0, 0.0])
            else:
                keypoints.append([
                    float(point["x"]),
                    float(point["y"]),
                    float(point.get("z", 0.0)),
                    float(point.get("visibility", 1.0)),
                ])
        return cls(timestamp=float(payload["t"]), keypoints=keypoints)

    def to_json(self) -> dict[str, Any]:
        return {
            "t": round(self.timestamp, 4),
            "keypoints": [
                {
                    "name": name,
                    "x": round(point[0], 6),
                    "y": round(point[1], 6),
                    "z": round(point[2], 6),
                    "visibility": round(point[3], 6),
                }
                for name, point in zip(POSE_LANDMARK_NAMES, self.keypoints)
            ],
        }


@dataclass(frozen=True)
class PoseSequence:
    frames: list[PoseFrame]
    source: str | None = None
    fps: float | None = None
    sampled_fps: float | None = None

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PoseSequence":
        return cls(
            frames=[PoseFrame.from_json(frame) for frame in payload["frames"]],
            source=payload.get("source"),
            fps=payload.get("fps"),
            sampled_fps=payload.get("sampled_fps"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "source": self.source,
            "fps": self.fps,
            "sampled_fps": self.sampled_fps,
            "frame_count": len(self.frames),
            "landmark_model": "mediapipe_pose_33",
            "frames": [frame.to_json() for frame in self.frames],
        }
