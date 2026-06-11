from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .schema import LANDMARK_INDEX, POSE_LANDMARK_NAMES, PoseFrame, PoseSequence


UPPER_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)


@dataclass
class ChartStep:
    """单个动作步骤的谱面定义。"""

    step_number: int
    name: str
    motion_type: Literal["pose", "circle"]
    start_time: float
    end_time: float

    template_frame: np.ndarray | None = None

    center_landmark: str | None = None
    limb_end: str | None = None
    circle_direction: Literal["cw", "ccw"] | None = None
    revolutions: int = 1
    expected_jerk: float = 0.08

    def to_json(self) -> dict[str, Any]:
        base = {
            "step_number": self.step_number,
            "name": self.name,
            "motion_type": self.motion_type,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
        }
        if self.motion_type == "pose" and self.template_frame is not None:
            base["template"] = {
                name: {
                    "x": round(float(self.template_frame[idx, 0]), 6),
                    "y": round(float(self.template_frame[idx, 1]), 6),
                    "z": round(float(self.template_frame[idx, 2]), 6),
                    "visibility": round(float(self.template_frame[idx, 3]), 6),
                }
                for idx, name in enumerate(POSE_LANDMARK_NAMES)
            }
        if self.motion_type == "circle":
            base["center_landmark"] = self.center_landmark
            base["limb_end"] = self.limb_end
            base["direction"] = self.circle_direction
            base["revolutions"] = self.revolutions
        base["expected_jerk"] = round(float(self.expected_jerk), 4)
        return base

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "ChartStep":
        template_frame = None
        if "template" in payload:
            template_frame = np.zeros((33, 4), dtype=np.float32)
            for name, pt in payload["template"].items():
                idx = LANDMARK_INDEX.get(name)
                if idx is not None:
                    template_frame[idx] = [
                        pt["x"], pt["y"], pt.get("z", 0.0), pt.get("visibility", 1.0)
                    ]
        return cls(
            step_number=payload["step_number"],
            name=payload["name"],
            motion_type=payload["motion_type"],
            start_time=payload["start_time"],
            end_time=payload["end_time"],
            template_frame=template_frame,
            center_landmark=payload.get("center_landmark"),
            limb_end=payload.get("limb_end"),
            circle_direction=payload.get("direction"),
            revolutions=payload.get("revolutions", 1),
            expected_jerk=payload.get("expected_jerk", 0.08),
        )


@dataclass
class Chart:
    """舞蹈谱面。"""

    name: str = "极乐净土"
    bpm: float = 148.0
    steps: list[ChartStep] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bpm": self.bpm,
            "steps": [step.to_json() for step in self.steps],
        }

    def save(self, path: str) -> None:
        import json
        from pathlib import Path
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(self.to_json(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "Chart":
        import json
        from pathlib import Path
        with Path(path).open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            name=payload.get("name", "极乐净土"),
            bpm=payload.get("bpm", 148.0),
            steps=[ChartStep.from_json(s) for s in payload["steps"]],
        )


def extract_chart_from_sequence(
    sequence: PoseSequence,
    step_defs: list[dict[str, Any]],
) -> Chart:
    """从标准动作序列中提取每个步骤的模板帧，构建谱面。"""
    from .normalize import normalize_frame

    timestamps = np.asarray([f.timestamp for f in sequence.frames], dtype=np.float32)

    steps: list[ChartStep] = []
    for defn in step_defs:
        mask = (timestamps >= defn["start_s"]) & (timestamps <= defn["end_s"])
        indices = np.where(mask)[0]
        if len(indices) == 0:
            indices = np.array([np.argmin(np.abs(timestamps - defn["start_s"]))])

        if defn["type"] == "pose":
            normed = np.stack([normalize_frame(sequence.frames[i]) for i in indices])
            template = np.median(normed, axis=0).astype(np.float32)
            steps.append(ChartStep(
                step_number=defn["step"],
                name=defn["name"],
                motion_type="pose",
                start_time=defn["start_s"],
                end_time=defn["end_s"],
                template_frame=template,
            ))
        elif defn["type"] == "circle":
            steps.append(ChartStep(
                step_number=defn["step"],
                name=defn["name"],
                motion_type="circle",
                start_time=defn["start_s"],
                end_time=defn["end_s"],
                center_landmark=defn.get("center"),
                limb_end=defn.get("limb"),
                circle_direction=defn.get("direction", "cw"),
                revolutions=defn.get("revolutions", 1),
            ))

    return Chart(steps=steps)