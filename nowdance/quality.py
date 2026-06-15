from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .chart import Chart, ChartStep
from .normalize import normalize_frame, frame_array
from .schema import LANDMARK_INDEX, PoseFrame, PoseSequence


UPPER_LANDMARKS_IDX = [LANDMARK_INDEX[name] for name in
    ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
     "left_wrist", "right_wrist")]

UPPER_ANGLE_JOINTS = [
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_elbow", "left_shoulder", "left_hip"),
    ("right_elbow", "right_shoulder", "right_hip"),
    ("left_shoulder", "right_shoulder", "right_elbow"),
    ("right_shoulder", "left_shoulder", "left_elbow"),
]


@dataclass
class PoseQualityResult:
    position_score: float        # 与模板的位置偏差 (0-100)
    symmetry_score: float        # 左右对称性 (0-100)
    smoothness_score: float      # 动作流畅性 (0-100)
    overall_quality: float       # 加权综合 (0-100)
    issues: list[str] = field(default_factory=list)


@dataclass
class CircleQualityResult:
    circularity_score: float     # 轨迹圆度 (0-100)
    radius_score: float          # 半径一致性 (0-100)
    timing_score: float          # 节奏均匀性 (0-100)
    overall_quality: float       # 加权综合 (0-100)
    actual_radius: float = 0.0
    expected_radius: float = 0.0
    issues: list[str] = field(default_factory=list)


@dataclass
class StepQualityReport:
    step_number: int
    motion_type: str
    score: float
    grade: str
    combo: int = 0
    golden: bool = False
    details: dict[str, Any] = None
    issues: list[str] = None


def evaluate_step(
    step: ChartStep,
    player_frames: list[PoseFrame],
    config: dict[str, Any] | None = None,
) -> StepQualityReport:
    """对单个步骤执行质量评估。"""
    cfg = {
        "visibility_threshold": 0.35,
        "smoothness_window": 0.2,
    }
    if config:
        cfg.update(config)

    if step.motion_type == "pose":
        result = _evaluate_pose(step, player_frames, cfg)
    elif step.motion_type == "circle":
        result = _evaluate_circle(step, player_frames, cfg)
    else:
        raise ValueError(f"未知动作类型: {step.motion_type}")

    score = round(result.overall_quality, 2)
    if score >= 90: grade = "Perfect"
    elif score >= 75: grade = "Great"
    elif score >= 60: grade = "Good"
    elif score >= 40: grade = "Okay"
    else: grade = "Miss"

    return StepQualityReport(
        step_number=step.step_number,
        motion_type=step.motion_type,
        score=score,
        grade=grade,
        details=result.__dict__,
        issues=result.issues,
    )


def evaluate_all(
    chart: Chart,
    player_sequence: PoseSequence,
    config: dict[str, Any] | None = None,
) -> list[StepQualityReport]:
    """对谱面中所有步骤执行质量评估。"""
    timestamps = np.asarray([f.timestamp for f in player_sequence.frames], dtype=np.float32)
    reports: list[StepQualityReport] = []

    combo = 0
    all_perfect = True
    for step in chart.steps:
        mask = (timestamps >= step.start_time) & (timestamps <= step.end_time)
        indices = np.where(mask)[0]
        player_frames = [player_sequence.frames[i] for i in indices]
        report = evaluate_step(step, player_frames, config)
        if report.grade == "Miss":
            combo = 0
            all_perfect = False
        else:
            combo += 1
            if report.grade != "Perfect":
                all_perfect = False
        report.combo = combo
        report.golden = all_perfect and combo > 0
        reports.append(report)

    return reports


def _evaluate_pose(
    step: ChartStep,
    player_frames: list[PoseFrame],
    cfg: dict[str, Any],
) -> PoseQualityResult:
    issues: list[str] = []

    if step.template_frame is None:
        return PoseQualityResult(0, 0, 0, 0, issues=["缺标准模板"])

    if not player_frames:
        return PoseQualityResult(0, 0, 0, 0, issues=["未检测到动作"])

    # --- position_score: 与模板的位置偏差 ---
    player_normed = [normalize_frame(f) for f in player_frames]
    median_player = np.median(np.stack(player_normed), axis=0).astype(np.float32)
    template_normed = normalize_frame(
        PoseFrame(timestamp=0, keypoints=step.template_frame.tolist())
    )

    deltas = median_player[UPPER_LANDMARKS_IDX, :2] - template_normed[UPPER_LANDMARKS_IDX, :2]
    distances = np.linalg.norm(deltas, axis=1)
    mean_dist = float(np.mean(distances))
    tol = getattr(step, "tolerance", 1.0)
    position_score = max(0.0, 100.0 * (1.0 - mean_dist / (0.5 * tol)))

    # --- symmetry_score: 左右对称性 ---
    angle_diffs = _upper_angle_diffs(median_player, template_normed)
    mean_angle_diff = float(np.mean(angle_diffs)) if angle_diffs else 0.0
    symmetry_score = max(0.0, 100.0 * (1.0 - mean_angle_diff / (math.pi * 0.4 * tol)))

    if mean_angle_diff > math.pi * 0.2:
        issues.append("左右关节角度不对称")

    # --- smoothness_score: ???????jerk???????---
    # ?????jerk?????jerk??????1???
    if len(player_frames) >= 4:
        normed_stack = np.stack(player_normed, axis=0)
        velocities = np.diff(normed_stack[:, UPPER_LANDMARKS_IDX, :2], axis=0)
        speed_mag = np.linalg.norm(velocities, axis=2)
        jerk = np.std(speed_mag, axis=0)
        player_jerk = float(np.mean(jerk))
        expected_jerk = max(step.expected_jerk, 0.01)
        jerk_ratio = player_jerk / expected_jerk
        smoothness_score = max(0.0, 100.0 * max(0.0, 1.0 - (jerk_ratio - 1.0) / 2.0))
        if jerk_ratio > 1.8:
            issues.append("????????????")
    else:
        smoothness_score = 80.0

    # --- overall ---
    overall = position_score * 0.45 + symmetry_score * 0.25 + smoothness_score * 0.30

    return PoseQualityResult(
        position_score=round(position_score, 2),
        symmetry_score=round(symmetry_score, 2),
        smoothness_score=round(smoothness_score, 2),
        overall_quality=round(overall, 2),
        issues=issues,
    )


def _evaluate_circle(
    step: ChartStep,
    player_frames: list[PoseFrame],
    cfg: dict[str, Any],
) -> CircleQualityResult:
    issues: list[str] = []

    if step.limb_end is None or step.center_landmark is None:
        return CircleQualityResult(0, 0, 0, 0, issues=["缺少画圈参数"])

    if len(player_frames) < 4:
        return CircleQualityResult(0, 0, 0, 0, issues=["帧数不足以评估画圈"])

    limb_idx = LANDMARK_INDEX.get(step.limb_end)
    center_idx = LANDMARK_INDEX.get(step.center_landmark)
    if limb_idx is None or center_idx is None:
        return CircleQualityResult(0, 0, 0, 0, issues=["无效的关键点名"])

    normed = np.stack([normalize_frame(f) for f in player_frames], axis=0)
    traj = normed[:, limb_idx, :2]
    center_pt = normed[:, center_idx, :2]
    visibility = normed[:, limb_idx, 3]

    if float(np.mean(visibility)) < cfg["visibility_threshold"]:
        return CircleQualityResult(0, 0, 0, 0, issues=["肢体不可见"])

    rel = traj - center_pt

    # --- circularity_score: 轨迹一致性 ---
    centroid = np.mean(rel, axis=0)
    radii = np.linalg.norm(rel - centroid, axis=1)
    mean_r = float(np.mean(radii))
    if mean_r < 1e-4:
        return CircleQualityResult(0, 0, 0, 0, issues=["画圈半径过小"])

    r_residual = float(np.std(radii) / mean_r)
    # 视频中的画圈天然不是完美圆，阈值放宽
    circularity_score = max(0.0, 100.0 * (1.0 - r_residual / 1.0))
    if r_residual > 0.6:
        issues.append("轨迹一致性较差")

    # --- radius_score: 半径一致性 ---
    shoulder_name = "right_shoulder" if "right" in step.limb_end else "left_shoulder"
    shoulder_idx = LANDMARK_INDEX.get(shoulder_name)
    if shoulder_idx is not None:
        center_frame = normed[len(normed) // 2]
        expected_r = float(np.linalg.norm(
            center_frame[limb_idx, :2] - center_frame[shoulder_idx, :2]
        ))
        radius_ratio = mean_r / max(expected_r, 1e-4)
        if radius_ratio < 0.5:
            issues.append("画圈幅度不足")
            radius_score = max(0.0, 100.0 * radius_ratio / 0.5)
        elif radius_ratio > 1.5:
            issues.append("画圈幅度过大")
            radius_score = max(0.0, 100.0 * (1.0 - (radius_ratio - 1.5)))
        else:
            radius_score = 100.0
    else:
        expected_r = mean_r
        radius_score = 80.0

    # --- timing_score: 节奏均匀性 ---
    if len(player_frames) >= 8:
        angles = np.arctan2(rel[:, 1], rel[:, 0])
        unwrapped = np.unwrap(angles)
        time_steps = np.linspace(0, 1, len(unwrapped))
        coeffs = np.polyfit(time_steps, unwrapped, 1)
        fitted = np.polyval(coeffs, time_steps)
        residuals = unwrapped - fitted
        timing_var = float(np.std(residuals))
        # timing_var < 0.8 -> 100, > 3.0 -> 0
        timing_score = max(0.0, 100.0 * (1.0 - timing_var / 3.0))
        if timing_var > 1.8:
            issues.append("画圈节奏不均匀")
    else:
        timing_score = 60.0
        issues.append("帧数不足以评估画圈节奏")

    overall = circularity_score * 0.40 + radius_score * 0.35 + timing_score * 0.25

    return CircleQualityResult(
        circularity_score=round(circularity_score, 2),
        radius_score=round(radius_score, 2),
        timing_score=round(timing_score, 2),
        overall_quality=round(overall, 2),
        actual_radius=round(mean_r, 4),
        expected_radius=round(expected_r, 4),
        issues=issues,
    )


def _upper_angle_diffs(
    player_normed: np.ndarray,
    template_normed: np.ndarray,
) -> list[float]:
    diffs = []
    for a_name, b_name, c_name in UPPER_ANGLE_JOINTS:
        a_idx = LANDMARK_INDEX[a_name]
        b_idx = LANDMARK_INDEX[b_name]
        c_idx = LANDMARK_INDEX[c_name]
        pa = player_normed[a_idx, :2]
        pb = player_normed[b_idx, :2]
        pc = player_normed[c_idx, :2]
        p_angle = _angle_between(pa, pb, pc)
        ta = template_normed[a_idx, :2]
        tb = template_normed[b_idx, :2]
        tc = template_normed[c_idx, :2]
        t_angle = _angle_between(ta, tb, tc)
        if not (math.isnan(p_angle) or math.isnan(t_angle)):
            diffs.append(abs(p_angle - t_angle))
    return diffs


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ab = a - b
    cb = c - b
    denom = float(np.linalg.norm(ab)) * float(np.linalg.norm(cb))
    if denom < 1e-6:
        return float("nan")
    cosine = np.clip(np.dot(ab, cb) / denom, -1.0, 1.0)
    return math.acos(float(cosine))