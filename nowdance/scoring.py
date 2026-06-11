from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .normalize import angle_features, normalize_frame, visible_mask
from .schema import LANDMARK_INDEX, PoseFrame, PoseSequence


UPPER_BODY_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
)

UPPER_BODY_ANGLE_JOINTS = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
    ("left_elbow", "left_shoulder", "left_hip"),
    ("right_elbow", "right_shoulder", "right_hip"),
)


@dataclass(frozen=True)
class ScoringConfig:
    visibility_threshold: float = 0.35
    landmark_tolerance: float = 0.55
    landmark_weight: float = 0.7
    angle_weight: float = 0.3
    dtw_band_ratio: float = 0.25
    landmark_names: tuple[str, ...] | None = None
    angle_joints: tuple[tuple[str, str, str], ...] | None = None


@dataclass(frozen=True)
class FrameScore:
    standard_index: int
    player_index: int
    standard_time: float
    player_time: float
    score: float
    landmark_score: float
    angle_score: float


def score_sequences(
    standard: PoseSequence,
    player: PoseSequence,
    config: ScoringConfig | None = None,
) -> dict:
    config = config or ScoringConfig()
    if not standard.frames:
        raise ValueError("标准动作序列为空")
    if not player.frames:
        raise ValueError("玩家动作序列为空")

    standard_norm = [normalize_frame(frame) for frame in standard.frames]
    player_norm = [normalize_frame(frame) for frame in player.frames]
    costs = _cost_matrix(standard.frames, player.frames, standard_norm, player_norm, config)
    path = _dtw_path(costs, config.dtw_band_ratio)

    frame_scores = [
        _score_pair(
            standard.frames[standard_index],
            player.frames[player_index],
            standard_norm[standard_index],
            player_norm[player_index],
            standard_index,
            player_index,
            config,
        )
        for standard_index, player_index in path
    ]
    scores = np.asarray([item.score for item in frame_scores], dtype=np.float32)
    average_score = float(scores.mean())

    return {
        "score": round(average_score, 2),
        "grade": grade_for_score(average_score),
        "matched_frames": len(frame_scores),
        "standard_frames": len(standard.frames),
        "player_frames": len(player.frames),
        "frame_scores": [
            {
                "standard_index": item.standard_index,
                "player_index": item.player_index,
                "standard_time": round(item.standard_time, 4),
                "player_time": round(item.player_time, 4),
                "score": round(item.score, 2),
                "landmark_score": round(item.landmark_score, 2),
                "angle_score": round(item.angle_score, 2),
            }
            for item in frame_scores
        ],
    }


def score_frame_pair(
    standard_frame: PoseFrame,
    player_frame: PoseFrame,
    config: ScoringConfig | None = None,
) -> FrameScore:
    config = config or ScoringConfig()
    return _score_pair(
        standard_frame,
        player_frame,
        normalize_frame(standard_frame),
        normalize_frame(player_frame),
        0,
        0,
        config,
    )


def upper_body_config(**overrides) -> ScoringConfig:
    return ScoringConfig(
        landmark_names=UPPER_BODY_LANDMARKS,
        angle_joints=UPPER_BODY_ANGLE_JOINTS,
        **overrides,
    )


def grade_for_score(score: float) -> str:
    if score >= 90:
        return "Perfect"
    if score >= 75:
        return "Great"
    if score >= 60:
        return "Good"
    if score >= 40:
        return "Okay"
    return "Miss"


def _cost_matrix(
    standard_frames: list[PoseFrame],
    player_frames: list[PoseFrame],
    standard_norm: list[np.ndarray],
    player_norm: list[np.ndarray],
    config: ScoringConfig,
) -> np.ndarray:
    costs = np.zeros((len(standard_frames), len(player_frames)), dtype=np.float32)
    for i, standard_frame in enumerate(standard_frames):
        for j, player_frame in enumerate(player_frames):
            frame_score = _score_pair(
                standard_frame,
                player_frame,
                standard_norm[i],
                player_norm[j],
                i,
                j,
                config,
            )
            costs[i, j] = 100.0 - frame_score.score
    return costs


def _score_pair(
    standard_frame: PoseFrame,
    player_frame: PoseFrame,
    standard_norm: np.ndarray,
    player_norm: np.ndarray,
    standard_index: int,
    player_index: int,
    config: ScoringConfig,
) -> FrameScore:
    landmark = _landmark_score(standard_norm, player_norm, config)
    angle = _angle_score(standard_norm, player_norm, config)
    score = config.landmark_weight * landmark + config.angle_weight * angle
    return FrameScore(
        standard_index=standard_index,
        player_index=player_index,
        standard_time=standard_frame.timestamp,
        player_time=player_frame.timestamp,
        score=float(score),
        landmark_score=float(landmark),
        angle_score=float(angle),
    )


def _landmark_score(standard_norm: np.ndarray, player_norm: np.ndarray, config: ScoringConfig) -> float:
    mask = visible_mask(standard_norm, player_norm, config.visibility_threshold)
    if config.landmark_names:
        subset = np.zeros_like(mask)
        for name in config.landmark_names:
            subset[LANDMARK_INDEX[name]] = True
        mask = mask & subset
    if not np.any(mask):
        return 0.0
    deltas = standard_norm[mask, :2] - player_norm[mask, :2]
    distances = np.linalg.norm(deltas, axis=1)
    weights = np.minimum(standard_norm[mask, 3], player_norm[mask, 3])
    mean_distance = float(np.average(distances, weights=weights))
    return max(0.0, 100.0 * (1.0 - mean_distance / config.landmark_tolerance))


def _angle_score(standard_norm: np.ndarray, player_norm: np.ndarray, config: ScoringConfig) -> float:
    joints = list(config.angle_joints) if config.angle_joints else None
    standard_angles = angle_features(standard_norm, joints)
    player_angles = angle_features(player_norm, joints)
    shared = sorted(set(standard_angles) & set(player_angles))
    if not shared:
        return 0.0
    diffs = np.asarray([abs(standard_angles[key] - player_angles[key]) for key in shared], dtype=np.float32)
    mean_diff = float(diffs.mean())
    return max(0.0, 100.0 * (1.0 - mean_diff / np.pi))


def _dtw_path(costs: np.ndarray, band_ratio: float) -> list[tuple[int, int]]:
    rows, cols = costs.shape
    band = max(abs(rows - cols), int(max(rows, cols) * band_ratio))
    accumulated = np.full((rows + 1, cols + 1), np.inf, dtype=np.float32)
    accumulated[0, 0] = 0.0

    for i in range(1, rows + 1):
        start = max(1, i - band)
        end = min(cols, i + band) + 1
        for j in range(start, end):
            accumulated[i, j] = costs[i - 1, j - 1] + min(
                accumulated[i - 1, j],
                accumulated[i, j - 1],
                accumulated[i - 1, j - 1],
            )

    if not np.isfinite(accumulated[rows, cols]):
        return [(index, min(index, cols - 1)) for index in range(rows)]

    i, j = rows, cols
    path: list[tuple[int, int]] = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        options = [
            (accumulated[i - 1, j - 1], i - 1, j - 1),
            (accumulated[i - 1, j], i - 1, j),
            (accumulated[i, j - 1], i, j - 1),
        ]
        _, i, j = min(options, key=lambda item: item[0])
    path.reverse()
    return path
