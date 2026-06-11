import unittest

from nowdance.schema import LANDMARK_INDEX, POSE_LANDMARK_NAMES, PoseFrame, PoseSequence
from nowdance.scoring import score_sequences


def make_frame(timestamp: float, phase: float, offset_x: float = 0.0, scale: float = 1.0) -> PoseFrame:
    points = [[0.0, 0.0, 0.0, 0.0] for _ in POSE_LANDMARK_NAMES]

    def set_point(name: str, x: float, y: float) -> None:
        points[LANDMARK_INDEX[name]] = [offset_x + x * scale, y * scale, 0.0, 1.0]

    arm = 0.16 * phase
    set_point("left_hip", 0.45, 0.70)
    set_point("right_hip", 0.55, 0.70)
    set_point("left_shoulder", 0.42, 0.45)
    set_point("right_shoulder", 0.58, 0.45)
    set_point("left_elbow", 0.35 - arm, 0.55 - arm)
    set_point("right_elbow", 0.65 + arm, 0.55 - arm)
    set_point("left_wrist", 0.30 - arm, 0.65 - arm)
    set_point("right_wrist", 0.70 + arm, 0.65 - arm)
    set_point("left_knee", 0.44, 0.86)
    set_point("right_knee", 0.56, 0.86)
    set_point("left_ankle", 0.43, 1.0)
    set_point("right_ankle", 0.57, 1.0)
    return PoseFrame(timestamp=timestamp, keypoints=points)


def make_sequence(phases: list[float], offset_x: float = 0.0, scale: float = 1.0) -> PoseSequence:
    frames = [
        make_frame(index / 15.0, phase=phase, offset_x=offset_x, scale=scale)
        for index, phase in enumerate(phases)
    ]
    return PoseSequence(frames=frames, source="synthetic", fps=15.0, sampled_fps=15.0)


class ScoringTest(unittest.TestCase):
    def test_identical_sequence_scores_high(self) -> None:
        sequence = make_sequence([0.0, 0.4, 0.8, 1.0])
        report = score_sequences(sequence, sequence)
        self.assertGreater(report["score"], 99.0)
        self.assertEqual(report["grade"], "Perfect")

    def test_translation_and_scale_are_normalized(self) -> None:
        standard = make_sequence([0.0, 0.4, 0.8, 1.0])
        player = make_sequence([0.0, 0.4, 0.8, 1.0], offset_x=0.2, scale=1.4)
        report = score_sequences(standard, player)
        self.assertGreater(report["score"], 95.0)

    def test_dtw_tolerates_speed_difference(self) -> None:
        standard = make_sequence([0.0, 0.25, 0.5, 0.75, 1.0])
        player = make_sequence([0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
        report = score_sequences(standard, player)
        self.assertGreater(report["score"], 90.0)

    def test_wrong_motion_scores_lower(self) -> None:
        standard = make_sequence([0.0, 0.4, 0.8, 1.0])
        player = make_sequence([1.0, 0.8, 0.4, 0.0])
        report = score_sequences(standard, player)
        self.assertLess(report["score"], 85.0)


if __name__ == "__main__":
    unittest.main()
