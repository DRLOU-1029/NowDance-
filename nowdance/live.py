from __future__ import annotations

import atexit
import os
import signal
import time
from collections import deque
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from .collect import _landmarks_to_keypoints, _resize_for_inference
from .scoring import ScoringConfig, grade_for_score, score_frame_pair, upper_body_config
from .schema import LANDMARK_INDEX, PoseFrame, PoseSequence
from .visualize import POSE_CONNECTIONS


_ACTIVE_AUDIO_PLAYERS: list["_AudioPlayer"] = []


def run_live_scoring(
    standard: PoseSequence,
    reference_video: str | Path,
    camera_index: int = 0,
    task_model: str | Path = "models/pose_landmarker_lite.task",
    canvas_width: int = 1280,
    canvas_height: int = 720,
    inference_width: int = 480,
    process_every: int = 1,
    body_mode: str = "full",
    bpm: float = 120.0,
    score_every_beats: int = 1,
    beat_offset: float = 0.0,
    judge_window: float = 0.35,
    hit_threshold: float = 45.0,
    pose_tolerance: float = 0.75,
    standard_pose_delay: float = 0.2,
    play_audio: bool = True,
    audio_player: str | Path | None = None,
    calibration_seconds: float = 3.0,
    calibration_target_y: float = -0.08,
    calibration_target_scale: float = 1.0,
    calibration_mode: str = "relaxed",
    skip_calibration: bool = False,
    mirror_camera: bool = True,
    visibility_threshold: float = 0.35,
) -> None:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        raise RuntimeError("实时采集需要 opencv-python 和 mediapipe，请先激活 nowdance Conda 环境。") from exc

    if not standard.frames:
        raise ValueError("标准动作序列为空")

    reference = cv2.VideoCapture(str(reference_video))
    if not reference.isOpened():
        raise FileNotFoundError(f"无法打开标准视频：{reference_video}")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        reference.release()
        raise RuntimeError(f"无法打开摄像头：{camera_index}")

    task_model = Path(task_model)
    if not task_model.exists():
        reference.release()
        camera.release()
        raise FileNotFoundError(f"缺少姿态模型：{task_model}")

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(task_model)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    config = _scoring_config(body_mode, visibility_threshold, pose_tolerance)
    connections = _connections_for_mode(body_mode)
    session = _LiveSession(hit_threshold=hit_threshold)
    beat_tracker = _BeatTracker(
        bpm=bpm,
        score_every_beats=score_every_beats,
        offset=beat_offset,
        window=judge_window,
    )
    window_name = "NowDance Live Scoring - press Q to quit, R to restart"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    audio = _AudioPlayer(reference_video, audio_player) if play_audio else None

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            if not skip_calibration:
                ok = _run_calibration(
                    camera,
                    landmarker,
                    mp,
                    cv2,
                    window_name,
                    canvas_width,
                    canvas_height,
                inference_width,
                calibration_seconds,
                calibration_target_y,
                calibration_target_scale,
                calibration_mode,
                mirror_camera,
                visibility_threshold,
                )
                if not ok:
                    return

            started_at = time.perf_counter()
            frame_counter = 0
            player_pose: PoseFrame | None = None
            frame_score = None
            if audio:
                audio.start()
            while True:
                elapsed = time.perf_counter() - started_at
                if audio:
                    audio.restart_if_finished()
                detector_timestamp_ms = _monotonic_timestamp_ms()
                ok_ref, reference_frame = _read_reference_frame(reference, elapsed, cv2)
                ok_cam, camera_frame = camera.read()
                if not ok_ref or not ok_cam:
                    break

                if mirror_camera:
                    camera_frame = cv2.flip(camera_frame, 1)

                standard_frame = _frame_at_time(standard, elapsed - standard_pose_delay)
                if frame_counter % max(1, process_every) == 0:
                    player_pose = _detect_camera_pose(
                        camera_frame,
                        elapsed,
                        detector_timestamp_ms,
                        landmarker,
                        mp,
                        cv2,
                        inference_width,
                    )
                    frame_score = score_frame_pair(standard_frame, player_pose, config) if player_pose else None
                session.observe_frame(elapsed, frame_score.score if frame_score else None, player_pose)
                beat_result = beat_tracker.consume_due_beat(elapsed, session, standard, config, standard_pose_delay)

                canvas = _compose_canvas(
                    reference_frame,
                    camera_frame,
                    standard_frame,
                    player_pose,
                    session,
                    beat_tracker,
                    beat_result,
                    elapsed,
                    canvas_width,
                    canvas_height,
                    visibility_threshold,
                    connections,
                    cv2,
                )
                cv2.imshow(window_name, canvas)
                frame_counter += 1

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("r"), ord("R")):
                    started_at = time.perf_counter()
                    session.reset()
                    beat_tracker.reset()
                    reference.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if audio:
                        audio.restart()
    finally:
        reference.release()
        camera.release()
        if audio:
            audio.stop()
        cv2.destroyAllWindows()


class _AudioPlayer:
    def __init__(self, video_path: str | Path, audio_player: str | Path | None) -> None:
        self.video_path = str(video_path)
        self.player = str(audio_player) if audio_player else _find_ffplay()
        self.process: subprocess.Popen | None = None
        self.enabled = self.player is not None
        if not self.enabled:
            print("未找到 ffplay，实时模式将只播放画面。安装 ffmpeg 后可自动播放声音。")

    def start(self) -> None:
        if not self.enabled:
            return
        flags = 0
        kwargs: dict[str, Any] = {}
        if sys.platform.startswith("win"):
            flags |= subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        self.process = subprocess.Popen(
            [
                self.player,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "error",
                self.video_path,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=flags,
            **kwargs,
        )
        if self not in _ACTIVE_AUDIO_PLAYERS:
            _ACTIVE_AUDIO_PLAYERS.append(self)

    def restart_if_finished(self) -> None:
        if self.enabled and self.process is not None and self.process.poll() is not None:
            self.start()

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            _terminate_process_tree(self.process)
        self.process = None
        if self in _ACTIVE_AUDIO_PLAYERS:
            _ACTIVE_AUDIO_PLAYERS.remove(self)


def _find_ffplay() -> str | None:
    candidates = [
        shutil.which("ffplay"),
        Path(sys.executable).parent / "ffplay.exe",
        Path(sys.executable).parent / "Library" / "bin" / "ffplay.exe",
        Path(sys.executable).parent.parent / "Library" / "bin" / "ffplay.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if sys.platform.startswith("win"):
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(process.pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=1.0)
    except Exception:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            process.kill()


def _stop_active_audio_players() -> None:
    for player in list(_ACTIVE_AUDIO_PLAYERS):
        player.stop()


atexit.register(_stop_active_audio_players)


def _run_calibration(
    camera: Any,
    landmarker: Any,
    mp: Any,
    cv2: Any,
    window_name: str,
    canvas_width: int,
    canvas_height: int,
    inference_width: int,
    hold_seconds: float,
    target_y: float,
    target_scale: float,
    calibration_mode: str,
    mirror_camera: bool,
    visibility_threshold: float,
) -> bool:
    hold_started_at: float | None = None
    player_pose: PoseFrame | None = None
    message = "Raise your right hand and stay inside the frame"

    while True:
        ok, camera_frame = camera.read()
        if not ok:
            return False
        if mirror_camera:
            camera_frame = cv2.flip(camera_frame, 1)

        now = time.perf_counter()
        player_pose = _detect_camera_pose(
            camera_frame,
            now,
            _monotonic_timestamp_ms(),
            landmarker,
            mp,
            cv2,
            inference_width,
        )
        matched = _matches_calibration_pose(player_pose, visibility_threshold, calibration_mode) if player_pose else False
        if matched:
            if hold_started_at is None:
                hold_started_at = now
            held = now - hold_started_at
        else:
            hold_started_at = None
            held = 0.0

        canvas = _compose_calibration_canvas(
            camera_frame,
            player_pose,
            held,
            hold_seconds,
            matched,
            message,
            target_y,
            target_scale,
            canvas_width,
            canvas_height,
            visibility_threshold,
            cv2,
        )
        cv2.imshow(window_name, canvas)

        if held >= hold_seconds:
            _draw_countdown_flash(cv2, window_name, canvas, "START")
            return True

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("s"), ord("S")):
            return True


def _matches_calibration_pose(pose: PoseFrame, visibility_threshold: float, calibration_mode: str) -> bool:
    required = [
        "left_shoulder",
        "right_shoulder",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
    ]
    relaxed_threshold = max(0.15, visibility_threshold * 0.55)
    if any(pose.keypoints[LANDMARK_INDEX[name]][3] < relaxed_threshold for name in required):
        return False

    left_shoulder = pose.keypoints[LANDMARK_INDEX["left_shoulder"]]
    right_shoulder = pose.keypoints[LANDMARK_INDEX["right_shoulder"]]
    left_wrist = pose.keypoints[LANDMARK_INDEX["left_wrist"]]
    right_wrist = pose.keypoints[LANDMARK_INDEX["right_wrist"]]
    left_hip = pose.keypoints[LANDMARK_INDEX["left_hip"]]
    right_hip = pose.keypoints[LANDMARK_INDEX["right_hip"]]

    shoulder_width = abs(left_shoulder[0] - right_shoulder[0])
    hip_width = abs(left_hip[0] - right_hip[0])
    body_visible = shoulder_width > 0.045 and hip_width > 0.035
    if calibration_mode == "relaxed":
        return body_visible

    right_hand_up = right_wrist[1] < right_shoulder[1] + 0.08
    left_hand_up = left_wrist[1] < left_shoulder[1] + 0.08
    return body_visible and (right_hand_up or left_hand_up)


def _compose_calibration_canvas(
    camera_frame: Any,
    player_pose: PoseFrame | None,
    held: float,
    hold_seconds: float,
    matched: bool,
    message: str,
    target_y: float,
    target_scale: float,
    width: int,
    height: int,
    visibility_threshold: float,
    cv2: Any,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (12, 16, 22), dtype=np.uint8)
    frame = _resize_cover(camera_frame, width, height, cv2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    canvas = cv2.addWeighted(frame, 0.72, overlay, 0.28, 0)

    rect = (0, 0, width, height)
    if player_pose:
        _draw_pose(canvas, player_pose, rect, visibility_threshold, _connections_for_mode("upper"), cv2, (246, 196, 83))

    _draw_calibration_target(canvas, matched, target_y, target_scale, cv2)
    progress = min(1.0, held / max(0.1, hold_seconds))
    color = (77, 214, 181) if matched else (90, 110, 135)
    cv2.rectangle(canvas, (80, height - 76), (width - 80, height - 44), (36, 43, 52), -1)
    cv2.rectangle(canvas, (80, height - 76), (80 + int((width - 160) * progress), height - 44), color, -1)
    cv2.putText(canvas, "MATCH THE POSE", (80, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (245, 248, 252), 3, cv2.LINE_AA)
    cv2.putText(canvas, message, (80, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (235, 241, 247), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"Hold {held:0.1f}/{hold_seconds:0.1f}s   Press S to skip   Q/Esc to quit",
        (80, height - 96),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (235, 241, 247),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _draw_calibration_target(
    canvas: np.ndarray,
    matched: bool,
    target_y: float,
    target_scale: float,
    cv2: Any,
) -> None:
    h, w = canvas.shape[:2]
    color = (85, 255, 95) if matched else (90, 110, 135)
    cx = w * 0.50
    cy = h * (0.38 + target_y)
    scale = max(0.5, target_scale)

    def p(dx: float, dy: float) -> tuple[int, int]:
        return (int(cx + w * dx * scale), int(cy + h * dy * scale))

    pts = {
        "left_shoulder": p(-0.08, -0.04),
        "right_shoulder": p(0.08, -0.04),
        "left_wrist": p(-0.16, 0.18),
        "right_wrist": p(0.12, -0.28),
        "left_hip": p(-0.05, 0.22),
        "right_hip": p(0.05, 0.22),
    }
    lines = [
        ("left_shoulder", "right_shoulder"),
        ("right_shoulder", "right_wrist"),
        ("left_shoulder", "left_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
    ]
    for a, b in lines:
        cv2.line(canvas, pts[a], pts[b], color, 8, cv2.LINE_AA)
    for point in pts.values():
        cv2.circle(canvas, point, 11, (245, 210, 255), -1, cv2.LINE_AA)


def _draw_countdown_flash(cv2: Any, window_name: str, canvas: np.ndarray, text: str) -> None:
    frame = canvas.copy()
    h, w = frame.shape[:2]
    cv2.putText(frame, text, (w // 2 - 95, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (77, 214, 181), 6, cv2.LINE_AA)
    cv2.imshow(window_name, frame)
    cv2.waitKey(450)


class _LiveSession:
    def __init__(self, hit_threshold: float) -> None:
        self.hit_threshold = hit_threshold
        self.recent_frames: deque[tuple[float, float]] = deque(maxlen=120)
        self.recent_poses: deque[tuple[float, PoseFrame]] = deque(maxlen=120)
        self.preview_scores: deque[float] = deque(maxlen=30)
        self.total = 0.0
        self.count = 0
        self.combo = 0
        self.best_combo = 0
        self.last_judgement = "Ready"
        self.last_beat_score = 0.0

    @property
    def preview(self) -> float:
        return float(np.mean(self.preview_scores)) if self.preview_scores else 0.0

    @property
    def average(self) -> float:
        return self.total / self.count if self.count else 0.0

    def observe_frame(self, timestamp: float, score: float | None, pose: PoseFrame | None) -> None:
        if score is None:
            return
        self.recent_frames.append((timestamp, score))
        self.preview_scores.append(score)
        if pose is not None:
            self.recent_poses.append((timestamp, pose))

    def judge_beat(
        self,
        beat_time: float,
        standard_frame: PoseFrame,
        window: float,
        config: ScoringConfig,
    ) -> float:
        candidates = [
            score_frame_pair(standard_frame, pose, config).score
            for timestamp, pose in self.recent_poses
            if abs(timestamp - beat_time) <= window
        ]
        score = max(candidates) if candidates else 0.0
        self.last_beat_score = score
        self.last_judgement = _judgement_for_score(score, self.hit_threshold)
        self.total += score
        self.count += 1
        if score >= self.hit_threshold:
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
        else:
            self.combo = 0
        return score

    def reset(self) -> None:
        self.recent_frames.clear()
        self.recent_poses.clear()
        self.preview_scores.clear()
        self.total = 0.0
        self.count = 0
        self.combo = 0
        self.best_combo = 0
        self.last_judgement = "Ready"
        self.last_beat_score = 0.0


class _BeatTracker:
    def __init__(self, bpm: float, score_every_beats: int, offset: float, window: float) -> None:
        self.beat_interval = 60.0 / max(1.0, bpm)
        self.score_every_beats = max(1, score_every_beats)
        self.interval = self.beat_interval * self.score_every_beats
        self.offset = max(0.0, offset)
        self.window = max(0.05, window)
        self.next_index = 0

    @property
    def next_beat_time(self) -> float:
        return self.offset + self.next_index * self.interval

    def consume_due_beat(
        self,
        elapsed: float,
        session: _LiveSession,
        standard: PoseSequence,
        config: ScoringConfig,
        standard_pose_delay: float,
    ) -> float | None:
        result = None
        while elapsed >= self.next_beat_time + self.window:
            beat_time = self.next_beat_time
            result = session.judge_beat(
                beat_time,
                _frame_at_time(standard, beat_time - standard_pose_delay),
                self.window,
                config,
            )
            self.next_index += 1
        return result

    def reset(self) -> None:
        self.next_index = 0


def _read_reference_frame(capture: Any, elapsed: float, cv2: Any) -> tuple[bool, Any]:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        target = int(elapsed * fps) % total_frames
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    return capture.read()


def _detect_camera_pose(
    frame: Any,
    elapsed: float,
    detector_timestamp_ms: int,
    landmarker: Any,
    mp: Any,
    cv2: Any,
    inference_width: int,
) -> PoseFrame | None:
    rgb = cv2.cvtColor(_resize_for_inference(frame, cv2, inference_width), cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, detector_timestamp_ms)
    if not result.pose_landmarks:
        return None
    return PoseFrame(timestamp=elapsed, keypoints=_landmarks_to_keypoints(result.pose_landmarks[0]))


def _monotonic_timestamp_ms() -> int:
    return int(time.perf_counter() * 1000)


def _frame_at_time(sequence: PoseSequence, elapsed: float) -> PoseFrame:
    t = _loop_time(sequence, elapsed)
    timestamps = [frame.timestamp for frame in sequence.frames]
    index = int(np.searchsorted(timestamps, t, side="left"))
    if index <= 0:
        return sequence.frames[0]
    if index >= len(sequence.frames):
        return sequence.frames[-1]
    previous_frame = sequence.frames[index - 1]
    next_frame = sequence.frames[index]
    if abs(previous_frame.timestamp - t) <= abs(next_frame.timestamp - t):
        return previous_frame
    return next_frame


def _loop_time(sequence: PoseSequence, elapsed: float) -> float:
    duration = sequence.frames[-1].timestamp if sequence.frames[-1].timestamp > 0 else 1.0
    return elapsed % duration


def _compose_canvas(
    reference_frame: Any,
    camera_frame: Any,
    standard_pose: PoseFrame,
    player_pose: PoseFrame | None,
    session: _LiveSession,
    beat_tracker: _BeatTracker,
    beat_result: float | None,
    elapsed: float,
    width: int,
    height: int,
    visibility_threshold: float,
    connections: list[tuple[str, str]],
    cv2: Any,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
    header_h = 86
    gap = 12
    panel_w = (width - gap * 3) // 2
    panel_h = height - header_h - gap * 2
    left = (gap, header_h + gap, panel_w, panel_h)
    right = (gap * 2 + panel_w, header_h + gap, panel_w, panel_h)

    _paste_panel(canvas, reference_frame, left, cv2)
    _paste_panel(canvas, camera_frame, right, cv2)
    _draw_pose(canvas, standard_pose, left, visibility_threshold, connections, cv2, color=(77, 214, 181))
    if player_pose:
        _draw_pose(canvas, player_pose, right, visibility_threshold, connections, cv2, color=(246, 196, 83))

    _draw_hud(canvas, session, beat_tracker, beat_result, elapsed, left, right, cv2)
    return canvas


def _paste_panel(canvas: np.ndarray, frame: Any, rect: tuple[int, int, int, int], cv2: Any) -> None:
    x, y, w, h = rect
    resized = _resize_cover(frame, w, h, cv2)
    canvas[y : y + h, x : x + w] = resized
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (235, 241, 247), 2)


def _resize_cover(frame: Any, width: int, height: int, cv2: Any) -> Any:
    scale = max(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
    y = max(0, (resized.shape[0] - height) // 2)
    x = max(0, (resized.shape[1] - width) // 2)
    return resized[y : y + height, x : x + width]


def _draw_pose(
    canvas: np.ndarray,
    pose: PoseFrame,
    rect: tuple[int, int, int, int],
    visibility_threshold: float,
    connections: list[tuple[str, str]],
    cv2: Any,
    color: tuple[int, int, int],
) -> None:
    x, y, w, h = rect

    def point(name: str) -> tuple[int, int]:
        landmark = pose.keypoints[LANDMARK_INDEX[name]]
        return (x + int(np.clip(landmark[0], 0.0, 1.0) * w), y + int(np.clip(landmark[1], 0.0, 1.0) * h))

    visible_names = {name for pair in connections for name in pair}

    for start, end in connections:
        a = pose.keypoints[LANDMARK_INDEX[start]]
        b = pose.keypoints[LANDMARK_INDEX[end]]
        if a[3] >= visibility_threshold and b[3] >= visibility_threshold:
            cv2.line(canvas, point(start), point(end), color, 4, cv2.LINE_AA)

    for name, index in LANDMARK_INDEX.items():
        if name not in visible_names:
            continue
        landmark = pose.keypoints[index]
        if landmark[3] >= visibility_threshold:
            cv2.circle(canvas, point(name), 5, color, -1, cv2.LINE_AA)


def _scoring_config(body_mode: str, visibility_threshold: float, pose_tolerance: float) -> ScoringConfig:
    if body_mode == "upper":
        return upper_body_config(
            visibility_threshold=visibility_threshold,
            landmark_tolerance=pose_tolerance,
        )
    return ScoringConfig(
        visibility_threshold=visibility_threshold,
        landmark_tolerance=pose_tolerance,
    )


def _connections_for_mode(body_mode: str) -> list[tuple[str, str]]:
    if body_mode == "upper":
        return [
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
        ]
    return POSE_CONNECTIONS


def _draw_hud(
    canvas: np.ndarray,
    session: _LiveSession,
    beat_tracker: _BeatTracker,
    beat_result: float | None,
    elapsed: float,
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
    cv2: Any,
) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 86), (24, 30, 38), -1)
    cv2.putText(canvas, "REFERENCE VIDEO", (left[0], 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 241, 247), 2)
    cv2.putText(canvas, "CAMERA + LIVE SCORE", (right[0], 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 241, 247), 2)
    cv2.putText(
        canvas,
        f"Beat {session.last_beat_score:05.1f}  Live {session.preview:05.1f}  Avg {session.average:05.1f}  {session.last_judgement}  Combo {session.combo}  Best {session.best_combo}",
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (77, 214, 181),
        2,
        cv2.LINE_AA,
    )
    time_to_beat = max(0.0, beat_tracker.next_beat_time - elapsed)
    cv2.putText(
        canvas,
        f"Next beat in {time_to_beat:.2f}s  t={elapsed:05.1f}s",
        (canvas.shape[1] - 370, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 241, 247),
        2,
        cv2.LINE_AA,
    )
    if beat_result is not None:
        cv2.circle(canvas, (canvas.shape[1] // 2, 43), 18, (246, 196, 83), -1, cv2.LINE_AA)


def _judgement_for_score(score: float, hit_threshold: float) -> str:
    if score >= 90:
        return "Perfect"
    if score >= 75:
        return "Great"
    if score >= hit_threshold:
        return "Good"
    if score >= hit_threshold * 0.65:
        return "Close"
    return "Miss"
