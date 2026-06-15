from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from .chart import Chart
from .collect import _landmarks_to_keypoints, _resize_for_inference
from .quality import evaluate_step, StepQualityReport
from .schema import LANDMARK_INDEX, PoseFrame, PoseSequence


HEADER_H = 110
GAP = 10


@dataclass
class StepResult:
    step_number: int
    name: str
    score: float
    grade: str
    combo: int
    golden: bool


UPPER_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


def run_quality_live(
    chart_path, reference_video,
    sequence_path=None, camera_index=0,
    task_model="models/pose_landmarker_lite.task",
    canvas_width=1280, canvas_height=720,
    inference_width=480, process_every=1,
    mirror_camera=True, visibility_threshold=0.35,
    play_audio=True,
):
    chart = Chart.load(str(chart_path))
    steps = chart.steps
    if not steps:
        raise ValueError("Chart has no steps")
    print(f"Chart: {chart.name}  BPM={chart.bpm}  {len(steps)} steps")

    standard_seq = None
    if sequence_path:
        from .io import load_sequence
        standard_seq = load_sequence(str(sequence_path))
        print(f"Standard seq: {len(standard_seq.frames)} frames")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera: {camera_index}")

    ref = cv2.VideoCapture(str(reference_video))
    if not ref.isOpened():
        camera.release()
        raise FileNotFoundError(f"Cannot open video: {reference_video}")
    ref_fps = float(ref.get(cv2.CAP_PROP_FPS) or 25.0)
    ref_total = int(ref.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    model_path = Path(task_model)
    if not model_path.exists():
        camera.release()
        ref.release()
        raise FileNotFoundError(f"Model not found: {task_model}")

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    audio = _SimpleAudioPlayer(reference_video) if play_audio else None
    window_name = "NowDance Live - press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:

            # Phase 1: hands-up countdown
            if not _hand_raise_countdown(
                camera, landmarker, mp, cv2, window_name,
                canvas_width, canvas_height, inference_width,
                mirror_camera, visibility_threshold, 3.0,
            ):
                return

                        # Phase 2 + 3: run scoring loop then show summary, repeat on restart
            while True:
                step_results = _run_scoring_loop(
                    steps, standard_seq, ref, ref_fps, ref_total,
                    camera, landmarker, mp, cv2, window_name,
                    canvas_width, canvas_height, inference_width,
                    process_every, mirror_camera, visibility_threshold,
                    audio, UPPER_CONNECTIONS,
                )
                if step_results is None:
                    break
                should_restart = _show_summary_screen(
                    camera, steps, step_results,
                    window_name, canvas_width, canvas_height,
                    mirror_camera, cv2,
                )
                if not should_restart:
                    break
                ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                if audio:
                    audio.restart()
                ok = _hand_raise_countdown(
                    camera, landmarker, mp, cv2, window_name,
                    canvas_width, canvas_height, inference_width,
                    mirror_camera, visibility_threshold, 3.0,
                )
                if not ok:
                    break


    finally:
        camera.release()
        ref.release()
        if audio:
            audio.stop()
        cv2.destroyAllWindows()


# === Phase 2: scoring loop (extracted for restartability) ===

def _run_scoring_loop(
    steps, standard_seq, ref, ref_fps, ref_total,
    camera, landmarker, mp, cv2, window_name,
    canvas_width, canvas_height, inference_width,
    process_every, mirror_camera, visibility_threshold,
    audio, connections,
):
    started_at = time.perf_counter()
    frame_counter = 0
    current_step_idx = -1
    step_cam_frames = []
    step_results = []
    combo = 0
    all_perfect = True
    all_done = False

    if audio:
        audio.start()

    while not all_done:
        elapsed = time.perf_counter() - started_at

        if elapsed > (steps[-1].end_time + 0.5):
            if current_step_idx >= 0 and step_cam_frames:
                combo, all_perfect = _finalize_and_push(
                    steps[current_step_idx], step_cam_frames,
                    step_results, visibility_threshold, combo, all_perfect,
                )
            all_done = True
            break

        new_idx = _find_current_step(steps, elapsed)

        if new_idx != current_step_idx:
            if current_step_idx >= 0 and step_cam_frames:
                combo, all_perfect = _finalize_and_push(
                    steps[current_step_idx], step_cam_frames,
                    step_results, visibility_threshold, combo, all_perfect,
                )
            current_step_idx = new_idx
            step_cam_frames = []

        ok_ref, ref_frame = _read_ref_frame(ref, elapsed, ref_fps, ref_total)
        ok_cam, cam_frame = camera.read()
        if not ok_ref or not ok_cam:
            break

        if mirror_camera:
            cam_frame = cv2.flip(cam_frame, 1)

        player_pose = None
        if frame_counter % max(1, process_every) == 0:
            ts_ms = int(time.perf_counter() * 1000) % (2**31)
            player_pose = _detect_pose(cam_frame, landmarker, mp, inference_width, ts_ms)
            if player_pose is not None and current_step_idx >= 0:
                step_cam_frames.append(player_pose)

        standard_pose = None
        if standard_seq is not None:
            standard_pose = _frame_at_time(standard_seq, elapsed)

        canvas = _compose_canvas(
            ref_frame, cam_frame,
            standard_pose, player_pose,
            steps, step_results,
            current_step_idx, combo, all_perfect,
            elapsed, canvas_width, canvas_height,
            visibility_threshold, connections, cv2,
        )
        cv2.imshow(window_name, canvas)
        frame_counter += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            all_done = True
            break
        if key in (ord("r"), ord("R")):
            started_at = time.perf_counter()
            frame_counter = 0
            current_step_idx = -1
            step_cam_frames.clear()
            step_results.clear()
            combo = 0
            all_perfect = True
            ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
            if audio:
                audio.restart()

    return step_results if step_results else None


# === Phase 1: hands-up countdown ===

def _hand_raise_countdown(
    camera, landmarker, mp, cv2, window_name,
    width, height, infer_w, mirror, vis_thresh, hold_seconds,
):
    hold_start = None
    while True:
        ok, frame = camera.read()
        if not ok:
            return False
        if mirror:
            frame = cv2.flip(frame, 1)

        now = time.perf_counter()
        ts_ms = int(now * 1000) % (2**31)
        pose = _detect_pose(frame, landmarker, mp, infer_w, ts_ms)

        hands_up = False
        if pose is not None and _both_hands_raised(pose, vis_thresh):
            hands_up = True
            if hold_start is None:
                hold_start = now
        else:
            hold_start = None

        held = (now - hold_start) if hold_start is not None else 0.0
        progress = min(1.0, held / max(0.1, hold_seconds))

        canvas = _compose_raise_screen(
            frame, pose, width, height,
            hands_up, held, hold_seconds, progress, vis_thresh, cv2,
        )
        cv2.imshow(window_name, canvas)

        if held >= hold_seconds:
            flash = canvas.copy()
            cv2.putText(
                flash, "START!", (width // 2 - 120, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (77, 214, 181), 4, cv2.LINE_AA,
            )
            cv2.imshow(window_name, flash)
            cv2.waitKey(600)
            return True

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("s"), ord("S")):
            return True


def _both_hands_raised(pose, vis_thresh):
    required = ["left_shoulder", "right_shoulder", "left_wrist", "right_wrist"]
    relaxed = max(0.15, vis_thresh * 0.55)
    for name in required:
        if pose.keypoints[LANDMARK_INDEX[name]][3] < relaxed:
            return False
    ls = pose.keypoints[LANDMARK_INDEX["left_shoulder"]]
    rs = pose.keypoints[LANDMARK_INDEX["right_shoulder"]]
    lw = pose.keypoints[LANDMARK_INDEX["left_wrist"]]
    rw = pose.keypoints[LANDMARK_INDEX["right_wrist"]]
    if abs(ls[0] - rs[0]) < 0.045:
        return False
    return lw[1] < ls[1] and rw[1] < rs[1]


def _compose_raise_screen(
    frame, pose, width, height,
    hands_up, held, hold_seconds, progress, vis_thresh, cv2,
):
    canvas = np.full((height, width, 3), (12, 16, 22), dtype=np.uint8)
    f = _resize_cover(frame, width, height, cv2)
    overlay = f.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    canvas = cv2.addWeighted(f, 0.72, overlay, 0.28, 0)

    if pose:
        _draw_pose(
            canvas, pose, (0, 0, width, height),
            vis_thresh, UPPER_CONNECTIONS, cv2, (246, 196, 83),
        )

    _draw_raise_guide(canvas, width, height, hands_up, cv2)

    color = (77, 214, 181) if hands_up else (90, 110, 135)
    cv2.putText(
        canvas, "PUT BOTH HANDS UP",
        (width // 2 - 200, 100),
        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (245, 248, 252), 3, cv2.LINE_AA,
    )

    bar_y = height - 60
    cv2.rectangle(canvas, (80, bar_y), (width - 80, bar_y + 28), (36, 43, 52), -1)
    cv2.rectangle(
        canvas, (80, bar_y),
        (80 + int((width - 160) * progress), bar_y + 28), color, -1,
    )
    cv2.putText(
        canvas,
        f"Hold {held:.1f}/{hold_seconds:.0f}s   S to skip   Q to quit",
        (80, bar_y - 16),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return canvas


def _draw_raise_guide(canvas, width, height, matched, cv2):
    color = (85, 255, 95) if matched else (90, 110, 135)
    cx, cy = width * 0.50, height * 0.38
    pts = {
        "ls": (int(cx - width * 0.08), int(cy)),
        "rs": (int(cx + width * 0.08), int(cy)),
        "lw": (int(cx - width * 0.16), int(cy - height * 0.18)),
        "rw": (int(cx + width * 0.16), int(cy - height * 0.18)),
        "lh": (int(cx - width * 0.05), int(cy + height * 0.22)),
        "rh": (int(cx + width * 0.05), int(cy + height * 0.22)),
    }
    for a, b in [
        ("ls", "rs"), ("rs", "rw"), ("ls", "lw"),
        ("ls", "lh"), ("rs", "rh"), ("lh", "rh"),
    ]:
        cv2.line(canvas, pts[a], pts[b], color, 6, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(canvas, p, 8, color, -1, cv2.LINE_AA)


# === Phase 3: summary screen ===

def _show_summary_screen(camera, steps, results, window_name, width, height, mirror, cv2):
    bg = None
    for _ in range(10):
        ok, frame = camera.read()
        if ok:
            bg = frame.copy()
            if mirror:
                bg = cv2.flip(bg, 1)
    if bg is None:
        bg = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)

    if not results:
        canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)
        cv2.putText(
            canvas, "No scores recorded", (width // 2 - 150, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2, cv2.LINE_AA,
        )
        while True:
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                return False
            if key in (ord("r"), ord("R")):
                return True

    total = sum(r.score for r in results)
    avg = total / len(results)
    perfect_count = sum(1 for r in results if r.grade == "Perfect")
    max_combo = max(r.combo for r in results)
    all_golden = all(r.golden for r in results)

    if avg >= 90:
        rank = "S"
        rank_color = (77, 214, 181)
    elif avg >= 75:
        rank = "A"
        rank_color = (246, 196, 83)
    elif avg >= 60:
        rank = "B"
        rank_color = (235, 241, 247)
    else:
        rank = "C"
        rank_color = (200, 200, 200)

    while True:
        canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)

        bg_small = _resize_cover(bg, width, height, cv2)
        overlay = bg_small.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        canvas = cv2.addWeighted(bg_small, 0.4, overlay, 0.6, 0)

        cv2.putText(
            canvas, "FINAL RESULT", (width // 2 - 160, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (245, 248, 252), 3, cv2.LINE_AA,
        )

        cv2.putText(
            canvas, rank, (width // 2 - 60, 160),
            cv2.FONT_HERSHEY_SIMPLEX, 4.0, rank_color, 6, cv2.LINE_AA,
        )

        stats_y = 200
        line_h = 36
        cv2.putText(
            canvas, f"Average Score:  {avg:.1f}",
            (width // 2 - 180, stats_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, rank_color, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"Perfect:  {perfect_count}/{len(results)}",
            (width // 2 - 180, stats_y + line_h), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (235, 241, 247), 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"Max Combo:  {max_combo}",
            (width // 2 - 180, stats_y + line_h * 2), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (235, 241, 247), 2, cv2.LINE_AA,
        )

        if all_golden:
            cv2.putText(
                canvas, "ALL GOLDEN COMBO!",
                (width // 2 - 180, stats_y + line_h * 3),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 215, 0), 3, cv2.LINE_AA,
            )

        col_x = [80, width // 2 + 40]
        per_col = (len(results) + 1) // 2
        list_top = stats_y + line_h * 4 + 20

        for idx, r in enumerate(results):
            col = idx // per_col
            row = idx % per_col
            x = col_x[col]
            y = list_top + row * 22
            g = "G" if r.golden else " "
            cv2.putText(
                canvas, f"Step {r.step_number:2d}  {r.score:5.1f}  {r.grade:7s}  C{r.combo:2d}{g}",
                (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
            )

        cv2.putText(
            canvas, "Press Q to quit  |  R to restart",
            (width // 2 - 160, height - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1, cv2.LINE_AA,
        )

        cv2.imshow(window_name, canvas)
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            return False
        if key in (ord("r"), ord("R")):
            return True


# === Helpers ===

def _find_current_step(steps, elapsed):
    for idx, step in enumerate(steps):
        if step.start_time <= elapsed <= step.end_time:
            return idx
        elif elapsed < step.start_time:
            return -1
    return -1


def _finalize_and_push(step, cam_frames, results, vis_thresh, combo, all_perfect):
    report = evaluate_step(step, cam_frames, {"visibility_threshold": vis_thresh})
    if report.grade == "Miss":
        combo = 0
        all_perfect = False
    else:
        combo += 1
        if report.grade != "Perfect":
            all_perfect = False
    results.append(StepResult(
        step_number=step.step_number,
        name=step.name,
        score=report.score,
        grade=report.grade,
        combo=combo,
        golden=all_perfect and combo > 0,
    ))
    return combo, all_perfect


def _read_ref_frame(capture, elapsed, fps, total):
    if total > 0:
        target = int(elapsed * fps) % total
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
    return capture.read()


def _detect_pose(frame, landmarker, mp, inference_width, timestamp_ms):
    rgb = cv2.cvtColor(
        _resize_for_inference(frame, cv2, inference_width), cv2.COLOR_BGR2RGB,
    )
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(img, timestamp_ms)
    if result.pose_landmarks:
        return PoseFrame(
            timestamp=timestamp_ms / 1000.0,
            keypoints=_landmarks_to_keypoints(result.pose_landmarks[0]),
        )
    return None


def _frame_at_time(seq, t):
    if not seq.frames:
        return None
    duration = seq.frames[-1].timestamp
    t = t % duration if duration > 0 else 0.0
    ts = np.asarray([f.timestamp for f in seq.frames], dtype=np.float32)
    idx = int(np.argmin(np.abs(ts - t)))
    return seq.frames[idx]


def _resize_cover(frame, width, height, cv2):
    scale = max(width / frame.shape[1], height / frame.shape[0])
    resized = cv2.resize(
        frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)),
    )
    cy = max(0, (resized.shape[0] - height) // 2)
    cx = max(0, (resized.shape[1] - width) // 2)
    return resized[cy:cy+height, cx:cx+width]


# === Drawing ===

def _compose_canvas(
    ref_frame, cam_frame,
    standard_pose, player_pose,
    steps, results,
    current_idx, combo, all_perfect,
    elapsed, width, height,
    vis_thresh, connections, cv2,
):
    canvas = np.full((height, width, 3), (18, 22, 28), dtype=np.uint8)

    panel_w = (width - GAP * 3) // 2
    panel_h = height - HEADER_H - GAP * 2
    left = (GAP, HEADER_H + GAP, panel_w, panel_h)
    right = (GAP * 2 + panel_w, HEADER_H + GAP, panel_w, panel_h)

    _paste_panel(canvas, ref_frame, left, cv2)
    _paste_panel(canvas, cam_frame, right, cv2)

    if standard_pose:
        _draw_pose(canvas, standard_pose, left, vis_thresh, connections, cv2, (77, 214, 181))
    if player_pose:
        _draw_pose(canvas, player_pose, right, vis_thresh, connections, cv2, (246, 196, 83))

    h = HEADER_H
    cv2.rectangle(canvas, (0, 0), (width, h), (24, 30, 38), -1)

    cur_step = steps[current_idx] if 0 <= current_idx < len(steps) else None
    if cur_step:
        step_text = f"Step {cur_step.step_number}/{len(steps)}:  {cur_step.name}"
        cv2.putText(
            canvas, step_text, (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (246, 196, 83), 2, cv2.LINE_AA,
        )
        progress = (elapsed - cur_step.start_time) / max(0.001, cur_step.end_time - cur_step.start_time)
        progress = min(1.0, max(0.0, progress))
        bar_w = width - 32
        cv2.rectangle(
            canvas, (16, h - 8), (16 + int(bar_w * progress), h - 2), (77, 214, 181), -1,
        )

    last = results[-1] if results else None
    if last:
        combo_text = f"Combo {last.combo}"
        if last.golden:
            combo_text += "  GOLDEN"
        cv2.putText(
            canvas, f"Score: {last.score:.1f}  {last.grade}  {combo_text}",
            (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (77, 214, 181), 2, cv2.LINE_AA,
        )

    cv2.putText(
        canvas, f"t={elapsed:.2f}s", (width - 160, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA,
    )
    return canvas


def _paste_panel(canvas, frame, rect, cv2):
    x, y, w, h = rect
    cropped = _resize_cover(frame, w, h, cv2)
    canvas[y:y+h, x:x+w] = cropped
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (235, 241, 247), 2)


def _draw_pose(canvas, pose, rect, vis_thresh, connections, cv2, color):
    x, y, w, h = rect

    def pt(name):
        lm = pose.keypoints[LANDMARK_INDEX[name]]
        return (x + int(np.clip(lm[0], 0.0, 1.0) * w), y + int(np.clip(lm[1], 0.0, 1.0) * h))

    for a, b in connections:
        ka = pose.keypoints[LANDMARK_INDEX[a]]
        kb = pose.keypoints[LANDMARK_INDEX[b]]
        if ka[3] >= vis_thresh and kb[3] >= vis_thresh:
            cv2.line(canvas, pt(a), pt(b), color, 4, cv2.LINE_AA)

    vis = {n for pair in connections for n in pair}
    for name in vis:
        lm = pose.keypoints[LANDMARK_INDEX[name]]
        if lm[3] >= vis_thresh:
            cv2.circle(canvas, pt(name), 5, color, -1, cv2.LINE_AA)


# === Audio ===

class _SimpleAudioPlayer:
    def __init__(self, video_path):
        import shutil
        self.path = str(video_path)
        self.player = shutil.which("ffplay")
        self.process = None
        self.enabled = self.player is not None
        if not self.enabled:
            print("ffplay not found, no audio will play.")

    def start(self):
        if not self.enabled:
            return
        import subprocess, sys
        if sys.platform.startswith("win"):
            flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
            self.process = subprocess.Popen(
                [self.player, "-nodisp", "-autoexit", "-loglevel", "error", self.path],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                creationflags=flags,
            )

    def restart(self):
        self.stop()
        self.start()

    def stop(self):
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
