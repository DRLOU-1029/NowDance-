from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from .collect import collect_pose_sequence_from_video
from .io import load_sequence, save_json, save_sequence
from .live import run_live_scoring
from .scoring import ScoringConfig, score_sequences
from .visualize import render_sequence_video


def main() -> None:
    parser = argparse.ArgumentParser(prog="nowdance", description="NowDance 算法验证工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="从舞蹈视频采集姿态序列")
    collect_parser.add_argument("--video", required=True, help="输入舞蹈视频路径")
    collect_parser.add_argument("--out", required=True, help="输出标准动作 JSON 路径")
    collect_parser.add_argument("--sample-fps", type=float, default=15.0, help="采样帧率")
    collect_parser.add_argument("--max-frames", type=int, default=None, help="最多采集帧数")
    collect_parser.add_argument("--task-model", default="models/pose_landmarker_lite.task", help="MediaPipe Tasks 模型路径")
    collect_parser.add_argument("--inference-width", type=int, default=720, help="姿态推理时的视频缩放宽度")

    score_parser = subparsers.add_parser("score", help="对玩家动作进行评分")
    score_parser.add_argument("--standard", required=True, help="标准动作 JSON 路径")
    score_source = score_parser.add_mutually_exclusive_group(required=True)
    score_source.add_argument("--player-video", help="玩家舞蹈视频路径")
    score_source.add_argument("--player-sequence", help="玩家动作 JSON 路径")
    score_parser.add_argument("--out", help="评分报告 JSON 路径")
    score_parser.add_argument("--sample-fps", type=float, default=15.0, help="玩家视频采样帧率")
    score_parser.add_argument("--task-model", default="models/pose_landmarker_lite.task", help="MediaPipe Tasks 模型路径")
    score_parser.add_argument("--inference-width", type=int, default=720, help="姿态推理时的视频缩放宽度")
    score_parser.add_argument("--visibility-threshold", type=float, default=0.35)
    score_parser.add_argument("--dtw-band-ratio", type=float, default=0.25)

    visualize_parser = subparsers.add_parser("visualize", help="将姿态序列渲染为骨架动画视频")
    visualize_parser.add_argument("--sequence", required=True, help="输入动作 JSON 路径")
    visualize_parser.add_argument("--out", required=True, help="输出 MP4 路径")
    visualize_parser.add_argument("--width", type=int, default=720, help="输出视频宽度")
    visualize_parser.add_argument("--height", type=int, default=1280, help="输出视频高度")
    visualize_parser.add_argument("--fps", type=float, default=None, help="输出视频帧率，默认使用序列采样帧率")
    visualize_parser.add_argument("--visibility-threshold", type=float, default=0.35)


    quality_parser = subparsers.add_parser("quality", help="????????????????")
    quality_parser.add_argument("--chart", default="charts/jljt_chart.json", help="?? JSON ??")
    quality_parser.add_argument("--video", required=True, help="?????????????")
    quality_parser.add_argument("--sequence", default="charts/jljt.json", help="?????? JSON ?????????????")
    quality_parser.add_argument("--camera-index", type=int, default=0, help="?????")
    quality_parser.add_argument("--task-model", default="models/pose_landmarker_lite.task", help="MediaPipe Tasks ????")
    quality_parser.add_argument("--canvas-width", type=int, default=1280, help="????")
    quality_parser.add_argument("--canvas-height", type=int, default=720, help="????")
    quality_parser.add_argument("--inference-width", type=int, default=480, help="???????????")
    quality_parser.add_argument("--process-every", type=int, default=1, help="????????????")
    quality_parser.add_argument("--no-mirror", action="store_true", help="????????")
    quality_parser.add_argument("--no-audio", action="store_true", help="?????????")
    live_parser = subparsers.add_parser("live", help="打开摄像头进行实时评分")
    live_parser.add_argument("--standard", required=True, help="标准动作 JSON 路径")
    live_parser.add_argument("--video", required=True, help="左侧播放的标准舞蹈视频路径")
    live_parser.add_argument("--camera-index", type=int, default=0, help="摄像头编号")
    live_parser.add_argument("--task-model", default="models/pose_landmarker_lite.task", help="MediaPipe Tasks 模型路径")
    live_parser.add_argument("--canvas-width", type=int, default=1280, help="画布宽度")
    live_parser.add_argument("--canvas-height", type=int, default=720, help="画布高度")
    live_parser.add_argument("--inference-width", type=int, default=480, help="摄像头姿态推理缩放宽度")
    live_parser.add_argument("--process-every", type=int, default=1, help="每隔多少帧做一次姿态推理")
    live_parser.add_argument("--body-mode", choices=["full", "upper"], default="full", help="评分和绘制的身体范围")
    live_parser.add_argument("--bpm", type=float, default=120.0, help="节拍速度，每个节拍结算一次评分")
    live_parser.add_argument("--score-every-beats", type=int, default=1, help="每隔多少拍结算一次评分")
    live_parser.add_argument("--beat-offset", type=float, default=0.0, help="第一个判定节拍相对开场的偏移秒数")
    live_parser.add_argument("--judge-window", type=float, default=0.35, help="节拍前后的宽容判定窗口，单位秒")
    live_parser.add_argument("--hit-threshold", type=float, default=45.0, help="达到多少分算命中并延续 Combo")
    live_parser.add_argument("--pose-tolerance", type=float, default=0.75, help="姿态距离容忍度，越大越宽松")
    live_parser.add_argument("--standard-pose-delay", type=float, default=0.2, help="标准动作骨架相对视频延迟秒数，用于同步微调")
    live_parser.add_argument("--no-audio", action="store_true", help="不播放标准视频音频")
    live_parser.add_argument("--audio-player", help="ffplay 可执行文件路径")
    live_parser.add_argument("--calibration-seconds", type=float, default=3.0, help="开局校准动作需要保持的秒数")
    live_parser.add_argument("--calibration-target-y", type=float, default=-0.08, help="校准目标骨架垂直偏移，负数上移")
    live_parser.add_argument("--calibration-target-scale", type=float, default=1.0, help="校准目标骨架缩放")
    live_parser.add_argument("--calibration-mode", choices=["relaxed", "raise-hand"], default="relaxed", help="校准判定模式")
    live_parser.add_argument("--skip-calibration", action="store_true", help="跳过开局校准")
    live_parser.add_argument("--no-mirror", action="store_true", help="不镜像摄像头画面")
    live_parser.add_argument("--visibility-threshold", type=float, default=0.35)

    args = parser.parse_args()
    if args.command == "collect":
        sequence = collect_pose_sequence_from_video(
            args.video,
            sample_fps=args.sample_fps,
            max_frames=args.max_frames,
            task_model=args.task_model,
            inference_width=args.inference_width,
        )
        save_sequence(sequence, args.out)
        print(f"已采集 {len(sequence.frames)} 帧姿态，保存到 {args.out}")
        return

    if args.command == "score":
        standard = load_sequence(args.standard)
        if args.player_video:
            player = collect_pose_sequence_from_video(
                args.player_video,
                sample_fps=args.sample_fps,
                task_model=args.task_model,
                inference_width=args.inference_width,
            )
        else:
            player = load_sequence(args.player_sequence)

        config = ScoringConfig(
            visibility_threshold=args.visibility_threshold,
            dtw_band_ratio=args.dtw_band_ratio,
        )
        report = score_sequences(standard, player, config)
        if args.out:
            save_json(report, args.out)
            print(f"总分 {report['score']}，评级 {report['grade']}，报告保存到 {args.out}")
        else:
            with tempfile.NamedTemporaryFile("w", delete=True):
                print(f"总分: {report['score']}")
                print(f"评级: {report['grade']}")
                print(f"匹配帧数: {report['matched_frames']}")
        return

    if args.command == "visualize":
        sequence = load_sequence(args.sequence)
        render_sequence_video(
            sequence,
            args.out,
            width=args.width,
            height=args.height,
            fps=args.fps,
            visibility_threshold=args.visibility_threshold,
        )
        print(f"已渲染 {len(sequence.frames)} 帧骨架动画，保存到 {args.out}")
        return


    if args.command == "quality":
        from .live_quality import run_quality_live
        run_quality_live(
            chart_path=args.chart,
            reference_video=args.video,
            sequence_path=args.sequence if args.sequence != "none" else None,
            camera_index=args.camera_index,
            task_model=args.task_model,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            inference_width=args.inference_width,
            process_every=args.process_every,
            mirror_camera=not args.no_mirror,
            play_audio=not args.no_audio,
        )
        return
    if args.command == "live":
        run_live_scoring(
            load_sequence(args.standard),
            args.video,
            camera_index=args.camera_index,
            task_model=args.task_model,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            inference_width=args.inference_width,
            process_every=args.process_every,
            body_mode=args.body_mode,
            bpm=args.bpm,
            score_every_beats=args.score_every_beats,
            beat_offset=args.beat_offset,
            judge_window=args.judge_window,
            hit_threshold=args.hit_threshold,
            pose_tolerance=args.pose_tolerance,
            standard_pose_delay=args.standard_pose_delay,
            play_audio=not args.no_audio,
            audio_player=args.audio_player,
            calibration_seconds=args.calibration_seconds,
            calibration_target_y=args.calibration_target_y,
            calibration_target_scale=args.calibration_target_scale,
            calibration_mode=args.calibration_mode,
            skip_calibration=args.skip_calibration,
            mirror_camera=not args.no_mirror,
            visibility_threshold=args.visibility_threshold,
        )


if __name__ == "__main__":
    main()
