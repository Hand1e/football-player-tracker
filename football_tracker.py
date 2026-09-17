#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from tracker_core import (
    RenderStyle,
    connection_pairs,
    extract_jersey_signature,
    extract_tracks,
    foot_point,
    match_roi_to_track,
    parse_named_color,
    resolve_connect_mode,
    roi_to_box,
    update_targets_with_guard,
)
from ui_tools import choose_style_ui, pick_start_frame, read_frame, select_targets


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="足球视频指定球员稳定跟踪、脚下圆环与动态连线。")
    p.add_argument("input")
    p.add_argument("-o", "--output", default="result.mp4")
    p.add_argument("--model", default="yolo26s.pt")
    p.add_argument("--tracker", default=str(Path(__file__).with_name("football_botsort.yaml")))
    p.add_argument("--targets", type=int, default=2)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--pick-start", "--select-frame", dest="pick_start", action="store_true")
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--smooth", type=float, default=0.28)
    p.add_argument("--hold-frames", type=int, default=15)
    p.add_argument("--repair-after", type=int, default=35)
    p.add_argument("--repair", action="store_true")
    p.add_argument("--connect", choices=["chain", "polygon", "all"], default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--style-ui", action="store_true")
    p.add_argument("--ring-color", type=parse_named_color, default="yellow")
    p.add_argument("--ring-thickness", type=int, default=4)
    p.add_argument("--line-color", type=parse_named_color, default="yellow")
    p.add_argument("--line-thickness", type=int, default=4)
    p.add_argument("--no-anti-switch", dest="anti_switch", action="store_false")
    p.set_defaults(anti_switch=True)
    p.add_argument("--occlusion-iou", type=float, default=0.10)
    p.add_argument("--occlusion-distance", type=float, default=1.15)
    p.add_argument("--separation-frames", type=int, default=4)
    p.add_argument("--identity-threshold", type=float, default=0.58)
    p.add_argument("--identity-margin", type=float, default=0.06)
    p.add_argument("--reid-gate", type=float, default=2.6)
    return p.parse_args()


def validate(args) -> None:
    if not 1 <= args.targets <= 26:
        raise ValueError("--targets 必须在 1~26")
    if not 0 < args.smooth <= 1:
        raise ValueError("--smooth 必须 >0 且 <=1")
    if not 1 <= args.ring_thickness <= 10 or not 1 <= args.line_thickness <= 10:
        raise ValueError("圆环/连线粗细必须在 1~10")


def draw_line(frame, p1, p2, style: RenderStyle, uncertain: bool) -> None:
    overlay = frame.copy()
    cv2.line(overlay, p1, p2, style.line_color, style.line_thickness, cv2.LINE_AA)
    alpha = 0.38 if uncertain else 0.62
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_ring(frame, target, style: RenderStyle) -> None:
    if target.point is None:
        return
    overlay = frame.copy()
    cv2.ellipse(
        overlay,
        tuple(map(int, target.point)),
        (target.ring_rx, target.ring_ry),
        0, 0, 360,
        style.ring_color,
        style.ring_thickness,
        cv2.LINE_AA,
    )
    alpha = 0.50 if target.uncertain else 0.82
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def manual_rebind(frame, target, tracks, other_ids) -> bool:
    print(f"[修复] Player {target.name} 长时间无法确认，请重新框选。")
    title = f"REPAIR Player {target.name}"
    roi = cv2.selectROI(title, frame, True, False)
    cv2.destroyWindow(title)
    if roi[2] <= 1 or roi[3] <= 1:
        return False
    new_id = match_roi_to_track(roi_to_box(roi), tracks, other_ids)
    if new_id is None:
        return False
    old_id, box = target.track_id, tracks[new_id]
    target.track_id = new_id
    target.update(foot_point(box), box, 1.0, extract_jersey_signature(frame, box), True)
    target.needs_reacquire = target.confusion_lock = target.uncertain = False
    target.separation_frames = 0
    print(f"[修复] Player {target.name}: ID {old_id} -> {new_id}")
    return True


def mux_audio(video_only: Path, source: Path, output: Path, start_seconds: float) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_only)]
    if start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]
    cmd += [
        "-i", str(source), "-map", "0:v:0", "-map", "1:a?",
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(output),
    ]
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    args = parse_args()
    try:
        validate(args)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    tracker_path = Path(args.tracker).expanduser().resolve()
    if not input_path.exists() or not tracker_path.exists():
        print("输入视频或 tracker YAML 不存在。", file=sys.stderr)
        return 2
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print("无法打开输入视频。", file=sys.stderr)
        return 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(args.start * fps)))
    if total_frames > 0:
        start_frame = min(start_frame, total_frames - 1)

    try:
        if args.pick_start:
            start_frame, first_frame = pick_start_frame(cap, fps, total_frames, start_frame)
        else:
            first_frame = read_frame(cap, start_frame)
            if first_frame is None:
                raise RuntimeError("无法读取起始帧")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + 1)
    except (RuntimeError, KeyboardInterrupt) as e:
        print(f"[取消] {e}")
        cap.release()
        cv2.destroyAllWindows()
        return 1

    h, w = first_frame.shape[:2]
    model = YOLO(args.model)
    kwargs = dict(
        persist=True, tracker=str(tracker_path), classes=[0], conf=args.conf,
        iou=args.iou, imgsz=args.imgsz, verbose=False,
    )
    if args.device is not None:
        kwargs["device"] = args.device

    first_result = model.track(first_frame, **kwargs)[0]
    first_tracks = extract_tracks(first_result)
    if not first_tracks:
        print("起始帧没有检测到 person；请换帧、降低 --conf 或增大 --imgsz。", file=sys.stderr)
        cap.release()
        return 2

    targets = select_targets(first_frame, first_tracks, args.targets)
    mode = resolve_connect_mode(len(targets), args.connect)
    pairs = connection_pairs(len(targets), mode)
    print(f"[连线] {mode}")

    style = RenderStyle(args.ring_color, args.ring_thickness, args.line_color, args.line_thickness)
    if args.style_ui:
        style = choose_style_ui(style)

    video_only = output_path.with_name(output_path.stem + ".video_only.mp4")
    writer = cv2.VideoWriter(str(video_only), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return 2

    frame, result = first_frame, first_result
    try:
        while True:
            tracks = extract_tracks(result)
            update_targets_with_guard(frame, targets, tracks, args, w, h)
            if args.repair:
                for target in targets:
                    if target.missed == args.repair_after:
                        other = {t.track_id for t in targets if t.name != target.name and not t.uncertain}
                        manual_rebind(frame, target, tracks, other)

            rendered = frame.copy()
            for i, j in pairs:
                a, b = targets[i], targets[j]
                if (
                    a.point is not None and b.point is not None
                    and a.missed <= args.hold_frames and b.missed <= args.hold_frames
                ):
                    draw_line(
                        rendered,
                        tuple(map(int, a.point)),
                        tuple(map(int, b.point)),
                        style,
                        a.uncertain or b.uncertain,
                    )
            for target in targets:
                if target.point is not None and target.missed <= args.hold_frames:
                    draw_ring(rendered, target, style)
            writer.write(rendered)

            if args.show:
                preview = rendered
                if preview.shape[1] > 1400:
                    scale = 1400 / preview.shape[1]
                    preview = cv2.resize(preview, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cv2.imshow("Football Player Tracker - q/ESC to stop", preview)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

            ok, frame = cap.read()
            if not ok:
                break
            result = model.track(frame, **kwargs)[0]
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    if mux_audio(video_only, input_path, output_path, start_frame / fps):
        video_only.unlink(missing_ok=True)
    else:
        if output_path.exists():
            output_path.unlink()
        video_only.replace(output_path)
    print(f"[完成] {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
