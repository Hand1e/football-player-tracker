#!/usr/bin/env python3
"""
football_tracker.py

用途：
1. 打开足球视频
2. 在起始帧手动框选几个特定球员
3. YOLO + BoT-SORT 持续跟踪
4. 在球员脚下画半透明椭圆环
5. 在目标球员之间画稳定连线
6. EMA 平滑坐标，短时丢失时做简单运动预测
7. 可选：长时间跟丢时自动暂停，让你重新框选该球员
8. 若系统安装了 ffmpeg，则把原视频音频自动合并回输出视频

依赖：
    pip install ultralytics opencv-python numpy

示例：
    python football_tracker.py input.mp4 --targets 3 --show --repair

按键（--show 时）：
    q / ESC : 提前结束
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Point = Tuple[float, float]
Box = Tuple[float, float, float, float]


@dataclass
class TargetState:
    name: str
    track_id: int
    point: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    box_w: float = 30.0
    box_h: float = 80.0
    missed: int = 0

    def update(self, new_point: Point, box: Box, alpha: float) -> None:
        p = np.array(new_point, dtype=np.float32)
        x1, y1, x2, y2 = box
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        if self.point is None:
            self.point = p
            self.velocity = np.zeros(2, dtype=np.float32)
            self.box_w = w
            self.box_h = h
        else:
            old = self.point.copy()
            smoothed = alpha * p + (1.0 - alpha) * old
            delta = smoothed - old

            if self.velocity is None:
                self.velocity = delta
            else:
                # 速度也做平滑，避免预测阶段突然冲出去
                self.velocity = 0.35 * delta + 0.65 * self.velocity

            self.point = smoothed
            self.box_w = alpha * w + (1.0 - alpha) * self.box_w
            self.box_h = alpha * h + (1.0 - alpha) * self.box_h

        self.missed = 0

    def predict(self, frame_w: int, frame_h: int) -> None:
        self.missed += 1
        if self.point is None:
            return

        if self.velocity is not None:
            # 短时丢失：做阻尼匀速预测
            self.velocity *= 0.88
            self.point += self.velocity

        self.point[0] = np.clip(self.point[0], 0, frame_w - 1)
        self.point[1] = np.clip(self.point[1], 0, frame_h - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="手动选择足球运动员，并在运动过程中稳定画圈和连线。"
    )
    parser.add_argument("input", help="输入视频，例如 input.mp4")
    parser.add_argument(
        "-o", "--output", default="result.mp4", help="输出视频，默认 result.mp4"
    )
    parser.add_argument(
        "--model",
        default="yolo26s.pt",
        help="Ultralytics YOLO 模型。默认 yolo26s.pt；速度优先可用 yolo26n.pt",
    )
    parser.add_argument(
        "--tracker",
        default=str(Path(__file__).with_name("football_botsort.yaml")),
        help="Tracker YAML，默认使用随脚本提供的 football_botsort.yaml",
    )
    parser.add_argument(
        "--targets", type=int, default=3, help="要选择的特定球员数量，默认 3"
    )
    parser.add_argument(
        "--start", type=float, default=0.0, help="从视频第几秒开始选择/处理，默认 0"
    )
    parser.add_argument(
        "--conf", type=float, default=0.12, help="人物检测置信度，默认 0.12"
    )
    parser.add_argument(
        "--iou", type=float, default=0.70, help="YOLO NMS IoU，默认 0.70"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="推理尺寸。远景足球建议 1280；显存不足可改 960/640",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.28,
        help="EMA 平滑系数 0~1。越小越稳但延迟越明显，默认 0.28",
    )
    parser.add_argument(
        "--hold-frames",
        type=int,
        default=15,
        help="目标短暂消失时继续预测/显示多少帧，默认 15",
    )
    parser.add_argument(
        "--repair-after",
        type=int,
        default=35,
        help="--repair 开启时，连续丢失多少帧后暂停人工重绑，默认 35",
    )
    parser.add_argument(
        "--repair",
        action="store_true",
        help="长时间跟丢后自动暂停，让你重新框选该球员",
    )
    parser.add_argument(
        "--connect",
        choices=["chain", "polygon", "all"],
        default="polygon",
        help="连线方式：chain=A-B-C；polygon=首尾闭合；all=两两相连",
    )
    parser.add_argument(
        "--show", action="store_true", help="处理时显示实时预览窗口"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="设备，例如 0、cpu、mps。不填则交给 Ultralytics 自动处理",
    )
    return parser.parse_args()


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def roi_to_box(roi: Sequence[float]) -> Box:
    x, y, w, h = map(float, roi)
    return (x, y, x + w, y + h)


def foot_point(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def extract_tracks(result) -> Dict[int, Box]:
    """
    从 Ultralytics Results 中提取 {track_id: xyxy_box}。
    只在 boxes.id 存在时返回。
    """
    if result.boxes is None or result.boxes.id is None:
        return {}

    boxes = result.boxes.xyxy.detach().cpu().numpy()
    ids = result.boxes.id.detach().cpu().numpy().astype(int)

    return {
        int(track_id): tuple(map(float, box))
        for track_id, box in zip(ids, boxes)
    }


def match_roi_to_track(
    roi: Box,
    tracks: Dict[int, Box],
    forbidden_ids: Optional[set[int]] = None,
) -> Optional[int]:
    """
    把人工 ROI 匹配到当前 tracker 的 ID。
    主要看 IoU，同时对“检测框中心落在人工 ROI 内”给予额外权重。
    """
    forbidden_ids = forbidden_ids or set()
    rx1, ry1, rx2, ry2 = roi

    best_id = None
    best_score = -1.0

    rcx = (rx1 + rx2) / 2.0
    rcy = (ry1 + ry2) / 2.0
    rdiag = max(1.0, math.hypot(rx2 - rx1, ry2 - ry1))

    for track_id, box in tracks.items():
        if track_id in forbidden_ids:
            continue

        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        iou = box_iou(roi, box)
        center_inside = rx1 <= cx <= rx2 and ry1 <= cy <= ry2
        dist = math.hypot(cx - rcx, cy - rcy) / rdiag

        score = iou + (0.65 if center_inside else 0.0) - 0.08 * dist

        if score > best_score:
            best_score = score
            best_id = track_id

    # 分数太差时宁愿要求重选，避免一开始就绑定错人
    return best_id if best_score >= 0.05 else None


def select_targets(
    frame: np.ndarray,
    tracks: Dict[int, Box],
    count: int,
) -> List[TargetState]:
    if count < 1 or count > 26:
        raise ValueError("--targets 必须在 1~26 之间")

    targets: List[TargetState] = []
    used_ids: set[int] = set()

    for i in range(count):
        name = chr(ord("A") + i)

        while True:
            title = f"Select Player {name} - drag box, ENTER/SPACE confirm, C cancel"
            roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(title)

            if roi[2] <= 1 or roi[3] <= 1:
                print(f"[选择] Player {name} 未选择，重新来。")
                continue

            manual_box = roi_to_box(roi)
            track_id = match_roi_to_track(manual_box, tracks, used_ids)

            if track_id is None:
                print(
                    f"[选择] Player {name} 的框没有匹配到 YOLO/Tracker 检测。"
                    "请把框画得稍微大一点，完整包含球员。"
                )
                continue

            used_ids.add(track_id)
            box = tracks[track_id]
            target = TargetState(name=name, track_id=track_id)
            target.update(foot_point(box), box, alpha=1.0)
            targets.append(target)
            print(f"[绑定] Player {name} -> Track ID {track_id}")
            break

    return targets


def connection_pairs(targets: Sequence[TargetState], mode: str):
    n = len(targets)
    if n < 2:
        return []

    if mode == "all":
        return list(combinations(range(n), 2))

    pairs = [(i, i + 1) for i in range(n - 1)]
    if mode == "polygon" and n >= 3:
        pairs.append((n - 1, 0))
    return pairs


def draw_alpha_line(
    frame: np.ndarray,
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    color=(0, 220, 255),
    thickness=4,
    alpha=0.58,
):
    overlay = frame.copy()
    cv2.line(overlay, p1, p2, color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_target(
    frame: np.ndarray,
    target: TargetState,
    detected_now: bool,
):
    if target.point is None:
        return

    x, y = map(int, target.point)

    # 圆环大小随球员 bbox 宽度变化；远景小、近景大
    rx = int(np.clip(target.box_w * 0.55, 13, 46))
    ry = max(6, int(rx * 0.34))

    # 正常检测为亮黄；预测中的目标略暗，提醒你它暂时没被 detector 看见
    color = (0, 230, 255) if detected_now else (80, 170, 200)

    overlay = frame.copy()
    cv2.ellipse(
        overlay,
        (x, y),
        (rx, ry),
        0,
        0,
        360,
        color,
        4,
        cv2.LINE_AA,
    )
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # 标签放在球员上方
    label_y = max(24, int(y - target.box_h - 10))
    label = target.name
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.72
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

    x1 = max(0, x - tw // 2 - 7)
    y1 = max(0, label_y - th - 7)
    x2 = min(frame.shape[1] - 1, x + tw // 2 + 7)
    y2 = min(frame.shape[0] - 1, label_y + baseline + 5)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.putText(
        frame,
        label,
        (x - tw // 2, label_y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def manual_rebind(
    frame: np.ndarray,
    target: TargetState,
    tracks: Dict[int, Box],
    other_ids: set[int],
) -> bool:
    """
    长时间跟丢时，暂停让用户重新框选。
    新框会绑定到当前 tracker 的另一个 Track ID。
    """
    print(
        f"\n[修复] Player {target.name} 已连续丢失较久。"
        "请在弹出的当前帧重新框选这个球员。"
    )

    title = f"REPAIR Player {target.name} - select again"
    roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)

    if roi[2] <= 1 or roi[3] <= 1:
        print(f"[修复] Player {target.name} 跳过人工重绑。")
        return False

    new_id = match_roi_to_track(roi_to_box(roi), tracks, other_ids)
    if new_id is None:
        print("[修复] 当前框仍未匹配到有效 Track ID。继续处理。")
        return False

    old_id = target.track_id
    target.track_id = new_id
    box = tracks[new_id]
    target.update(foot_point(box), box, alpha=1.0)

    print(f"[修复] Player {target.name}: ID {old_id} -> ID {new_id}")
    return True


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def mux_audio(
    video_only: Path,
    source_video: Path,
    output_path: Path,
    start_seconds: float,
) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False

    cmd = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(video_only),
    ]

    if start_seconds > 0:
        cmd += ["-ss", f"{start_seconds:.3f}"]

    cmd += [
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def main() -> int:
    args = parse_args()

    if not (0.0 < args.smooth <= 1.0):
        print("--smooth 必须 > 0 且 <= 1", file=sys.stderr)
        return 2

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    tracker_path = Path(args.tracker).expanduser().resolve()

    if not input_path.exists():
        print(f"找不到输入视频: {input_path}", file=sys.stderr)
        return 2

    if not tracker_path.exists():
        print(f"找不到 tracker YAML: {tracker_path}", file=sys.stderr)
        return 2

    ensure_parent(output_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"无法打开视频: {input_path}", file=sys.stderr)
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(args.start * fps)))
    if total_frames > 0:
        start_frame = min(start_frame, max(0, total_frames - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ok, first_frame = cap.read()
    if not ok:
        print("无法读取起始帧。", file=sys.stderr)
        cap.release()
        return 2

    height, width = first_frame.shape[:2]

    print("[模型] 加载 YOLO...")
    model = YOLO(args.model)

    track_kwargs = dict(
        persist=True,
        tracker=str(tracker_path),
        classes=[0],  # COCO person
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False,
    )
    if args.device is not None:
        track_kwargs["device"] = args.device

    print("[检测] 分析选择帧...")
    first_result = model.track(first_frame, **track_kwargs)[0]
    first_tracks = extract_tracks(first_result)

    if not first_tracks:
        print(
            "起始帧没有检测到 person。尝试：\n"
            "  1) 用 --start 选择球员更清楚的时间点\n"
            "  2) 把 --conf 降低，例如 --conf 0.08\n"
            "  3) 增大 --imgsz，例如 1600\n",
            file=sys.stderr,
        )
        cap.release()
        return 2

    print(
        f"[选择] 当前帧检测到 {len(first_tracks)} 个 person track。\n"
        "接下来逐个框选你要跟踪的球员。框尽量完整覆盖该球员。"
    )

    targets = select_targets(first_frame, first_tracks, args.targets)
    pairs = connection_pairs(targets, args.connect)

    # 先写无音频临时视频，最后有 ffmpeg 的话再把原音频 mux 回来
    video_only = output_path.with_name(output_path.stem + ".video_only.mp4")
    writer = cv2.VideoWriter(
        str(video_only),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        print("无法创建输出视频。", file=sys.stderr)
        cap.release()
        return 2

    # first_frame 已经走过一次 tracker，直接从它开始写
    current_result = first_result
    current_frame = first_frame
    frame_index = start_frame

    last_progress = -1

    try:
        while True:
            tracks = extract_tracks(current_result)
            detected_ids = set(tracks.keys())

            # 更新目标状态
            for target in targets:
                if target.track_id in tracks:
                    box = tracks[target.track_id]
                    target.update(
                        foot_point(box),
                        box,
                        alpha=args.smooth,
                    )
                else:
                    target.predict(width, height)

            # 可选人工修复：长时间丢失后重新框一下
            if args.repair:
                for target in targets:
                    if target.missed == args.repair_after:
                        other_ids = {
                            t.track_id
                            for t in targets
                            if t.name != target.name and t.missed == 0
                        }
                        manual_rebind(
                            current_frame,
                            target,
                            tracks,
                            other_ids,
                        )

            rendered = current_frame.copy()

            # 先画线，让圈压在线上面
            for i, j in pairs:
                a = targets[i]
                b = targets[j]

                if (
                    a.point is not None
                    and b.point is not None
                    and a.missed <= args.hold_frames
                    and b.missed <= args.hold_frames
                ):
                    p1 = tuple(map(int, a.point))
                    p2 = tuple(map(int, b.point))
                    draw_alpha_line(rendered, p1, p2)

            # 再画圈和标签
            for target in targets:
                if target.point is not None and target.missed <= args.hold_frames:
                    draw_target(
                        rendered,
                        target,
                        detected_now=(target.track_id in detected_ids),
                    )

            writer.write(rendered)

            if args.show:
                preview = rendered
                # 过大的视频缩小预览，不影响输出分辨率
                max_preview_w = 1400
                if preview.shape[1] > max_preview_w:
                    scale = max_preview_w / preview.shape[1]
                    preview = cv2.resize(
                        preview,
                        None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow("Football Player Tracker - q/ESC to stop", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("[停止] 用户提前结束。")
                    break

            if total_frames > 0:
                done = frame_index - start_frame + 1
                total = max(1, total_frames - start_frame)
                progress = int(done * 100 / total)
                if progress // 5 != last_progress // 5:
                    print(f"[进度] {min(progress, 100)}%")
                    last_progress = progress

            ok, next_frame = cap.read()
            if not ok:
                break

            frame_index += 1
            current_frame = next_frame
            current_result = model.track(current_frame, **track_kwargs)[0]

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # 尝试恢复原音频
    muxed = mux_audio(
        video_only=video_only,
        source_video=input_path,
        output_path=output_path,
        start_seconds=args.start,
    )

    if muxed:
        try:
            video_only.unlink(missing_ok=True)
        except Exception:
            pass
        print(f"\n[完成] 已输出视频（含原音频，如源视频有音频）:\n{output_path}")
    else:
        # 没有 ffmpeg 或 mux 失败：直接把无音频视频作为最终输出
        if output_path.exists():
            output_path.unlink()
        video_only.replace(output_path)
        print(
            f"\n[完成] 已输出视频:\n{output_path}\n"
            "注意：系统没有可用的 ffmpeg，或音频合并失败，因此当前输出可能没有音频。"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
