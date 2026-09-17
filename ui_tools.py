from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from tracker_core import (
    PALETTE,
    Box,
    RenderStyle,
    TargetState,
    extract_jersey_signature,
    foot_point,
    match_roi_to_track,
    roi_to_box,
)


def read_frame(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
    ok, frame = cap.read()
    return frame if ok else None


def pick_start_frame(
    cap: cv2.VideoCapture, fps: float, total_frames: int, initial_frame: int
) -> Tuple[int, np.ndarray]:
    """Interactive frame picker: trackbar, play/pause, single-frame and 1-second jumps."""
    if total_frames <= 0:
        frame = read_frame(cap, initial_frame)
        if frame is None:
            raise RuntimeError("无法读取起始帧")
        return initial_frame, frame

    win = "Select start frame"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 760)
    state = {"frame": int(np.clip(initial_frame, 0, total_frames - 1)), "dirty": True}
    current: Optional[np.ndarray] = None
    playing = False
    last_tick = time.monotonic()

    def on_trackbar(pos: int) -> None:
        state["frame"] = int(np.clip(pos, 0, total_frames - 1))
        state["dirty"] = True

    cv2.createTrackbar("Frame", win, state["frame"], max(1, total_frames - 1), on_trackbar)
    left_codes, right_codes = {2424832, 65361, 81}, {2555904, 65363, 83}

    while True:
        if state["dirty"] or current is None:
            current = read_frame(cap, state["frame"])
            state["dirty"] = False
            if current is None:
                raise RuntimeError("选择起始帧时读取失败")
        display = current.copy()
        cv2.rectangle(display, (12, 12), (min(display.shape[1] - 12, 1050), 82), (18, 18, 18), -1)
        cv2.putText(
            display,
            f"Frame {state['frame']}/{total_frames - 1}   Time {state['frame']/fps:.2f}s",
            (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "SPACE Play/Pause | Left/Right 1 frame | [ / ] 1 sec | ENTER confirm | Q/ESC cancel",
            (28, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA,
        )
        cv2.imshow(win, display)
        key = cv2.waitKeyEx(15)
        now = time.monotonic()
        if playing and now - last_tick >= max(0.001, 1.0 / fps):
            if state["frame"] < total_frames - 1:
                state["frame"] += 1
                cv2.setTrackbarPos("Frame", win, state["frame"])
                state["dirty"] = True
            else:
                playing = False
            last_tick = now
        if key < 0:
            continue
        if key in (13, 10):
            cv2.destroyWindow(win)
            cap.set(cv2.CAP_PROP_POS_FRAMES, state["frame"] + 1)
            return state["frame"], current
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(win)
            raise KeyboardInterrupt("用户取消起始帧选择")
        if key == 32:
            playing = not playing
            last_tick = now
            continue
        new_frame = state["frame"]
        if key in left_codes or key in (ord("a"), ord("A")):
            new_frame -= 1
            playing = False
        elif key in right_codes or key in (ord("d"), ord("D")):
            new_frame += 1
            playing = False
        elif key == ord("["):
            new_frame -= max(1, int(round(fps)))
            playing = False
        elif key == ord("]"):
            new_frame += max(1, int(round(fps)))
            playing = False
        new_frame = int(np.clip(new_frame, 0, total_frames - 1))
        if new_frame != state["frame"]:
            state["frame"] = new_frame
            cv2.setTrackbarPos("Frame", win, new_frame)
            state["dirty"] = True


def select_targets(frame: np.ndarray, tracks: Dict[int, Box], count: int) -> List[TargetState]:
    targets: List[TargetState] = []
    used: set[int] = set()
    for i in range(count):
        name = chr(ord("A") + i)
        while True:
            title = f"Select Player {name} - drag box, ENTER/SPACE confirm"
            roi = cv2.selectROI(title, frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(title)
            if roi[2] <= 1 or roi[3] <= 1:
                continue
            tid = match_roi_to_track(roi_to_box(roi), tracks, used)
            if tid is None:
                print(f"[选择] Player {name} 未匹配到检测框，请稍微扩大框选范围。")
                continue
            used.add(tid)
            box = tracks[tid]
            target = TargetState(name, tid)
            target.initialize(foot_point(box), box, extract_jersey_signature(frame, box))
            targets.append(target)
            print(f"[绑定] Player {name} -> Track ID {tid}")
            break
    return targets


def choose_style_ui(initial: RenderStyle) -> RenderStyle:
    """Five-color clickable palette + 1..10 ring/line thickness sliders."""
    names = list(PALETTE)
    state = {
        "ring": initial.ring_color_name,
        "line": initial.line_color_name,
        "ring_t": initial.ring_thickness,
        "line_t": initial.line_thickness,
    }
    original = RenderStyle(**initial.__dict__)
    win, w, h = "Style selector", 900, 430
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, w, h)
    cv2.createTrackbar("Ring thickness", win, int(state["ring_t"]), 10, lambda _x: None)
    cv2.createTrackbar("Line thickness", win, int(state["line_t"]), 10, lambda _x: None)
    swatches = {}
    for row, y in (("ring", 65), ("line", 155)):
        for i, name in enumerate(names):
            x1 = 150 + i * 112
            swatches[(row, name)] = (x1, y, x1 + 100, y + 54)

    def mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for (row, name), (x1, y1, x2, y2) in swatches.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                state[row] = name
                break

    cv2.setMouseCallback(win, mouse)
    while True:
        rt = max(1, cv2.getTrackbarPos("Ring thickness", win))
        lt = max(1, cv2.getTrackbarPos("Line thickness", win))
        if cv2.getTrackbarPos("Ring thickness", win) == 0:
            cv2.setTrackbarPos("Ring thickness", win, 1)
        if cv2.getTrackbarPos("Line thickness", win) == 0:
            cv2.setTrackbarPos("Line thickness", win, 1)
        state["ring_t"], state["line_t"] = rt, lt
        canvas = np.full((h, w, 3), 28, np.uint8)
        cv2.putText(canvas, "Ring color", (30, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
        cv2.putText(canvas, "Line color", (30, 191), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2)
        for (row, name), (x1, y1, x2, y2) in swatches.items():
            cv2.rectangle(canvas, (x1, y1), (x2, y2), PALETTE[name], -1)
            selected = state[row] == name
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255) if selected else (90, 90, 90), 4 if selected else 1)
            cv2.putText(canvas, name, (x1 + 5, y2 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
        p1, p2 = (295, 330), (605, 330)
        cv2.line(canvas, p1, p2, PALETTE[state["line"]], int(state["line_t"]), cv2.LINE_AA)
        cv2.ellipse(canvas, p1, (36, 12), 0, 0, 360, PALETTE[state["ring"]], int(state["ring_t"]), cv2.LINE_AA)
        cv2.ellipse(canvas, p2, (36, 12), 0, 0, 360, PALETTE[state["ring"]], int(state["ring_t"]), cv2.LINE_AA)
        cv2.putText(canvas, "ENTER / SPACE confirm    Q / ESC cancel", (220, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1)
        cv2.imshow(win, canvas)
        key = cv2.waitKeyEx(20)
        if key in (13, 10, 32):
            cv2.destroyWindow(win)
            return RenderStyle(str(state["ring"]), int(state["ring_t"]), str(state["line"]), int(state["line_t"]))
        if key in (27, ord("q"), ord("Q")):
            cv2.destroyWindow(win)
            return original
