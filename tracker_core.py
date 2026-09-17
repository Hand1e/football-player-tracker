from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

Point = Tuple[float, float]
Box = Tuple[float, float, float, float]

# OpenCV uses BGR.
PALETTE = {
    "yellow": (0, 220, 255),
    "red": (40, 50, 235),
    "blue": (235, 120, 35),
    "purple": (190, 70, 205),
    "orange": (0, 135, 255),
}
COLOR_ALIASES = {
    "yellow": "yellow", "黄": "yellow", "黄色": "yellow",
    "red": "red", "红": "red", "红色": "red",
    "blue": "blue", "蓝": "blue", "蓝色": "blue",
    "purple": "purple", "紫": "purple", "紫色": "purple",
    "orange": "orange", "橙": "orange", "橙色": "orange",
}


@dataclass
class JerseySignature:
    hist: np.ndarray
    median_v: float
    bright_ratio: float
    dark_ratio: float


@dataclass
class RenderStyle:
    ring_color_name: str = "yellow"
    ring_thickness: int = 4
    line_color_name: str = "yellow"
    line_thickness: int = 4

    @property
    def ring_color(self) -> Tuple[int, int, int]:
        return PALETTE[self.ring_color_name]

    @property
    def line_color(self) -> Tuple[int, int, int]:
        return PALETTE[self.line_color_name]


@dataclass
class TrackObservation:
    track_id: int
    box: Box
    point: np.ndarray
    signature: Optional[JerseySignature] = None


@dataclass
class TargetState:
    name: str
    track_id: int
    point: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    box_w: float = 30.0
    box_h: float = 80.0
    last_box: Optional[Box] = None
    missed: int = 0
    ring_rx: int = 20
    ring_ry: int = 7
    appearance: Optional[JerseySignature] = None
    uncertain: bool = False
    needs_reacquire: bool = False
    confusion_lock: bool = False
    separation_frames: int = 0

    def initialize(self, point: Point, box: Box, signature: Optional[JerseySignature]) -> None:
        self.point = np.array(point, np.float32)
        self.velocity = np.zeros(2, np.float32)
        self.box_w, self.box_h = box_size(box)
        self.last_box = box
        self.ring_rx = int(np.clip(self.box_w * 0.55, 13, 46))
        self.ring_ry = max(6, int(self.ring_rx * 0.34))
        self.appearance = signature
        self.missed = 0
        self.uncertain = self.needs_reacquire = self.confusion_lock = False
        self.separation_frames = 0

    def update(
        self,
        point: Point,
        box: Box,
        alpha: float,
        signature: Optional[JerseySignature] = None,
        update_appearance: bool = True,
    ) -> None:
        p = np.array(point, np.float32)
        if self.point is None:
            self.point = p
            self.velocity = np.zeros(2, np.float32)
        else:
            old = self.point.copy()
            smoothed = alpha * p + (1.0 - alpha) * old
            delta = smoothed - old
            self.velocity = delta if self.velocity is None else 0.35 * delta + 0.65 * self.velocity
            self.point = smoothed
        w, h = box_size(box)
        self.box_w = 0.15 * w + 0.85 * self.box_w
        self.box_h = 0.15 * h + 0.85 * self.box_h
        self.last_box = box
        self.missed = 0
        self.uncertain = False
        if update_appearance and signature is not None:
            self.appearance = blend_signature(self.appearance, signature, 0.08)

    def predict(self, frame_w: int, frame_h: int) -> None:
        self.missed += 1
        self.uncertain = True
        if self.point is None:
            return
        if self.velocity is not None:
            self.velocity *= 0.90
            self.point += self.velocity
        self.point[0] = np.clip(self.point[0], 0, frame_w - 1)
        self.point[1] = np.clip(self.point[1], 0, frame_h - 1)

    def predicted_point(self) -> Optional[np.ndarray]:
        if self.point is None:
            return None
        return self.point.copy() if self.velocity is None else self.point + self.velocity


def parse_named_color(value: str) -> str:
    key = value.strip().lower()
    if key not in COLOR_ALIASES:
        raise argparse.ArgumentTypeError(
            f"不支持颜色 {value!r}；可用 yellow/red/blue/purple/orange 或 黄/红/蓝/紫/橙"
        )
    return COLOR_ALIASES[key]


def box_size(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def roi_to_box(roi: Sequence[float]) -> Box:
    x, y, w, h = map(float, roi)
    return x, y, x + w, y + h


def foot_point(box: Box) -> Point:
    x1, _, x2, y2 = box
    return (x1 + x2) / 2.0, y2


def extract_tracks(result) -> Dict[int, Box]:
    if result.boxes is None or result.boxes.id is None:
        return {}
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    ids = result.boxes.id.detach().cpu().numpy().astype(int)
    return {int(tid): tuple(map(float, box)) for tid, box in zip(ids, boxes)}


def extract_jersey_signature(frame: np.ndarray, box: Box) -> Optional[JerseySignature]:
    """3-D HSV torso histogram; removes high-saturation grass-green pixels."""
    ih, iw = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(2.0, x2 - x1), max(2.0, y2 - y1)
    sx1 = int(np.clip(x1 + 0.15 * bw, 0, iw - 1))
    sx2 = int(np.clip(x2 - 0.15 * bw, sx1 + 1, iw))
    sy1 = int(np.clip(y1 + 0.08 * bh, 0, ih - 1))
    sy2 = int(np.clip(y1 + 0.60 * bh, sy1 + 1, ih))
    crop = frame[sy1:sy2, sx1:sx2]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 3:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    grass = (h >= 30) & (h <= 95) & (s >= 45) & (v >= 35)
    valid = ~grass
    if int(valid.sum()) < 20:
        valid = np.ones_like(h, dtype=bool)
    mask = valid.astype(np.uint8) * 255
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None, norm_type=cv2.NORM_L1).astype(np.float32)
    vals = v[valid].astype(np.float32) / 255.0
    return JerseySignature(
        hist,
        float(np.median(vals)) if vals.size else 0.5,
        float(np.mean(vals > 0.72)) if vals.size else 0.0,
        float(np.mean(vals < 0.32)) if vals.size else 0.0,
    )


def blend_signature(old: Optional[JerseySignature], new: JerseySignature, alpha: float) -> JerseySignature:
    if old is None:
        return new
    hist = cv2.normalize((1 - alpha) * old.hist + alpha * new.hist, None, norm_type=cv2.NORM_L1)
    return JerseySignature(
        hist.astype(np.float32),
        (1 - alpha) * old.median_v + alpha * new.median_v,
        (1 - alpha) * old.bright_ratio + alpha * new.bright_ratio,
        (1 - alpha) * old.dark_ratio + alpha * new.dark_ratio,
    )


def appearance_similarity(a: Optional[JerseySignature], b: Optional[JerseySignature]) -> float:
    if a is None or b is None:
        return 0.50
    corr = np.clip((cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_CORREL) + 1.0) / 2.0, 0.0, 1.0)
    bhatta = cv2.compareHist(a.hist, b.hist, cv2.HISTCMP_BHATTACHARYYA)
    score = 0.65 * corr + 0.35 * np.clip(1.0 - bhatta, 0.0, 1.0)
    mismatch = (
        (a.bright_ratio > 0.48 and b.dark_ratio > 0.45)
        or (b.bright_ratio > 0.48 and a.dark_ratio > 0.45)
    )
    if mismatch:
        return min(float(score), 0.16)
    dv = abs(a.median_v - b.median_v)
    if dv > 0.42:
        score *= 0.45
    elif dv > 0.28:
        score *= 0.72
    if abs(a.bright_ratio - b.bright_ratio) > 0.48:
        score *= 0.60
    return float(np.clip(score, 0.0, 1.0))


def match_roi_to_track(roi: Box, tracks: Dict[int, Box], forbidden: Optional[set[int]] = None) -> Optional[int]:
    forbidden = forbidden or set()
    rx1, ry1, rx2, ry2 = roi
    rcx, rcy = (rx1 + rx2) / 2, (ry1 + ry2) / 2
    rdiag = max(1.0, math.hypot(rx2 - rx1, ry2 - ry1))
    best_id, best_score = None, -1.0
    for tid, box in tracks.items():
        if tid in forbidden:
            continue
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        score = box_iou(roi, box) + (0.65 if rx1 <= cx <= rx2 and ry1 <= cy <= ry2 else 0.0)
        score -= 0.08 * math.hypot(cx - rcx, cy - rcy) / rdiag
        if score > best_score:
            best_id, best_score = tid, score
    return best_id if best_score >= 0.05 else None


def resolve_connect_mode(count: int, requested: Optional[str]) -> str:
    return requested if requested else ("chain" if count == 2 else "polygon")


def connection_pairs(count: int, mode: str) -> List[Tuple[int, int]]:
    if count < 2:
        return []
    if mode == "all":
        return list(combinations(range(count), 2))
    pairs = [(i, i + 1) for i in range(count - 1)]
    if mode == "polygon" and count >= 3:
        pairs.append((count - 1, 0))
    return pairs


def _observations(tracks: Dict[int, Box]) -> Dict[int, TrackObservation]:
    return {tid: TrackObservation(tid, box, np.array(foot_point(box), np.float32)) for tid, box in tracks.items()}


def _signature(frame: np.ndarray, obs: TrackObservation) -> Optional[JerseySignature]:
    if obs.signature is None:
        obs.signature = extract_jersey_signature(frame, obs.box)
    return obs.signature


def _confused(a: TargetState, b: TargetState, obs: Dict[int, TrackObservation], args) -> bool:
    oa, ob = obs.get(a.track_id), obs.get(b.track_id)
    if oa is None or ob is None:
        return False
    if box_iou(oa.box, ob.box) >= args.occlusion_iou:
        return True
    wa, _ = box_size(oa.box)
    wb, _ = box_size(ob.box)
    return np.linalg.norm(oa.point - ob.point) <= args.occlusion_distance * max(10.0, (wa + wb) / 2.0)


def _score(frame: np.ndarray, target: TargetState, obs: TrackObservation, args) -> float:
    pred = target.predicted_point()
    if pred is None:
        return 0.0
    app = appearance_similarity(target.appearance, _signature(frame, obs))
    dist = float(np.linalg.norm(obs.point - pred))
    gate = max(35.0, target.box_h * args.reid_gate)
    pos = math.exp(-((dist / gate) ** 2))
    direction = 0.50
    if target.point is not None and target.velocity is not None:
        v = target.velocity.astype(float)
        d = (obs.point - target.point).astype(float)
        nv, nd = np.linalg.norm(v), np.linalg.norm(d)
        if nv > 0.8 and nd > 0.8:
            direction = (float(np.clip(np.dot(v, d) / (nv * nd), -1, 1)) + 1) / 2
    _, oh = box_size(obs.box)
    size = math.exp(-abs(math.log(max(1e-3, oh / max(1.0, target.box_h)))))
    score = 0.50 * app + 0.28 * pos + 0.12 * direction + 0.10 * size
    if obs.track_id == target.track_id:
        score += 0.025
    return float(np.clip(score, 0, 1))


def _continuity_safe(frame: np.ndarray, target: TargetState, obs: TrackObservation) -> bool:
    pred = target.predicted_point()
    if pred is None:
        return True
    if np.linalg.norm(obs.point - pred) > max(45.0, 1.35 * target.box_h):
        return False
    return appearance_similarity(target.appearance, _signature(frame, obs)) >= 0.20


def _reassign(
    frame: np.ndarray,
    indices: List[int],
    targets: List[TargetState],
    obs: Dict[int, TrackObservation],
    args,
) -> bool:
    if not indices or not obs:
        return False
    occupied = {t.track_id for i, t in enumerate(targets) if i not in indices and not t.uncertain}
    candidates: List[int] = []
    for tid, ob in obs.items():
        if tid in occupied:
            continue
        for i in indices:
            pred = targets[i].predicted_point()
            if pred is not None and np.linalg.norm(ob.point - pred) <= max(45.0, targets[i].box_h * args.reid_gate):
                candidates.append(tid)
                break
    if len(candidates) < len(indices):
        return False
    if len(candidates) > 12:
        candidates = [tid for _, tid in sorted(
            ((max(_score(frame, targets[i], obs[tid], args) for i in indices), tid) for tid in candidates),
            reverse=True,
        )[:12]]
    scores = np.array([[_score(frame, targets[i], obs[tid], args) for tid in candidates] for i in indices], np.float32)
    rows, cols = linear_sum_assignment(1.0 - scores)
    assignment = {int(r): int(c) for r, c in zip(rows, cols)}
    if len(assignment) < len(indices):
        return False
    for r in range(len(indices)):
        c = assignment[r]
        chosen = float(scores[r, c])
        second = max([float(scores[r, j]) for j in range(len(candidates)) if j != c], default=0.0)
        if chosen < args.identity_threshold or chosen - second < args.identity_margin:
            return False
    for r, target_index in enumerate(indices):
        target = targets[target_index]
        ob = obs[candidates[assignment[r]]]
        old_id = target.track_id
        target.track_id = ob.track_id
        target.update(tuple(ob.point), ob.box, max(args.smooth, 0.45), _signature(frame, ob), True)
        target.needs_reacquire = target.confusion_lock = target.uncertain = False
        target.separation_frames = 0
        if old_id != target.track_id:
            print(f"[重关联] Player {target.name}: Track {old_id} -> {target.track_id}")
    return True


def update_targets_with_guard(
    frame: np.ndarray,
    targets: List[TargetState],
    tracks: Dict[int, Box],
    args,
    frame_w: int,
    frame_h: int,
) -> None:
    obs = _observations(tracks)
    if not args.anti_switch:
        for t in targets:
            ob = obs.get(t.track_id)
            if ob is None:
                t.predict(frame_w, frame_h)
            else:
                t.update(tuple(ob.point), ob.box, args.smooth, _signature(frame, ob), True)
        return

    confused: set[int] = set()
    for i, j in combinations(range(len(targets)), 2):
        if _confused(targets[i], targets[j], obs, args):
            confused.update((i, j))
    for i in confused:
        t = targets[i]
        t.confusion_lock = t.needs_reacquire = True
        t.separation_frames = 0

    immediate: List[int] = []
    delayed: List[int] = []
    for i, t in enumerate(targets):
        if i in confused:
            t.predict(frame_w, frame_h)
            continue
        if t.confusion_lock:
            t.separation_frames += 1
            t.predict(frame_w, frame_h)
            if t.separation_frames >= args.separation_frames:
                delayed.append(i)
            continue
        ob = obs.get(t.track_id)
        if ob is None or not _continuity_safe(frame, t, ob):
            t.needs_reacquire = True
            t.predict(frame_w, frame_h)
            immediate.append(i)
            continue
        t.update(tuple(ob.point), ob.box, args.smooth, _signature(frame, ob), True)
        t.needs_reacquire = False

    if delayed:
        _reassign(frame, sorted(set(delayed)), targets, obs, args)
    immediate = [i for i in immediate if not targets[i].confusion_lock]
    if immediate:
        _reassign(frame, sorted(set(immediate)), targets, obs, args)
