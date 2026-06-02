"""
zone_counter.py — Count persons and vehicles crossing a virtual line
or entering defined polygon zones in a video.

Usage:
    python zone_counter.py --source video.mp4 --mode line
    python zone_counter.py --source video.mp4 --mode zone
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from detector import YOLOv8Detector, Visualizer, COLORS, TARGET_CLASSES, HUMAN_CLASSES


# ── Helpers ───────────────────────────────────────────────────────────────────

def box_centroid(box: tuple) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return (x1 + x2) // 2, (y1 + y2) // 2


def point_side(pt, line_start, line_end) -> int:
    """Returns +1 or -1 based on which side of a line a point lies."""
    x, y     = pt
    x1, y1   = line_start
    x2, y2   = line_end
    cross    = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
    return 1 if cross >= 0 else -1


def point_in_polygon(pt, polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, pt, False) >= 0


# ── Line counter ──────────────────────────────────────────────────────────────

class LineCrossCounter:
    """Counts objects crossing a horizontal/vertical counting line."""

    def __init__(self, line_y_ratio: float = 0.5):
        self.line_y_ratio = line_y_ratio  # fraction of frame height
        self._prev_side: dict[int, int]   = {}
        self.counts: dict[str, int]       = {}
        self._next_id                     = 0
        self._track_map: dict             = {}  # centroid → pseudo-id

    def _assign_id(self, centroid, threshold=40) -> int:
        for cid, prev in self._track_map.items():
            if abs(prev[0] - centroid[0]) < threshold and abs(prev[1] - centroid[1]) < threshold:
                self._track_map[cid] = centroid
                return cid
        nid = self._next_id
        self._next_id += 1
        self._track_map[nid] = centroid
        return nid

    def update(self, detections, frame_h: int, line_start, line_end):
        crossed = []
        for d in detections:
            c  = box_centroid(d.box)
            oid = self._assign_id(c)
            side = point_side(c, line_start, line_end)
            if oid in self._prev_side and self._prev_side[oid] != side:
                self.counts[d.label] = self.counts.get(d.label, 0) + 1
                crossed.append(d.label)
            self._prev_side[oid] = side
        return crossed

    def draw(self, img, line_start, line_end):
        cv2.line(img, line_start, line_end, (0, 255, 255), 2)
        cv2.putText(img, "COUNT LINE", (line_start[0] + 5, line_start[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y = 30
        for label, cnt in sorted(self.counts.items()):
            color = COLORS.get(label, (200, 200, 200))
            cv2.putText(img, f"{label}: {cnt}", (img.shape[1] - 180, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 24


# ── Zone counter ──────────────────────────────────────────────────────────────

class ZoneCounter:
    """Counts objects present inside named polygon zones."""

    def __init__(self, zones: dict[str, np.ndarray]):
        self.zones  = zones   # {zone_name: np.array of points}
        self.counts: dict[str, dict[str, int]] = {z: {} for z in zones}

    def update(self, detections):
        for zone_name in self.zones:
            self.counts[zone_name] = {}
        for d in detections:
            c = box_centroid(d.box)
            for zone_name, poly in self.zones.items():
                if point_in_polygon(c, poly):
                    self.counts[zone_name][d.label] = \
                        self.counts[zone_name].get(d.label, 0) + 1

    def draw(self, img):
        zone_colors = [(255, 100, 0), (0, 200, 100), (200, 0, 200)]
        for i, (zone_name, poly) in enumerate(self.zones.items()):
            zc = zone_colors[i % len(zone_colors)]
            overlay = img.copy()
            cv2.fillPoly(overlay, [poly], zc)
            cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
            cv2.polylines(img, [poly], True, zc, 2)

            # label
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(img, zone_name, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, zc, 2)

            # counts inside zone
            cnt_text = "  ".join(
                f"{lbl}:{n}" for lbl, n in self.counts[zone_name].items()
            ) or "empty"
            cv2.putText(img, cnt_text, (cx - 50, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


# ── Runner ────────────────────────────────────────────────────────────────────

def run(args):
    cap = cv2.VideoCapture(args.source if not args.source.isdigit() else int(args.source))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {args.source}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = YOLOv8Detector(model_path=args.model, confidence=args.conf, device=args.device)

    # ── counting primitives ───────────────────────────────────────────
    if args.mode == "line":
        line_start = (0,       int(H * 0.5))
        line_end   = (W,       int(H * 0.5))
        counter    = LineCrossCounter()
    else:
        # Three example zones (adapt to your scene)
        zones = {
            "Zone A": np.array([[0,     0],     [W//2, 0],     [W//2, H//2], [0,     H//2]], np.int32),
            "Zone B": np.array([[W//2,  0],     [W,    0],     [W,    H//2], [W//2,  H//2]], np.int32),
            "Zone C": np.array([[0,     H//2],  [W,    H//2],  [W,    H],    [0,     H   ]], np.int32),
        }
        counter = ZoneCounter(zones)

    print(f"[INFO] Mode: {args.mode}  |  Press Q to quit")
    frame_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        detections, stats = detector.detect(frame)
        frame_times.append(time.perf_counter() - t0)
        if len(frame_times) > 30:
            frame_times.pop(0)
        stats.fps = 1.0 / (sum(frame_times) / len(frame_times))

        # Draw standard boxes
        out = Visualizer.draw(frame, detections, stats)

        # Overlay counting layer
        if args.mode == "line":
            counter.update(detections, H, line_start, line_end)
            counter.draw(out, line_start, line_end)
        else:
            counter.update(detections)
            counter.draw(out)

        cv2.imshow(f"YOLOv8 Zone Counter [{args.mode}]  (Q quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8 Zone / Line Counter")
    p.add_argument("--source", required=True,        help="Video file or webcam index")
    p.add_argument("--model",  default="yolov8n.pt")
    p.add_argument("--conf",   default=0.40, type=float)
    p.add_argument("--device", default="cpu")
    p.add_argument("--mode",   default="line", choices=["line", "zone"],
                   help="'line' — crossing counter | 'zone' — polygon presence counter")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())