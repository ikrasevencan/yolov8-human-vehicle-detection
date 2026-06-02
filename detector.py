"""
AI-Based Human and Vehicle Detection Using YOLOv8
==================================================
Detects persons and vehicles (car, truck, bus, motorcycle, bicycle)
from images, video files, or webcam streams.
"""

import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

# ── Target class IDs in COCO dataset ──────────────────────────────────────────
HUMAN_CLASSES   = {0: "Person"}
VEHICLE_CLASSES = {
    1: "Bicycle", 2: "Car", 3: "Motorcycle",
    5: "Bus",     7: "Truck"
}
TARGET_CLASSES  = {**HUMAN_CLASSES, **VEHICLE_CLASSES}

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
COLORS = {
    "Person":     (0,   200, 255),   # amber
    "Bicycle":    (0,   255, 128),   # green-mint
    "Car":        (255, 80,  80 ),   # blue
    "Motorcycle": (200, 0,   255),   # purple
    "Bus":        (0,   128, 255),   # orange
    "Truck":      (50,  200, 200),   # teal
}

# ── Detection result ──────────────────────────────────────────────────────────
@dataclass
class Detection:
    label:      str
    confidence: float
    box:        tuple          # (x1, y1, x2, y2)
    category:   str            # "human" | "vehicle"


# ── Stats tracker ─────────────────────────────────────────────────────────────
@dataclass
class FrameStats:
    total:    int                      = 0
    humans:   int                      = 0
    vehicles: int                      = 0
    counts:   dict = field(default_factory=lambda: defaultdict(int))
    fps:      float                    = 0.0


# ══════════════════════════════════════════════════════════════════════════════
class YOLOv8Detector:
    """Wraps Ultralytics YOLOv8 for human & vehicle detection."""

    def __init__(
        self,
        model_path:  str   = "yolov8n.pt",
        confidence:  float = 0.40,
        iou:         float = 0.45,
        device:      str   = "cpu",
    ):
        from ultralytics import YOLO
        print(f"[INFO] Loading model: {model_path}  (device={device})")
        self.model      = YOLO(model_path)
        self.confidence = confidence
        self.iou        = iou
        self.device     = device
        self.target_ids = list(TARGET_CLASSES.keys())
        print("[INFO] Model ready.")

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> tuple[list[Detection], FrameStats]:
        """Run inference on one BGR frame; return detections + stats."""
        # Kareyi liste içinde göndermek hata riskini sıfırlar
        results = self.model(
            source=[frame], 
            conf=self.confidence, 
            iou=self.iou, 
            classes=self.target_ids, 
            device=self.device, 
            verbose=False
        )[0]

        detections: list[Detection] = []
        stats = FrameStats()

        if results.boxes is None:
            return detections, stats

        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in TARGET_CLASSES:
                continue

            label      = TARGET_CLASSES[cls_id]
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            category = "human" if cls_id in HUMAN_CLASSES else "vehicle"

            detections.append(Detection(label, confidence, (x1, y1, x2, y2), category))
            stats.counts[label] += 1
            if category == "human":
                stats.humans += 1
            else:
                stats.vehicles += 1

        stats.total = len(detections)
        return detections, stats


# ══════════════════════════════════════════════════════════════════════════════
class Visualizer:
    """Draws detections and HUD onto frames."""

    BOX_THICKNESS   = 2
    FONT            = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE      = 0.55
    FONT_THICKNESS  = 1
    HUD_ALPHA       = 0.55

    # ------------------------------------------------------------------
    @classmethod
    def draw(cls, frame: np.ndarray, detections: list[Detection], stats: FrameStats) -> np.ndarray:
        out = frame.copy()
        for d in detections:
            cls.draw_box(out, d)
        cls.draw_hud(out, stats)
        return out

    # ------------------------------------------------------------------
    @classmethod
    def draw_box(cls, img: np.ndarray, d: Detection) -> None:
        x1, y1, x2, y2 = d.box
        color = COLORS.get(d.label, (255, 255, 255))

        # bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, cls.BOX_THICKNESS)

        # label background
        text   = f"{d.label}  {d.confidence:.0%}"
        (tw, th), bl = cv2.getTextSize(text, cls.FONT, cls.FONT_SCALE, cls.FONT_THICKNESS)
        pad    = 3
        label_y = max(y1 - th - 2 * pad, 0)
        cv2.rectangle(img, (x1, label_y), (x1 + tw + 2 * pad, label_y + th + 2 * pad), color, -1)

        # label text
        text_color = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(img, text, (x1 + pad, label_y + th + pad),
                    cls.FONT, cls.FONT_SCALE, text_color, cls.FONT_THICKNESS, cv2.LINE_AA)

        # corner accents
        accent = 14
        lw     = cls.BOX_THICKNESS + 1
        for (sx, sy, dx, dy) in [
            (x1, y1,  1,  1), (x2, y1, -1,  1),
            (x1, y2,  1, -1), (x2, y2, -1, -1),
        ]:
            cv2.line(img, (sx, sy), (sx + dx * accent, sy), color, lw)
            cv2.line(img, (sx, sy), (sx, sy + dy * accent), color, lw)

    # ------------------------------------------------------------------
    @classmethod
    def draw_hud(cls, img: np.ndarray, stats: FrameStats) -> None:
        h, w = img.shape[:2]
        lines = [
            f"FPS      : {stats.fps:5.1f}",
            f"Persons  : {stats.humans}",
            f"Vehicles : {stats.vehicles}",
            f"Total    : {stats.total}",
            "─────────────────",
        ]
        for lbl, cnt in sorted(stats.counts.items()):
            lines.append(f"  {lbl:<12}: {cnt}")

        pad    = 10
        lh     = 20
        box_h  = pad * 2 + lh * len(lines)
        box_w  = 200
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (box_w, box_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, cls.HUD_ALPHA, img, 1 - cls.HUD_ALPHA, 0, img)

        for i, line in enumerate(lines):
            y = pad + (i + 1) * lh
            color = (0, 220, 255) if i < 4 else (160, 160, 160)
            cv2.putText(img, line, (pad, y), cls.FONT, 0.46, color, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
def process_image(detector: YOLOv8Detector, path: str, output_dir: str) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {path}")
        return

    t0 = time.perf_counter()
    detections, stats = detector.detect(frame)
    stats.fps = 1.0 / (time.perf_counter() - t0 + 1e-9)

    out = Visualizer.draw(frame, detections, stats)
    out_path = Path(output_dir) / ("detected_" + Path(path).name)
    cv2.imwrite(str(out_path), out)
    print(f"[DONE] {path}  →  {out_path}")
    print(f"       Persons: {stats.humans}  |  Vehicles: {stats.vehicles}  |  Inference: {1000/stats.fps:.1f} ms")


# ══════════════════════════════════════════════════════════════════════════════
def process_video(
    detector:   YOLOv8Detector,
    source,                        # path string or int (webcam index)
    output_dir: str,
    save:       bool = True,
) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save and isinstance(source, str):
        out_path = Path(output_dir) / ("detected_" + Path(source).name)
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        writer   = cv2.VideoWriter(str(out_path), fourcc, fps_in, (W, H))
        print(f"[INFO] Saving output → {out_path}")

    frame_times: list[float] = []
    print("[INFO] Press  Q  to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # GÜVENLİK FİLTRESİ: Boş, tanımsız veya siyah veri gelirse pas geç
        if frame is None or frame.size == 0 or np.sum(frame) == 0:
            continue

        try:
            t0 = time.perf_counter()
            detections, stats = detector.detect(frame)
            elapsed = time.perf_counter() - t0
            frame_times.append(elapsed)
            if len(frame_times) > 30:
                frame_times.pop(0)
            stats.fps = 1.0 / (sum(frame_times) / len(frame_times))

            out = Visualizer.draw(frame, detections, stats)

            if writer:
                writer.write(out)

            cv2.imshow("YOLOv8 — Human & Vehicle Detection  (Q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print(f"[WARN] Kare işlenirken hata oluştu, atlanıyor: {e}")
            continue

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI-Based Human & Vehicle Detection — YOLOv8",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--source",     default="0",           help="Image/video path or webcam index (default: 0)")
    p.add_argument("--model",      default="yolov8n.pt",  help="YOLOv8 model weights  (default: yolov8n.pt)")
    p.add_argument("--conf",       default=0.40, type=float, help="Confidence threshold (default: 0.40)")
    p.add_argument("--iou",        default=0.45, type=float, help="IoU threshold for NMS (default: 0.45)")
    p.add_argument("--device",     default="cpu",          help="Inference device: cpu | cuda | mps")
    p.add_argument("--output-dir", default="output",       help="Output folder for saved results")
    p.add_argument("--no-save",    action="store_true",    help="Do not save output video")
    return p.parse_args()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    detector = YOLOv8Detector(
        model_path = args.model,
        confidence = args.conf,
        iou        = args.iou,
        device     = args.device,
    )

    # Determine source type
    src = args.source
    if src.isdigit():
        src = int(src)                                  # webcam
        process_video(detector, src, args.output_dir, save=False)
    elif src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        process_image(detector, src, args.output_dir)   # image
    else:
        process_video(detector, src, args.output_dir, not args.no_save)  #