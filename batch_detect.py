"""
batch_detect.py — Run YOLOv8 detection on an entire folder and produce a
CSV summary + per-image annotated results.

Usage:
    python batch_detect.py --input ./images --output ./output
"""

import cv2
import csv
import time
import argparse
from pathlib import Path

# Re-use core detector from detector.py
from detector import YOLOv8Detector, Visualizer, Detection, FrameStats

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


# ── helpers ───────────────────────────────────────────────────────────────────

def detect_image(detector: YOLOv8Detector, img_path: Path):
    frame = cv2.imread(str(img_path))
    if frame is None:
        return None, None, None
    t0 = time.perf_counter()
    detections, stats = detector.detect(frame)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return frame, detections, stats, elapsed_ms


def save_annotated(frame, detections, stats, out_path: Path):
    annotated = Visualizer.draw(frame, detections, stats)
    cv2.imwrite(str(out_path), annotated)


# ── main ──────────────────────────────────────────────────────────────────────

def run_batch(args):
    in_dir  = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = [p for p in sorted(in_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"[WARN] No images found in {in_dir}")
        return

    detector = YOLOv8Detector(
        model_path = args.model,
        confidence = args.conf,
        iou        = args.iou,
        device     = args.device,
    )

    csv_path = out_dir / "detection_summary.csv"
    fieldnames = ["file", "persons", "vehicles", "total",
                  "car", "truck", "bus", "motorcycle", "bicycle",
                  "inference_ms"]

    total_persons  = 0
    total_vehicles = 0

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, img_path in enumerate(images, 1):
            result = detect_image(detector, img_path)
            if result[0] is None:
                print(f"  [SKIP] {img_path.name}")
                continue
            frame, detections, stats, elapsed_ms = result

            out_path = out_dir / ("det_" + img_path.name)
            save_annotated(frame, detections, stats, out_path)

            row = {
                "file":         img_path.name,
                "persons":      stats.humans,
                "vehicles":     stats.vehicles,
                "total":        stats.total,
                "car":          stats.counts.get("Car",        0),
                "truck":        stats.counts.get("Truck",      0),
                "bus":          stats.counts.get("Bus",        0),
                "motorcycle":   stats.counts.get("Motorcycle", 0),
                "bicycle":      stats.counts.get("Bicycle",    0),
                "inference_ms": f"{elapsed_ms:.1f}",
            }
            writer.writerow(row)
            total_persons  += stats.humans
            total_vehicles += stats.vehicles

            print(f"  [{i:>3}/{len(images)}] {img_path.name:<35} "
                  f"persons={stats.humans}  vehicles={stats.vehicles}  "
                  f"({elapsed_ms:.0f} ms)")

    print(f"\n[DONE]  Annotated images → {out_dir}")
    print(f"        CSV summary      → {csv_path}")
    print(f"        Total persons    : {total_persons}")
    print(f"        Total vehicles   : {total_vehicles}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Batch Human & Vehicle Detection — YOLOv8")
    p.add_argument("--input",   required=True,          help="Folder containing input images")
    p.add_argument("--output",  default="output",       help="Folder for annotated images + CSV")
    p.add_argument("--model",   default="yolov8n.pt",   help="YOLOv8 weights")
    p.add_argument("--conf",    default=0.40, type=float)
    p.add_argument("--iou",     default=0.45, type=float)
    p.add_argument("--device",  default="cpu")
    return p.parse_args()


if __name__ == "__main__":
    run_batch(parse_args())