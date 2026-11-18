"""
YOLOv8 Real-Time Fire Detection System with Custom HUD Overlay
Author: Bilge
"""

import os
import sys
import time
import math
import argparse
import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Fire Detection with HUD Telemetry & Bounding Box Overlay"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input video path or camera index (e.g. 0 for webcam, or path/to/video.mp4)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "weights", "best_final.pt"),
        help="Path to trained YOLO model (.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.20,
        help="Confidence threshold for fire detection (default: 0.20)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional path to save processed output video (e.g. outputs/result.mp4)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image resolution (default: 640)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run in headless mode without displaying output window",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: 'cpu', '0' (CUDA), or auto-detect if empty",
    )
    return parser.parse_args()


def draw_fire_box(frame, x1, y1, x2, y2, confidence_pct):
    """Draws a highlighted bounding box with corner markers and percentage tag."""
    # Outer & inner bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 140, 255), 1)

    # Stylized corner accent markers
    line_len = min(25, max(8, (x2 - x1) // 3), max(8, (y2 - y1) // 3))
    accent_color = (0, 255, 255)
    thick = 3

    # Top-Left
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), accent_color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), accent_color, thick)
    # Top-Right
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), accent_color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), accent_color, thick)
    # Bottom-Left
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), accent_color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), accent_color, thick)
    # Bottom-Right
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), accent_color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - line_len), accent_color, thick)

    # Percentage label tag
    label = f"FIRE: %{confidence_pct}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.70
    font_thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    label_y1 = max(0, y1 - th - 12)
    label_y2 = y1
    label_x1 = x1
    label_x2 = x1 + tw + 16

    # Tag background
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 180), -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 255), 1)
    cv2.putText(
        frame,
        label,
        (label_x1 + 8, label_y2 - 6),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_hud(frame, status_text, status_color, frame_info, fps_info):
    """Draws a semi-transparent HUD telemetry banner at the top-left."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (450, 110), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (20, 20), (450, 110), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, status_text, (35, 55), font, 0.75, status_color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{frame_info} | {fps_info}",
        (35, 85),
        font,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    # Determine compute device
    if not args.device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)

    # Check model path
    if not os.path.exists(args.model):
        print(f"[!] Error: Model file not found at: {args.model}")
        print("    Please check the path or ensure weights/best_final.pt exists.")
        sys.exit(1)

    print(f"[*] Loading YOLO model: {args.model}")
    print(f"[*] Running on device: {device.upper()}")
    model = YOLO(args.model)

    # Open video capture source
    source = args.source
    is_cam = source.isdigit()
    cap_src = int(source) if is_cam else source

    cap = cv2.VideoCapture(cap_src)
    if not cap.isOpened():
        print(f"[!] Error: Unable to open input source: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_cam else 0

    # Video Writer (if output specified)
    writer = None
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (1920, 1080))
        print(f"[*] Saving processed output to: {args.output}")

    print("[*] Fire detection started... (Press 'q' in video window to exit)")

    frame_idx = 0
    t_start = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            t_frame_start = time.time()

            # Resize to standard 1080p for uniform display & output
            frame_resized = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

            # YOLO inference (Class 0: Fire)
            results = model.predict(
                frame_resized,
                imgsz=args.imgsz,
                conf=args.conf,
                classes=[0],
                device=device,
                verbose=False,
            )[0]

            boxes = results.boxes
            fire_count = len(boxes)

            # Draw fire bounding boxes
            for b in boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf_pct = int(b.conf[0] * 100)
                draw_fire_box(frame_resized, x1, y1, x2, y2, conf_pct)

            # Telemetry status
            if fire_count > 0:
                status_text = f"DURUM: YANGIN ALGILANDI ({fire_count})"
                status_color = (0, 0, 255)
            else:
                status_text = "DURUM: NORMAL"
                status_color = (0, 255, 0)

            frame_info = f"Kare: {frame_idx}/{total_frames}" if total_frames > 0 else f"Kare: {frame_idx}"
            fps_info = f"FPS: {fps_display:.1f}"

            draw_hud(frame_resized, status_text, status_color, frame_info, fps_info)

            # Save frame to output video if writer is active
            if writer is not None:
                writer.write(frame_resized)

            # Calculate FPS
            t_frame_end = time.time()
            dt = t_frame_end - t_frame_start
            if dt > 0:
                fps_display = 0.9 * fps_display + 0.1 * (1.0 / dt)

            # Live preview window
            if not args.no_show:
                display_frame = cv2.resize(frame_resized, (1280, 720))
                cv2.imshow("YOLO Fire Detection & Telemetry HUD", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[*] Interrupted by user.")
                    break

            if total_frames > 0 and frame_idx % 50 == 0:
                pct = (frame_idx / total_frames) * 100
                print(f"  -> Progress: {pct:5.1f}% ({frame_idx}/{total_frames}) | Current Speed: {fps_display:.1f} FPS")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\n[✔] Finished! Processed {frame_idx} frames in {elapsed:.1f}s (~{frame_idx/max(0.1, elapsed):.1f} avg FPS).")
    if args.output:
        print(f"[✔] Saved output file: {args.output}")


if __name__ == "__main__":
    main()
