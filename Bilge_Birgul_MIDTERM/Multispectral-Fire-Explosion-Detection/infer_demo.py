"""
infer_demo.py
-------------
Eğitilmiş Multispektral YOLOv8s Modeli ile Görsel / Video / Kamera Çıkarım Betiği.

Kullanım:
    # Resim üzerinde test
    python infer_demo.py --source test.jpg

    # Termal veya RGB video üzerinde test
    python infer_demo.py --source video.mp4 --conf 0.30

    # Web kamerası canlı test
    python infer_demo.py --source 0 --show
"""

import argparse
import time
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


def cv2_imwrite_utf8(path, img, quality=90):
    """Windows üzerinde Türkçe / Unicode yollara güvenli görsel yazma."""
    try:
        ext = Path(path).suffix
        is_success, buffer = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if is_success:
            with open(str(path), "wb") as f:
                f.write(buffer)
            return True
    except Exception:
        pass
    return False


def run_inference(source, weights_path, conf_thresh=0.25, imgsz=640, show=False, save_dir="results/demo"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(" MULTİSPEKTRAL YANGIN VE PATLAMA TESPİTİ - ÇIKARIM DEMOSU")
    print("=" * 65)
    print(f"[*] Model Ağırlığı: {weights_path}")
    print(f"[*] Kaynak:         {source}")
    print(f"[*] Güven Eşiği:    {conf_thresh}")

    if not Path(weights_path).exists():
        print(f"[!] HATA: Ağırlık dosyası bulunamadı: {weights_path}")
        return

    model = YOLO(weights_path)

    # 1. Kaynak Web Kamera veya Video mu?
    is_webcam = source.isdigit()
    source_path = int(source) if is_webcam else Path(source)

    is_video = is_webcam or (hasattr(source_path, "suffix") and source_path.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"])

    if is_video:
        cap = cv2.VideoCapture(source_path if not is_webcam else int(source))
        if not cap.isOpened():
            print(f"[!] Video kaynağı açılamadı: {source}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

        out_video_path = save_path / "output_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video_path), fourcc, fps_in, (w, h))

        print(f"[*] Video işleniyor ({w}x{h} @ {fps_in:.1f} FPS)... Çıkmak için 'q' tuşuna basın.")
        frame_count = 0
        total_time = 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            results = model.predict(frame, conf=conf_thresh, imgsz=imgsz, verbose=False)
            dt = time.time() - t0
            total_time += dt
            frame_count += 1

            annotated = results[0].plot()

            # FPS Göstergesi
            current_fps = 1.0 / dt if dt > 0 else 0
            cv2.putText(annotated, f"FPS: {current_fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            writer.write(annotated)

            if show:
                cv2.imshow("Multispectral Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"[+] Video tamamlandı: {out_video_path}")
        print(f"[+] İşlenen Kare: {frame_count}, Ortalama Hız: {avg_fps:.1f} FPS")

    else:
        # Tekil Görsel
        results = model.predict(source, conf=conf_thresh, imgsz=imgsz)
        annotated = results[0].plot()

        out_img_path = save_path / f"detected_{Path(source).name}"
        cv2_imwrite_utf8(out_img_path, annotated)
        print(f"[+] Tespit edilen görsel kaydedildi: {out_img_path}")

        if show:
            cv2.imshow("Multispectral Detection", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multispektral Yangın ve Patlama Tespiti Çıkarım Betiği")
    parser.add_argument("--source", type=str, default="results/samples/FLIR_WhiteHot_frame_338_detected.jpg",
                        help="Görsel dosya yolu, video yolu veya webcam ID ('0')")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="YOLOv8s model ağırlık dosyası")
    parser.add_argument("--conf", type=float, default=0.25, help="Güven (confidence) eşiği")
    parser.add_argument("--imgsz", type=int, default=640, help="Giriş görsel boyutu")
    parser.add_argument("--show", action="store_true", help="Tespit sonuçlarını ekranda canlı pencerede göster")
    parser.add_argument("--save-dir", type=str, default="results/demo", help="Sonuçların kaydedileceği dizin")

    args = parser.parse_args()
    run_inference(args.source, args.weights, args.conf, args.imgsz, args.show, args.save_dir)
