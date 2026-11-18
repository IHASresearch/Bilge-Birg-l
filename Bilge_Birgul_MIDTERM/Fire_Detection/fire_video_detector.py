import os
import cv2
import time
import math
import torch
from ultralytics import YOLO

# CPU çoklu iş parçacığı optimizasyonu
torch.set_num_threads(os.cpu_count() or 4)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Model yolu (önce weights klasörüne, sonra ana dizine bakar)
MODEL_PATH = os.path.join(CURRENT_DIR, "weights", "best_final.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(CURRENT_DIR, "best_final.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = r"C:\Users\bilge\Downloads\best_final.pt"

print(f"[*] Yüklenen Model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# İşlenecek video listesi (mevcut klasördeki veya downloads klasöründeki videolar)
VIDEO_CANDIDATES = [
    "14297196_3840_2160_24fps.mp4",
    "14296700_3840_2160_24fps.mp4",
    "12773175_3840_2160_24fps.mp4"
]

OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONF_THRESHOLD = 0.20  # Yangın tespit güven eşiği
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080

def draw_fire_detection(frame, x1, y1, x2, y2, confidence_pct):
    # Ana yangın kutusu (kırmızı ve çift katmanlı kalın çerçeve)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 140, 255), 1)

    # Köşe vurguları
    line_len = min(25, max(8, (x2 - x1) // 3), max(8, (y2 - y1) // 3))
    c_color = (0, 255, 255)
    thick = 3
    # Sol-üst
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), c_color, thick)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), c_color, thick)
    # Sağ-üst
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), c_color, thick)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), c_color, thick)
    # Sol-alt
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), c_color, thick)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), c_color, thick)
    # Sağ-alt
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), c_color, thick)
    cv2.line(frame, (x2, y2), (x2, y2 - line_len), c_color, thick)

    # Etiket ve yüzde yazısı
    label = f"FIRE: %{confidence_pct}"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.75
    font_thickness = 2
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

    # Etiket arka planı
    label_y1 = max(0, y1 - th - 12)
    label_y2 = y1
    label_x1 = x1
    label_x2 = x1 + tw + 16

    # Yarı saydam veya koyu kırmızı etiket arka planı
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 0, 180), -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (0, 255, 255), 1)
    cv2.putText(frame, label, (label_x1 + 8, label_y2 - 6), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

def process_video(video_path):
    video_name = os.path.basename(video_path)
    output_path = os.path.join(OUTPUT_DIR, f"detected_{video_name}")
    
    print(f"\n==========================================")
    print(f"[*] İşleniyor: {video_name}")
    print(f"[*] Hedef Çıktı: {output_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[!] HATA: Video açılamadı: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (TARGET_WIDTH, TARGET_HEIGHT))

    frame_idx = 0
    detected_frames_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        # 1920x1080 ölçeklendirme
        frame_resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_LINEAR)

        # YOLO tahmin (Sadece class 0: Fire)
        results = model.predict(frame_resized, imgsz=640, conf=CONF_THRESHOLD, classes=[0], verbose=False)[0]

        fire_boxes = results.boxes
        fire_count = len(fire_boxes)
        if fire_count > 0:
            detected_frames_count += 1

        for box in fire_boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
            conf_pct = int(box.conf[0] * 100)
            draw_fire_detection(frame_resized, bx1, by1, bx2, by2, conf_pct)

        # Bilgi Paneli (Üst sol köşe)
        overlay = frame_resized.copy()
        cv2.rectangle(overlay, (20, 20), (450, 110), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame_resized, 0.3, 0, frame_resized)
        cv2.rectangle(frame_resized, (20, 20), (450, 110), (80, 80, 80), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        if fire_count > 0:
            status_text = f"DURUM: YANGIN ALGILANDI ({fire_count})"
            status_color = (0, 0, 255)
        else:
            status_text = "DURUM: NORMAL"
            status_color = (0, 255, 0)

        cv2.putText(frame_resized, status_text, (35, 55), font, 0.75, status_color, 2, cv2.LINE_AA)
        cv2.putText(frame_resized, f"Kare: {frame_idx}/{total_frames} | FPS: {fps:.1f}", (35, 85), font, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        out.write(frame_resized)

        if frame_idx % 50 == 0 or frame_idx == total_frames:
            elapsed = time.time() - start_time
            proc_fps = frame_idx / elapsed if elapsed > 0 else 0
            pct = (frame_idx / total_frames) * 100
            print(f"  -> İlerleme: %{pct:5.1f} ({frame_idx}/{total_frames} kare) | İşleme Hızı: {proc_fps:.1f} fps")

    cap.release()
    out.release()
    total_time = time.time() - start_time
    print(f"[+] Tamamlandı: {output_path}")
    print(f"[+] Toplam Süre: {total_time:.1f} sn | Yangın Tespit Edilen Kare Sayısı: {detected_frames_count}/{total_frames}")

def main():
    print("=" * 60)
    print("       YOLO Yangın Tespit & Video Kayıt Sistemi")
    print("=" * 60)
    
    for vname in VIDEO_CANDIDATES:
        local_path = os.path.join(CURRENT_DIR, vname)
        download_path = os.path.join(r"C:\Users\bilge\Downloads", vname)
        if os.path.exists(local_path):
            process_video(local_path)
        elif os.path.exists(download_path):
            process_video(download_path)
        else:
            print(f"[!] Video bulunamadı: {vname}")

    print("\n[✔] Tüm videolar başarıyla işlendi ve kaydedildi!")

if __name__ == "__main__":
    main()
