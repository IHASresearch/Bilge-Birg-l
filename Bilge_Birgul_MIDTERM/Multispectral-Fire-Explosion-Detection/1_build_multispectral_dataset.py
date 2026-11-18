"""
1_build_multispectral_dataset.py
--------------------------------
Kaynak Veri Seti Tarama, YOLO Etiketleme ve Sentetik IR Veri Artırma Pipeline'ı.

- Kaynak ana dizin: C:\\Savunma_RGB_IR_Datasets (pathlib.Path ile yönetilir)
- Sınıflar:
    0: fire
    1: explosion
- Veri Setleri:
    1. FLAME Dataset (Maskelerden cv2.findContours ile YOLO bbox üretimi -> 0: fire)
    2. Satellite Fire Dataset (Uydu yangın etiketleri -> 0: fire)
    3. AeroBlast & Roboflow Datasets (Patlama etiketleri -> 1: explosion)
- RGB-to-IR Sentetik Termal Veri Artırma:
    Train setindeki RGB resimlerin %50'si için FLIR WhiteHot ve Inferno simülasyonları
- %85 Train, %15 Val bölümleme ve data.yaml oluşturma
- Windows Unicode / Türkçe karakter (Masaüstü) uyumlu I/O fonksiyonları
"""

import os
import sys
import random
import shutil
import zipfile
from pathlib import Path
import numpy as np

# Python ortamı güvencesi (Torch & CUDA destekli Python 3.11 kontrolü)
PY311_PATH = Path(r"C:\Users\bilge\AppData\Local\Programs\Python\Python311\python.exe")
if sys.executable.lower() != str(PY311_PATH).lower() and PY311_PATH.exists():
    try:
        import cv2
        import yaml
        from tqdm import tqdm
    except ImportError:
        print(f"[*] Yeniden baslatiliyor: {PY311_PATH}")
        os.execv(str(PY311_PATH), [str(PY311_PATH)] + sys.argv)

import cv2
import yaml
from tqdm import tqdm

# Sabit tohum (tekrarlanabilirlik için)
random.seed(42)
np.random.seed(42)

# Dizin Yolları (pathlib.Path)
CURRENT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = CURRENT_DIR / "dataset_multispectral"
RESULTS_ROOT = CURRENT_DIR / "results_multispectral"
SOURCE_ROOT = Path(r"C:\Savunma_RGB_IR_Datasets")

# Alt Dizinler
TRAIN_IMG_DIR = DATASET_ROOT / "images" / "train"
VAL_IMG_DIR = DATASET_ROOT / "images" / "val"
TRAIN_LBL_DIR = DATASET_ROOT / "labels" / "train"
VAL_LBL_DIR = DATASET_ROOT / "labels" / "val"
SAMPLES_DIR = RESULTS_ROOT / "samples"


def cv2_imread_utf8(path, flags=cv2.IMREAD_COLOR):
    """Windows üzerinde Türkçe / Unicode yollardan güvenli görsel okuma."""
    try:
        with open(str(path), "rb") as f:
            buffer = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(buffer, flags)
    except Exception as e:
        return None


def cv2_imwrite_utf8(path, img, quality=90):
    """Windows üzerinde Türkçe / Unicode yollara güvenli görsel yazma."""
    try:
        ext = Path(path).suffix
        is_success, buffer = cv2.imencode(ext, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if is_success:
            with open(str(path), "wb") as f:
                f.write(buffer)
            return True
    except Exception as e:
        pass
    return False


def setup_directories():
    """Çalışma ve hedef klasör hiyerarşisini oluşturur."""
    print("[*] Klasör yapısı hazırlanıyor...")
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR, SAMPLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print(f"[*] Hedef Veri Seti Dizini: {DATASET_ROOT}")
    print(f"[*] Sonuçlar Dizini: {RESULTS_ROOT}")


def extract_satellite_zip_if_needed():
    """Uydu yangın zip arşivi açılmamışsa çıkartır."""
    sat_dir = SOURCE_ROOT / "2_Satellite_Fire"
    zip_files = list(sat_dir.glob("*.zip"))
    if not zip_files:
        return sat_dir

    extracted_dir = sat_dir / "extracted"
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        print(f"[*] Uydu veri seti zaten açık: {extracted_dir}")
        return extracted_dir

    print(f"[*] Uydu yangın zip arşivi açılıyor: {zip_files[0].name}...")
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_files[0], 'r') as zf:
        zf.extractall(extracted_dir)
    print(f"[+] Uydu yangın verisi çıkarıldı: {extracted_dir}")
    return extracted_dir


def process_flame_dataset():
    """
    1_FLAME_Dataset:
    Images ve Masks eşleştirilir.
    Maskelerdeki beyaz piksellerden (cv2.findContours) dikdörtgen koordinatlar çıkarılır.
    0: fire sınıfı olarak YOLO formatına dönüştürülür.
    """
    print("\n--- 1. FLAME Yangın Veri Seti İşleniyor ---")
    flame_root = SOURCE_ROOT / "1_FLAME_Dataset"
    img_dir = flame_root / "Images" / "Images"
    mask_dir = flame_root / "Masks" / "Masks"

    if not img_dir.exists() or not mask_dir.exists():
        print(f"[!] FLAME klasörleri bulunamadı: {img_dir} veya {mask_dir}")
        return []

    samples = []
    mask_files = {p.stem: p for p in mask_dir.glob("*.png")}
    img_files = list(img_dir.glob("*.jpg"))

    print(f"[*] Toplam {len(img_files)} FLAME görüntüsü taranıyor...")
    for img_path in tqdm(img_files, desc="FLAME BBox Çıkarımı"):
        mask_path = mask_files.get(img_path.stem)
        if not mask_path:
            continue

        mask = cv2_imread_utf8(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape[:2]
        # FLAME maskeleri 0 ve 1 değerlerindedir
        binary_mask = (mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        yolo_lines = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if area < 6 or bw < 3 or bh < 3:
                continue

            x_center = (bx + bw / 2.0) / w
            y_center = (by + bh / 2.0) / h
            norm_w = bw / float(w)
            norm_h = bh / float(h)

            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        if yolo_lines:
            samples.append({
                "source": "FLAME",
                "img_path": img_path,
                "lines": yolo_lines,
                "prefix": "flame_"
            })

    print(f"[+] FLAME: {len(samples)} etiketli yangın görüntüsü hazırlandı.")
    return samples


def process_satellite_dataset(sat_root):
    """
    2_Satellite_Fire:
    data.yaml ve .txt etiketleri okunur, tüm etiketler 0 (fire) sınıfına eşlenir.
    """
    print("\n--- 2. Uydu Yangın Veri Seti İşleniyor ---")
    samples = []
    lbl_files = list(sat_root.rglob("*.txt"))
    lbl_files = [p for p in lbl_files if "labels" in p.parts and p.stem != "README"]

    print(f"[*] Toplam {len(lbl_files)} Uydu etiket dosyası bulundu.")
    random.shuffle(lbl_files)
    selected_lbls = lbl_files[:2500]

    for lbl_path in tqdm(selected_lbls, desc="Uydu Yangın Eşleme"):
        img_candidates = [
            lbl_path.parents[1] / "images" / f"{lbl_path.stem}.jpg",
            lbl_path.parents[1] / "images" / f"{lbl_path.stem}.png"
        ]
        img_path = None
        for cand in img_candidates:
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        try:
            content = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except Exception:
            continue

        yolo_lines = []
        for line in content:
            parts = line.strip().split()
            if len(parts) >= 5:
                yolo_lines.append(f"0 {' '.join(parts[1:5])}")

        if yolo_lines:
            samples.append({
                "source": "SATELLITE",
                "img_path": img_path,
                "lines": yolo_lines,
                "prefix": "sat_"
            })

    print(f"[+] Uydu Yangın: {len(samples)} görüntü hazırlandı.")
    return samples


def process_explosion_aeroblast():
    """
    3_Explosion_AeroBlast:
    UAV patlama veri seti. Tüm etiketler 1 (explosion) sınıfına eşlenir.
    """
    print("\n--- 3. AeroBlast Patlama Veri Seti İşleniyor ---")
    samples = []
    aero_root = SOURCE_ROOT / "3_Explosion_AeroBlast"
    lbl_files = list(aero_root.rglob("*.txt"))
    lbl_files = [p for p in lbl_files if "labels" in p.parts and not p.name.startswith("README")]

    print(f"[*] Toplam {len(lbl_files)} AeroBlast etiket dosyası bulundu.")
    random.shuffle(lbl_files)
    selected_lbls = lbl_files[:3200]

    for lbl_path in tqdm(selected_lbls, desc="AeroBlast Patlama Eşleme"):
        img_candidates = [
            lbl_path.parents[1] / "images" / f"{lbl_path.stem}.jpg",
            lbl_path.parents[1] / "images" / f"{lbl_path.stem}.png"
        ]
        img_path = None
        for cand in img_candidates:
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        try:
            content = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except Exception:
            continue

        yolo_lines = []
        for line in content:
            parts = line.strip().split()
            if len(parts) >= 5:
                yolo_lines.append(f"1 {' '.join(parts[1:5])}")

        if yolo_lines:
            samples.append({
                "source": "AEROBLAST",
                "img_path": img_path,
                "lines": yolo_lines,
                "prefix": "aero_"
            })

    print(f"[+] AeroBlast: {len(samples)} patlama görüntüsü hazırlandı.")
    return samples


def process_explosion_roboflow():
    """
    4_Explosion_Roboflow:
    Roboflow patlama veri seti. Tüm etiketler 1 (explosion) sınıfına eşlenir.
    """
    print("\n--- 4. Roboflow Patlama Veri Seti İşleniyor ---")
    samples = []
    robo_root = SOURCE_ROOT / "4_Explosion_Roboflow"
    lbl_files = list(robo_root.rglob("*.txt"))
    lbl_files = [p for p in lbl_files if "labels" in p.parts and not p.name.startswith("README")]

    print(f"[*] Toplam {len(lbl_files)} Roboflow etiket dosyası bulundu.")
    for lbl_path in tqdm(lbl_files, desc="Roboflow Patlama Eşleme"):
        img_candidates = [
            lbl_path.parents[1] / "images" / f"{lbl_path.stem}.jpg",
            lbl_path.parents[1] / "images" / f"{lbl_path.stem}.png"
        ]
        img_path = None
        for cand in img_candidates:
            if cand.exists():
                img_path = cand
                break
        if img_path is None:
            continue

        try:
            content = lbl_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        except Exception:
            continue

        yolo_lines = []
        for line in content:
            parts = line.strip().split()
            if len(parts) >= 5:
                yolo_lines.append(f"1 {' '.join(parts[1:5])}")

        if yolo_lines:
            samples.append({
                "source": "ROBOFLOW",
                "img_path": img_path,
                "lines": yolo_lines,
                "prefix": "robo_"
            })

    print(f"[+] Roboflow: {len(samples)} patlama görüntüsü hazırlandı.")
    return samples


def generate_synthetic_ir(img_bgr, mode="inferno"):
    """
    RGB-to-IR (Sentetik Kızılötesi / Gece Görüş) Spektrum Dönüşümü:
    1. Görüntüyü gri tonlamalı kızılötesi yoğunluk haritasına çevir.
    2. Düşük ışık / gece kontrastı uygula (CLAHE + arka plan karartma).
    3. FLIR termal kamera simülasyonu (cv2.COLORMAP_INFERNO veya WhiteHot termal kopyalar).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    norm = enhanced_gray.astype(np.float32) / 255.0
    thermal_intensity = np.power(norm, 1.25)
    thermal_8u = np.clip(thermal_intensity * 255.0, 0, 255).astype(np.uint8)

    if mode == "inferno":
        ir_img = cv2.applyColorMap(thermal_8u, cv2.COLORMAP_INFERNO)
    else:
        ir_img = cv2.cvtColor(thermal_8u, cv2.COLOR_GRAY2BGR)

    return ir_img


def build_and_split_dataset(all_samples):
    """
    Veriyi %85 train, %15 val olarak böler, diske yazar ve
    train setinin %50'sine otomatik RGB-to-IR sentetik artırma uygular.
    """
    print("\n--- 5. Veri Bölümleme (%85 Train, %15 Val) ve Yazma ---")
    random.shuffle(all_samples)

    split_idx = int(len(all_samples) * 0.85)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"[*] Toplam Örnek Sayısı: {len(all_samples)}")
    print(f"[*] Train Kümesi: {len(train_samples)} (%85)")
    print(f"[*] Val Kümesi:   {len(val_samples)} (%15)")

    # 1. Val Kümesini diske yaz (orijinal RGB)
    print("\n[*] Val kümesi kaydediliyor...")
    val_saved = 0
    for idx, sample in enumerate(tqdm(val_samples, desc="Val Yazma")):
        stem = f"{sample['prefix']}val_{idx:05d}"
        dest_img = VAL_IMG_DIR / f"{stem}.jpg"
        dest_lbl = VAL_LBL_DIR / f"{stem}.txt"

        img = cv2_imread_utf8(sample["img_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        success = cv2_imwrite_utf8(dest_img, img, quality=90)
        if success:
            dest_lbl.write_text("\n".join(sample["lines"]), encoding="utf-8")
            val_saved += 1

    # 2. Train Kümesini diske yaz (orijinal RGB)
    print("\n[*] Train kümesi ve RGB-to-IR sentetik artırma uygulanıyor...")
    num_ir = int(len(train_samples) * 0.50)
    ir_indices = set(random.sample(range(len(train_samples)), num_ir))

    train_saved = 0
    ir_created_count = 0
    for idx, sample in enumerate(tqdm(train_samples, desc="Train & IR")):
        stem = f"{sample['prefix']}tr_{idx:05d}"
        dest_img = TRAIN_IMG_DIR / f"{stem}.jpg"
        dest_lbl = TRAIN_LBL_DIR / f"{stem}.txt"

        img = cv2_imread_utf8(sample["img_path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        if max(h, w) > 1280:
            scale = 1280.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        label_text = "\n".join(sample["lines"])
        success = cv2_imwrite_utf8(dest_img, img, quality=90)
        if success:
            dest_lbl.write_text(label_text, encoding="utf-8")
            train_saved += 1

        # %50 Sentetik IR Artırma
        if idx in ir_indices and success:
            mode = "inferno" if (idx % 2 == 0) else "whitehot"
            ir_img = generate_synthetic_ir(img, mode=mode)

            ir_img_dest = TRAIN_IMG_DIR / f"{stem}_ir.jpg"
            ir_lbl_dest = TRAIN_LBL_DIR / f"{stem}_ir.txt"

            if cv2_imwrite_utf8(ir_img_dest, ir_img, quality=90):
                ir_lbl_dest.write_text(label_text, encoding="utf-8")
                ir_created_count += 1

    print(f"\n[+] Train Orijinal RGB: {train_saved}")
    print(f"[+] Üretilen Sentetik IR (_ir.jpg): {ir_created_count}")
    print(f"[+] Toplam Train Görsel Sayısı: {train_saved + ir_created_count}")
    print(f"[+] Toplam Val Görsel Sayısı: {val_saved}")


def create_data_yaml():
    """YOLO eğitimi için data.yaml dosyasını oluşturur."""
    print("\n--- 6. data.yaml Yapılandırması Oluşturuluyor ---")
    data_yaml_path = DATASET_ROOT / "data.yaml"

    yaml_data = {
        "path": str(DATASET_ROOT.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "fire",
            1: "explosion"
        }
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

    print(f"[+] data.yaml başarıyla oluşturuldu: {data_yaml_path}")
    print(yaml.dump(yaml_data, sort_keys=False, allow_unicode=True))


def main():
    print("=" * 65)
    print(" SAVUNMA SANAYİİ MULTİSPEKTRAL VERİ SETİ DÖNÜŞTÜRME PİPELİNE'I")
    print("=" * 65)

    if not SOURCE_ROOT.exists():
        print(f"[!] HATA: Kaynak dizin bulunamadı: {SOURCE_ROOT}")
        sys.exit(1)

    setup_directories()
    sat_extracted = extract_satellite_zip_if_needed()

    flame_samples = process_flame_dataset()
    sat_samples = process_satellite_dataset(sat_extracted)
    aero_samples = process_explosion_aeroblast()
    robo_samples = process_explosion_roboflow()

    all_samples = flame_samples + sat_samples + aero_samples + robo_samples
    print(f"\n[*] Toplam Birleştirilmiş Örnek: {len(all_samples)}")

    if not all_samples:
        print("[!] Hata: Hiçbir örnek toplanamadı!")
        sys.exit(1)

    build_and_split_dataset(all_samples)
    create_data_yaml()

    print("\n" + "=" * 65)
    print(" VERİ SETİ HAZIRLAMA VE SENTETİK IR DÖNÜŞÜMÜ TAMAMLANDI!")
    print("=" * 65)


if __name__ == "__main__":
    main()
