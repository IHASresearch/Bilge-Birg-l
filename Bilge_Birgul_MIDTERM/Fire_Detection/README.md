# YOLO Yangın Tespit Sistemi (Fire Detection with HUD Overlay)

Bu proje, YOLO derin öğrenme mimarisini kullanarak video ve kamera akışlarında gerçek zamanlı yangın algılama (Fire Detection) gerçekleştiren, tespit edilen yangın bölgelerini güven skoruyla birlikte çerçeve içine alan ve canlı telemetri/HUD bilgi paneli sunan bir bilgisayarlı görü uygulamasıdır.

![Yangın Tespit Demo](assets/demo.jpg)

## 📌 Özellikler
- **Gerçek Zamanlı Yangın Tespiti:** YOLO modeli ile alev ve yangın odaklarının yüksek doğrulukla tespiti.
- **HUD Bilgi Paneli:** Canlı durum (Yangın Algılandı / Normal), aktif yangın sayısı ve FPS telemetri paneli.
- **Özelleştirilmiş Sınırlayıcı Kutu:** Çift katmanlı kırmızı çerçeve, köşe vurguları ve güven yüzdesi etiketi (`FIRE: %XX`).
- **Geniş Kaynak Desteği:** Video dosyaları (`.mp4`), canlı web kamerası veya RTSP akışları ile uyumlu.
- **Donanım Optimizasyonu:** CUDA destekli GPU veya çok iş parçacıklı CPU desteği.

---

## 🚀 Kurulum

1. Depoyu klonlayın:
```bash
git clone https://github.com/KULLANICI_ADINIZ/fire-detection-yolo.git
cd fire-detection-yolo
```

2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

---

## 💻 Kullanım

### 1. Canlı Web Kamerası ile Çalıştırma
```bash
python main.py --source 0
```

### 2. Video Dosyası Üzerinde Çalıştırma ve Kaydetme
```bash
python main.py --source "ornek_video.mp4" --output "outputs/tespit_sonucu.mp4"
```

### 3. Özel Model veya Güven Eşiği Belirleme
```bash
python main.py --source "ornek_video.mp4" --model "weights/best_final.pt" --conf 0.25
```

---

## 📂 Proje Yapısı
```text
fire-detection-yolo/
├── assets/                  # Demo görsel ve ekran görüntüleri
│   └── demo.jpg
├── weights/                 # Eğitilmiş YOLO ağırlık dosyaları
│   └── best_final.pt
├── outputs/                 # İşlenmiş video kayıt çıktıları
├── main.py                  # Komut satırı parametreli ana çalıştırma betiği
├── fire_video_detector.py   # Toplu video işleme betiği
├── requirements.txt         # Bağımlılık listesi
└── .gitignore               # Büyük video ve geçici dosyaları hariç tutma kuralları
```
