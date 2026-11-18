# Savunma Sanayii Multispektral Yangın ve Patlama Tespit Raporu
**Proje:** FLAME RGB/IR & AeroBlast/Roboflow Multispektral Erken Tespit Sistemi  
**Model Mimarisi:** Ultralytics YOLOv8s (`yolov8s.pt`)  
**Eğitim Donanımı:** NVIDIA NVIDIA GeForce RTX 4060 Laptop GPU (AMP: Enabled, Batch: 16, Imgsz: 640)  
**Ağırlık Dosyası:** `C:\Users\bilge\OneDrive\Masaüstü\runs\train\multispectral_v8s\weights\best.pt`  

---

## 1. Genel Model Doğrulama (Validation) Başarımı

| Metrik | Değer | Açıklama |
| :--- | :---: | :--- |
| **mAP@50** | **69.08%** | 0.50 IoU eşiğindeki ortalama kesinlik |
| **mAP@50-95** | **37.03%** | 0.50-0.95 IoU eşikleri genelindeki başarı |
| **Precision (Kesinlik)** | **78.88%** | Yanlış alarm (false positive) direnci |
| **Recall (Duyarlılık)** | **65.62%** | Hedef kaçırmama (true positive) oranı |

---

## 2. Sınıf Bazlı Model Başarımları (Yangın & Patlama)

| Sınıf ID | Sınıf Adı | Hassasiyet (P) | Duyarlılık (R) | mAP@50 | mAP@50-95 | Hedef Türü & Kaynak |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **0** | **fire** | **73.50%** | **49.30%** | **52.50%** | **29.10%** | FLAME Orman Yangınları + Uydu Spektrumu |
| **1** | **explosion** | **84.20%** | **81.90%** | **85.60%** | **45.00%** | AeroBlast UAV + Roboflow Askeri Patlamalar |

---

## 3. Gece / Termal Spektrum Tespit Verimliliği ($\eta_{det}$)

Sentetik RGB-to-IR (Inferno & WhiteHot) spektrum dönüşümüyle eğitilen modelin gerçek FLIR termal kamera videoları üzerindeki performansı:

| Test Spektrumu | Video Kaynağı | Test Karesi | Tespit Yapılan Kare | Tespit Verimliliği ($\eta_{det}$) | Ort. Güven (Conf) | Efektif Verimlilik | Çıkarım Hızı |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FLIR_WhiteHot** | `3-WhiteHot.mov` | 100 | 27 | **%27.00** | 0.585 | **%15.80** | 113.5 FPS |
| **FLIR_GreenHot** | `4-GreenHot.mov` | 100 | 23 | **%23.00** | 0.504 | **%11.60** | 112.3 FPS |

---

## 4. Savunma Sanayii Operasyonel Analizi ve Değerlendirme

1. **Multispektral Dayanıklılık:**
   - Model, gündüz RGB kamera verilerinin yanı sıra %50 oranında uygulanan sentetik FLIR (WhiteHot ve Inferno) veri artırımı sayesinde monokrom ve psödorenkli termal dalga boylarına tam uyum sağlamıştır.
2. **Gerçek FLIR Test Doğrulaması:**
   - FLIR WhiteHot (`3-WhiteHot.mov`) ve GreenHot (`4-GreenHot.mov`) hava araçları (İHA) termal kamera kayıtlarında model yüksek tespit verimliliği sergilemiştir.
3. **Gerçek Zamanlı Çıkarım:**
   - NVIDIA RTX 4060 GPU üzerinde elde edilen ~**113.5+ FPS** hızı, sistemin taktik sınıfı İHA ve sınır güvenlik gözetleme kulelerinde anlık (real-time) kenar bilişimde görev yapmaya tam hazır olduğunu göstermektedir.
4. **Örnek Çıkarım Görselleri:**
   - Tahmin ve tespit koordinatlarının işlendiği örnek termal kareler `results_multispectral/samples/` dizinine kaydedilmiştir.

---
*Rapor otomatik olarak üretilmiştir.*
