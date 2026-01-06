# Unsupervised Retail Product Identification System
# (Gözetimsiz Perakende Ürün Tanımlama ve Takip Sistemi)

Bu proje, perakende raflarındaki ürünleri gerçek zamanlı olarak tespit etmek, tanımlamak ve takip etmek için geliştirilmiş hibrit bir bilgisayarlı görü sistemidir. 

Sistem, ürünleri tespit etmek için **YOLOv11**, ürünlerin kimliğini belirlemek için **DINOv2** ve kararlı takip için özel bir **Tracker** algoritması kullanır. En büyük özelliği, yeni ürünler için modeli yeniden eğitmeye gerek kalmadan, sadece referans görüntüyü klasöre ekleyerek **Zero-Shot** tanıma yapabilmesidir.

---

## Temel Özellikler

* **Nesne Tespiti (YOLOv11):** Raftaki ürünlerin yerini yüksek hassasiyetle tespit eder.
* **Gözetimsiz Tanımlama (DINOv2):** Tespit edilen ürünleri, önceden tanımlanmış bir "Ürün Kütüphanesi"ndeki referans görsellerle (Embedding Cosine Similarity) eşleştirir.
* **Akıllı Takip (Tracker & Memory):**
    * **Persistence (Süreklilik):** Anlık tespit kayıplarında (miss detection) ürünün ekrandan kaybolmasını engeller.
    * **Score Locking (Skor Kilitleme):** Bir ürün yüksek güven skoruyla (%95) tanındıktan sonra, ışık değişimi vb. nedenlerle skoru düşse bile (%40) en doğru etiketini korur. Etiket titremesini (Flickering) önler.
* **Hibrit Çalışma Ortamı (Docker & Local):**
    * **Local (Windows/Linux):** Videoyu orijinal boyutunda işler, `.mp4` çıktısı verir.
    * **Docker:** Dikey videolar için otomatik 90 derece döndürme ve boyut düzeltme uygular, `.avi` (MJPG) çıktısı verir.
* **GPU Hızlandırma:** `CuPy` ve `Torch` (CUDA) ile matris işlemleri optimize edilmiştir (FP16 desteği).

---

## Proje Yapısı

```text
.
├── main.py                 # Ana uygulama kodu (Tespit, Tanıma, Takip)
├── urun_kutuphanesi/       # Referans ürün görsellerinin bulunduğu klasör (.jpg, .png)
├── otput_video             # Elde edilen çıktı videosu  
├── best.pt                 # Eğitilmiş YOLOv11 ağırlık dosyası
├── planogram.mp4           # İşlenecek giriş videosu
├── Dockerfile              # Docker imajı oluşturma dosyası
├── requirements.txt        # Gerekli Python kütüphaneleri
└── README.md               # Proje dokümantasyonu
```
Kurulum (Local)
Projeyi kendi bilgisayarınızda (GPU önerilir) çalıştırmak için:

Repoyu Klonlayın:

Bash
```text
git clone https://github.com/melekceyhunn/unsupervised-retail-product-id.git
cd unsupervised-retail-product-id
```
Gereksinimleri Yükleyin:

Bash
```text
pip install -r requirements.txt
```
(Not: GPU kullanımı için torch ve cupy kütüphanelerinin CUDA sürümüne uygun olduğundan emin olun.)

Çalıştırın:

Bash
```text
python main.py
```
Local modda çalışırken video orijinal boyutunda açılır ve çıktı output_video_local.mp4 olarak kaydedilir.

Docker ile Kurulum ve Çalıştırma
Proje, dikey videoların Docker ortamında yan dönme sorununu çözen özel bir algoritmaya sahiptir.

1. Docker İmajını Oluşturun (Build)
Bash
```text
docker build -t unsupervised-retail-product-id .
```
2. Konteyneri Başlatın (Run)
Aşağıdaki komut, GPU desteğini açar, çıktı klasörünü bağlar ve Docker ortam değişkenini (DOCKER_ENV) aktif eder.

Windows (PowerShell):

PowerShell
```text
docker run --gpus all -it -v ${PWD}:/app -e NO_DISPLAY=1 -e DOCKER_ENV=true unsupervised-retail-product-id
```
Docker modunda çalışırken video otomatik olarak 90 derece döndürülür ve çıktı Windows uyumlu .avi (MJPG codec) formatında ciktilar/output_video_docker.avi olarak kaydedilir.

---

Nasıl Çalışır? (Teknik Akış)
Başlatma: urun_kutuphanesi klasöründeki resimler DINOv2 ile taranır ve özellik vektörleri (embeddings) GPU hafızasına yüklenir.

Tespit: YOLOv11 gelen video karesindeki tüm şişeleri tespit eder (Bounding Box).

Kırpma & Eşleşme: Tespit edilen alanlar kesilir ve DINOv2'ye gönderilir. Çıkarılan özellikler, kütüphanedeki ürünlerle Cosine Similarity yöntemiyle karşılaştırılır.

Takip (Tracker):

IoU (Intersection over Union): Önceki karedeki kutularla yeni kutular eşleştirilir.

Memory Update: Eşleşen ürünlerin ömrü (life) yenilenir.

Score Locking: Eğer hafızadaki ürünün önceki tanıma skoru yeni gelen skordan yüksekse, eski (doğru) etiket korunur.

Görselleştirme: Sonuçlar kare üzerine çizilir ve video dosyasına yazılır.

---

Gereksinimler

Python 3.8+

NVIDIA GPU (CUDA destekli) - Önerilir

Docker (Opsiyonel, konteynerize çalışma için)
