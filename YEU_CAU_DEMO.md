# Hệ Thống Nhận Dạng Khuôn Mặt

## Mục Lục
1. [Yêu Cầu Demo](#yêu-cầu-demo)
2. [Kiến Trúc Hệ Thống](#kiến-trúc-hệ-thống)
3. [Công Nghệ Sử Dụng](#công-nghệ-sử-dụng)
4. [Chi Tiết Kỹ Thuật](#chi-tiết-kỹ-thuật)

---

## Yêu Cầu Demo

- Tìm hiểu các phương pháp xác định vị trí khuôn mặt trong hệ thống nhận dạng khuôn mặt
- Tìm hiểu các phương pháp trích chọn đặc trưng hình dạng trong hệ thống nhận dạng khuôn mặt
- Tìm hiểu các phương pháp trích chọn đặc trưng hình học trong hệ thống nhận dạng khuôn mặt
- Tìm hiểu các kỹ thuật xác thực dựa trên khuôn mặt
- Tìm hiểu về các kỹ thuật hỗ trợ phân biệt giới tính
- Tìm hiểu về các kỹ thuật nhận dạng cảm xúc

---

## Kiến Trúc Hệ Thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    FACE RECOGNITION SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│  │  Camera  │───►│ Detector │───►│ Encoder  │───►│Recognizer│   │
│  │  Input   │    │(MediaPipe)│   │(Fallback)│    │ (Cosine) │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│                        │                               │         │
│                        ▼                               ▼         │
│                 ┌──────────┐                    ┌──────────┐     │
│                 │ Emotion  │                    │ Database │     │
│                 │ Detector │                    │ (SQLite) │     │
│                 └──────────┘                    └──────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Luồng Xử Lý (Pipeline)

```
Camera Frame
     │
     ▼
┌─────────────────┐
│ 1. Face Detection│  ← MediaPipe Face Landmarker
│    (Phát hiện)   │     Trả về: Bounding Box + 478 Landmarks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Face Encoding │  ← Fallback Encoder (Multi-feature)
│    (Mã hóa)      │     Trả về: Vector 512D
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 3. Recognition   │  ← Cosine Similarity + Threshold
│    (Nhận dạng)   │     So sánh với Database
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Analysis      │  ← Gender + Emotion Detection
│    (Phân tích)   │     Landmark-based Rules
└─────────────────┘
```

---

## Công Nghệ Sử Dụng

### 1. Ngôn Ngữ & Framework

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **Python** | 3.14.2 | Ngôn ngữ lập trình chính |
| **OpenCV** | 4.x | Xử lý ảnh, hiển thị video |
| **NumPy** | 2.x | Tính toán ma trận, vector |
| **MediaPipe** | 0.10.32 | Phát hiện khuôn mặt & landmarks |
| **SQLite** | Built-in | Lưu trữ database |

### 2. Models & Algorithms

| Thành phần | Model/Algorithm | Chi tiết |
|------------|-----------------|----------|
| **Face Detection** | MediaPipe Face Landmarker | CNN-based, 478 landmarks |
| **Face Encoding** | Custom Fallback Encoder | Multi-feature extraction |
| **Face Matching** | Cosine Similarity | Threshold: 0.97 |
| **Gender Detection** | Landmark Geometry Rules | Tỉ lệ khuôn mặt |
| **Emotion Detection** | Landmark Analysis | Độ mở mắt, miệng |

---

## Chi Tiết Kỹ Thuật

### 1. Phát Hiện Khuôn Mặt (Face Detection)

#### Model: MediaPipe Face Landmarker

**Nguồn gốc:** Google MediaPipe Tasks API (2024)

**Kiến trúc:**
- Sử dụng **BlazeFace** - một Single Shot Detector (SSD) được tối ưu cho mobile
- Backbone: MobileNetV2 (lightweight CNN)
- Đầu ra: 478 facial landmarks (468 face mesh + 10 iris points)

**File model:** `face_landmarker.task` (float16, ~4MB)
```
URL: https://storage.googleapis.com/mediapipe-models/face_landmarker/
     face_landmarker/float16/1/face_landmarker.task
```

**478 Landmarks bao gồm:**
```
┌─────────────────────────────────────┐
│  Face Mesh: 468 điểm               │
│  ├── Viền mặt (Face Oval): 36 điểm │
│  ├── Mắt trái: 16 điểm             │
│  ├── Mắt phải: 16 điểm             │
│  ├── Lông mày trái: 8 điểm         │
│  ├── Lông mày phải: 8 điểm         │
│  ├── Mũi: 9 điểm                   │
│  ├── Môi trên: 12 điểm             │
│  ├── Môi dưới: 12 điểm             │
│  └── Các điểm khác trên mặt        │
│                                     │
│  Iris: 10 điểm                      │
│  ├── Mống mắt trái: 5 điểm         │
│  └── Mống mắt phải: 5 điểm         │
└─────────────────────────────────────┘
```

**Ưu điểm:**
- Realtime (30+ FPS trên CPU)
- Chính xác cao (>95% detection rate)
- Cung cấp landmarks chi tiết cho phân tích

---

### 2. Trích Xuất Đặc Trưng Hình Dạng (Shape Features)

#### Phương pháp: Multi-Scale Feature Extraction

Hệ thống trích xuất đặc trưng hình dạng từ ảnh khuôn mặt bằng nhiều kỹ thuật:

**a) Multi-Scale Pixel Sampling (64D)**
```python
# Resize ảnh về nhiều kích thước và lấy mẫu
scales = [8, 16, 32]  # 8x8, 16x16, 32x32
# Flatten và normalize → 64 features
```

**b) Regional Histogram (64D)**
```python
# Chia ảnh thành 4 vùng (2x2 grid)
# Mỗi vùng: tính histogram 16 bins
# Tổng: 4 × 16 = 64 features
```

**c) Edge Detection - Sobel (64D)**
```python
# Sobel X và Y để phát hiện cạnh
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
magnitude = sqrt(sobel_x² + sobel_y²)
direction = arctan2(sobel_y, sobel_x)
# → 32 magnitude + 32 direction features
```

**d) Gabor Texture (64D)**
```python
# Gabor filter với nhiều hướng
orientations = [0°, 45°, 90°, 135°]
frequencies = [0.1, 0.3]
# Mỗi filter → mean + std
# Tổng: 4 × 2 × 2 = 16 features (padded to 64)
```

---

### 3. Trích Xuất Đặc Trưng Hình Học (Geometric Features)

#### Phương pháp: Landmark-Based Geometry (256D)

Sử dụng 478 landmarks để tính các đặc trưng hình học:

**a) Khoảng Cách Giữa Các Điểm Quan Trọng**
```
Các cặp điểm được đo:
├── Mắt trái ↔ Mắt phải (khoảng cách 2 mắt)
├── Mắt ↔ Mũi (chiều cao mắt-mũi)
├── Mũi ↔ Miệng (chiều cao mũi-miệng)
├── Miệng ↔ Cằm (chiều cao miệng-cằm)
├── Má trái ↔ Má phải (chiều rộng mặt)
├── Trán ↔ Cằm (chiều cao mặt)
├── Hàm trái ↔ Hàm phải (chiều rộng hàm)
└── ... (tổng ~50 cặp điểm)
```

**b) Góc Giữa Các Điểm**
```
Các góc được tính:
├── Góc mắt-mũi-mắt
├── Góc miệng trái-mũi-miệng phải
├── Góc lông mày
├── Góc hàm
└── ... (tổng ~20 góc)
```

**c) Tỉ Lệ Khuôn Mặt**
```
Các tỉ lệ quan trọng:
├── Chiều rộng / Chiều cao mặt
├── Khoảng cách 2 mắt / Chiều rộng mặt
├── Chiều rộng mũi / Chiều rộng mặt
├── Chiều rộng miệng / Chiều rộng mặt
├── Chiều rộng hàm / Chiều rộng mặt
└── ... (tổng ~30 tỉ lệ)
```

**d) Vị Trí Tương Đối**
```
Vị trí các điểm so với tâm mặt:
├── Normalized X, Y của mỗi landmark quan trọng
└── Tổng: ~50 vị trí × 2 = 100 features
```

---

### 4. Xác Thực Khuôn Mặt (Face Authentication)

#### Phương pháp: Cosine Similarity Matching

**Vector Embedding:**
```
512D Embedding = Shape Features (256D) + Geometric Features (256D)
```

**Công thức Cosine Similarity:**
```
                    A · B
similarity = ─────────────────
              ||A|| × ||B||

Trong đó:
- A: embedding của khuôn mặt cần nhận dạng
- B: embedding đã lưu trong database
- Kết quả: giá trị từ -1 đến 1 (1 = giống hoàn toàn)
```

**Threshold và Quyết định:**
```
┌─────────────────────────────────────────┐
│  Similarity ≥ 0.97  →  MATCH (Nhận dạng)│
│  Similarity < 0.97  →  UNKNOWN (Lạ)     │
└─────────────────────────────────────────┘

Lý do chọn 0.97:
- Chính chủ đạt ~99% similarity
- Người khác đạt ~96% similarity
- Threshold 0.97 phân biệt được 2 nhóm
```

**Quy trình đăng ký (Multi-Angle Registration):**
```
5 góc độ được chụp:
1. [O] Chính diện (nhìn thẳng)
2. [<] Nghiêng trái nhẹ
3. [>] Nghiêng phải nhẹ
4. [^] Ngẩng lên nhẹ
5. [v] Cúi xuống nhẹ

→ Mỗi góc tạo 1 embedding
→ Khi nhận dạng: so sánh với TẤT CẢ embeddings, lấy MAX similarity
```

---

### 5. Phân Biệt Giới Tính (Gender Detection)

#### Phương pháp: Landmark Geometry Rules

**Nguyên lý:** Khuôn mặt nam và nữ có tỉ lệ hình học khác nhau

**Các đặc điểm được phân tích:**

| Đặc điểm | Nam | Nữ |
|----------|-----|-----|
| Tỉ lệ mặt (rộng/cao) | > 0.88 (vuông) | < 0.80 (thon) |
| Tỉ lệ hàm | > 0.85 (rộng) | < 0.78 (nhỏ) |
| Độ dày môi | < 0.04 (mỏng) | > 0.06 (dày) |

**Landmarks sử dụng:**
```
Điểm 10:  Đỉnh trán
Điểm 152: Cằm
Điểm 234: Má trái
Điểm 454: Má phải
Điểm 33:  Góc mắt trái
Điểm 263: Góc mắt phải
Điểm 172: Hàm trái
Điểm 397: Hàm phải
Điểm 13:  Môi trên
Điểm 14:  Môi dưới
```

**Scoring System:**
```python
male_score = 0

if face_ratio > 0.88:      male_score += 0.3  # Mặt vuông
elif face_ratio < 0.80:    male_score -= 0.3  # Mặt thon

if jaw_ratio > 0.85:       male_score += 0.3  # Hàm rộng
elif jaw_ratio < 0.78:     male_score -= 0.3  # Hàm nhỏ

if lip_ratio < 0.04:       male_score += 0.2  # Môi mỏng
elif lip_ratio > 0.06:     male_score -= 0.2  # Môi dày

# Quyết định
if male_score > 0.15:   → "Male"
elif male_score < -0.15: → "Female"
else:                    → "Unknown"
```

**Độ chính xác:** ~75-85% (rule-based, không dùng AI model)

**Ưu tiên:** Nếu người đã đăng ký → lấy giới tính từ database (100% chính xác)

---

### 6. Nhận Dạng Cảm Xúc (Emotion Detection)

#### Phương pháp: Landmark Analysis

**Nguyên lý:** Cảm xúc thể hiện qua độ mở của mắt và miệng

**Các chỉ số được tính:**

**a) Eye Aspect Ratio (EAR) - Độ mở mắt:**
```
         ||p2 - p6|| + ||p3 - p5||
EAR = ─────────────────────────────
              2 × ||p1 - p4||

Trong đó p1-p6 là 6 điểm quanh mắt

EAR thấp → Mắt nhắm (buồn ngủ, buồn)
EAR cao  → Mắt mở to (ngạc nhiên, sợ)
```

**b) Mouth Aspect Ratio (MAR) - Độ mở miệng:**
```
         ||p2 - p8|| + ||p3 - p7|| + ||p4 - p6||
MAR = ─────────────────────────────────────────────
                    3 × ||p1 - p5||

MAR thấp → Miệng đóng (bình thường, buồn)
MAR cao  → Miệng mở (ngạc nhiên, vui)
```

**c) Mouth Corner Ratio - Góc miệng:**
```
Đo độ cong của miệng (cười hay mếu)
```

**Bảng quyết định cảm xúc:**

| EAR | MAR | Góc miệng | Cảm xúc |
|-----|-----|-----------|---------|
| Cao | Cao | - | 😲 Surprise |
| Thấp | Thấp | Xuống | 😢 Sad |
| Bình thường | Cao | Lên | 😊 Happy |
| Thấp | Thấp | Bình thường | 😐 Neutral |
| Cao | Thấp | Xuống | 😨 Fear |
| Bình thường | Thấp | Xuống | 😠 Angry |

**Độ chính xác:** ~60-70% (rule-based)

---

## Cấu Trúc Thư Mục

```
D:\facedev\
├── main.py                 # Entry point
├── YEU_CAU_DEMO.md         # Tài liệu này
│
├── src/
│   ├── config.py           # Cấu hình (threshold, camera...)
│   │
│   ├── core/
│   │   ├── face_detector.py    # MediaPipe Face Landmarker
│   │   ├── face_encoder.py     # Multi-feature Encoder
│   │   ├── face_recognizer.py  # Cosine Similarity Matching
│   │   └── emotion_detector.py # Gender + Emotion Analysis
│   │
│   ├── database/
│   │   └── db_manager.py       # SQLite Database
│   │
│   └── utils/
│       └── helpers.py          # Drawing utilities
│
├── data/
│   ├── faces.db            # SQLite database
│   └── models/             # MediaPipe model files
│
└── models/                 # Downloaded models
    └── face_landmarker.task
```

---

## Tham Khảo

1. **MediaPipe Face Landmarker**
   - https://developers.google.com/mediapipe/solutions/vision/face_landmarker

2. **BlazeFace Paper**
   - Bazarevsky et al., "BlazeFace: Sub-millisecond Neural Face Detection on Mobile GPUs", 2019

3. **Cosine Similarity in Face Recognition**
   - https://en.wikipedia.org/wiki/Cosine_similarity

4. **Eye Aspect Ratio (EAR)**
   - Soukupová & Čech, "Real-Time Eye Blink Detection using Facial Landmarks", 2016

5. **Facial Geometry for Gender Classification**
   - Burton et al., "Sex Discrimination: How Do We Tell the Difference between Male and Female Faces?", 1993