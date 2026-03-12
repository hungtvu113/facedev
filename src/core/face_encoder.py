"""
Module trích xuất đặc trưng khuôn mặt (Face Embedding)
Sử dụng DeepFace với các model: VGG-Face, Facenet, ArcFace
"""
import cv2
import numpy as np
from typing import Optional, Tuple
import os

# Tắt warning của TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class FaceEncoder:
    """
    Trích xuất vector đặc trưng 128D/512D từ khuôn mặt
    Sử dụng DeepFace library
    """
    
    def __init__(self, model_name: str = "Facenet"):
        """
        Khởi tạo encoder
        
        Args:
            model_name: Tên model ("VGG-Face", "Facenet", "Facenet512", "ArcFace", "SFace")
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model DeepFace"""
        try:
            from deepface import DeepFace
            # Pre-load model bằng cách chạy một lần
            dummy_img = np.zeros((160, 160, 3), dtype=np.uint8)
            try:
                DeepFace.represent(
                    dummy_img, 
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend="skip"
                )
            except:
                pass
            self.deepface = DeepFace
            print(f"[FaceEncoder] Loaded model: {self.model_name}")
        except ImportError:
            print("[FaceEncoder] DeepFace not installed. Using fallback encoder.")
            self.deepface = None
    
    def encode(self, face_image: np.ndarray,
               landmarks: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Trích xuất vector embedding từ ảnh khuôn mặt

        Args:
            face_image: Ảnh khuôn mặt BGR (đã crop và align)
            landmarks: 478 landmarks (optional, cho fallback encoder)

        Returns:
            Vector embedding (128D hoặc 512D tùy model)
        """
        if face_image is None or face_image.size == 0:
            return None

        # Resize về kích thước chuẩn
        face_resized = cv2.resize(face_image, (160, 160))

        if self.deepface is not None:
            try:
                result = self.deepface.represent(
                    face_resized,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend="skip"
                )
                if result and len(result) > 0:
                    embedding = np.array(result[0]["embedding"])
                    return embedding
            except Exception as e:
                print(f"[FaceEncoder] Error: {e}")
                return self._fallback_encode(face_resized, landmarks)
        else:
            return self._fallback_encode(face_resized, landmarks)

        return None
    
    def _fallback_encode(self, face_image: np.ndarray,
                         landmarks: np.ndarray = None) -> np.ndarray:
        """
        Fallback encoder khi không có DeepFace
        Kết hợp: image features + landmark geometry
        Luôn trả về vector 512 chiều
        """
        features_list = []

        # === PHẦN 1: Image-based features (256 features) ===
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image

        # 1a. Multi-scale pixel features (64 features)
        for size in [8, 16]:
            small = cv2.resize(gray, (size, size))
            normalized = small.astype(np.float32) / 255.0
            features_list.append(normalized.flatten()[:32])  # 32 * 2 = 64

        # 1b. Histogram features - nhiều vùng (64 features)
        h, w = gray.shape
        regions = [
            gray[:h//2, :w//2],      # Top-left
            gray[:h//2, w//2:],      # Top-right
            gray[h//2:, :w//2],      # Bottom-left
            gray[h//2:, w//2:],      # Bottom-right
        ]
        for region in regions:
            hist = cv2.calcHist([region], [0], None, [16], [0, 256])
            hist = hist.flatten() / (hist.sum() + 1e-6)
            features_list.append(hist)  # 16 * 4 = 64

        # 1c. Edge features - Sobel (64 features)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        edge_dir = np.arctan2(sobely, sobelx)

        edge_small = cv2.resize(edge_mag, (8, 8)).flatten()
        edge_small = edge_small / (np.max(edge_small) + 1e-6)
        features_list.append(edge_small.astype(np.float32)[:32])  # 32

        dir_small = cv2.resize(edge_dir, (8, 8)).flatten()
        dir_small = (dir_small + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        features_list.append(dir_small.astype(np.float32)[:32])  # 32

        # 1d. Gabor-like texture features (64 features)
        gabor_features = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            kernel = cv2.getGaborKernel((5, 5), 1.0, theta, 5.0, 0.5, 0)
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            gabor_small = cv2.resize(filtered, (4, 4)).flatten()
            gabor_small = gabor_small / (np.max(np.abs(gabor_small)) + 1e-6)
            gabor_features.extend(gabor_small[:16])
        features_list.append(np.array(gabor_features, dtype=np.float32))  # 64

        # === PHẦN 2: Landmark geometry features (256 features) ===
        if landmarks is not None and len(landmarks) >= 468:
            landmark_features = self._extract_landmark_features(landmarks)
            features_list.append(landmark_features)  # 256
        else:
            # Nếu không có landmarks, dùng thêm image features
            # HOG-like features
            hog_features = self._compute_simple_hog(gray)
            features_list.append(hog_features[:256])  # 256

        # === Combine all features ===
        embedding = np.concatenate(features_list)

        # Đảm bảo đúng 512 chiều
        if len(embedding) < 512:
            embedding = np.pad(embedding, (0, 512 - len(embedding)))
        else:
            embedding = embedding[:512]

        # L2 Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

        return embedding.astype(np.float32)

    def _extract_landmark_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Trích xuất features từ 478 landmarks
        Tập trung vào tỉ lệ và khoảng cách giữa các điểm quan trọng
        """
        features = []

        # Key landmark indices
        key_points = {
            'forehead': 10,
            'chin': 152,
            'left_cheek': 234,
            'right_cheek': 454,
            'nose_tip': 1,
            'left_eye_outer': 33,
            'left_eye_inner': 133,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'left_mouth': 61,
            'right_mouth': 291,
            'upper_lip': 13,
            'lower_lip': 14,
            'left_eyebrow_outer': 70,
            'left_eyebrow_inner': 107,
            'right_eyebrow_inner': 336,
            'right_eyebrow_outer': 300,
            'left_jaw': 172,
            'right_jaw': 397,
        }

        # Lấy tọa độ các điểm quan trọng
        points = {}
        for name, idx in key_points.items():
            if idx < len(landmarks):
                points[name] = landmarks[idx]

        if len(points) < 10:
            return np.zeros(256, dtype=np.float32)

        # Tính face dimensions để normalize
        face_height = np.linalg.norm(points['forehead'] - points['chin'])
        face_width = np.linalg.norm(points['left_cheek'] - points['right_cheek'])

        if face_height < 1 or face_width < 1:
            return np.zeros(256, dtype=np.float32)

        # === Feature 1: Pairwise distances (normalized) ===
        point_names = list(points.keys())
        for i in range(len(point_names)):
            for j in range(i + 1, len(point_names)):
                dist = np.linalg.norm(points[point_names[i]] - points[point_names[j]])
                normalized_dist = dist / face_height
                features.append(normalized_dist)

        # === Feature 2: Angles between key points ===
        # Góc mắt-mũi-mắt
        v1 = points['left_eye_outer'] - points['nose_tip']
        v2 = points['right_eye_outer'] - points['nose_tip']
        angle = self._angle_between(v1, v2)
        features.append(angle / np.pi)

        # Góc miệng
        v1 = points['left_mouth'] - points['upper_lip']
        v2 = points['right_mouth'] - points['upper_lip']
        angle = self._angle_between(v1, v2)
        features.append(angle / np.pi)

        # === Feature 3: Ratios ===
        # Tỉ lệ mặt
        features.append(face_width / face_height)

        # Tỉ lệ mắt
        eye_width = np.linalg.norm(points['left_eye_outer'] - points['right_eye_outer'])
        features.append(eye_width / face_width)

        # Tỉ lệ miệng
        mouth_width = np.linalg.norm(points['left_mouth'] - points['right_mouth'])
        features.append(mouth_width / face_width)

        # Tỉ lệ hàm
        jaw_width = np.linalg.norm(points['left_jaw'] - points['right_jaw'])
        features.append(jaw_width / face_width)

        # === Feature 4: Relative positions ===
        center = (points['forehead'] + points['chin']) / 2
        for name, point in points.items():
            rel_pos = (point - center) / face_height
            features.extend(rel_pos[:2])  # x, y only

        # Pad hoặc truncate về 256
        features = np.array(features, dtype=np.float32)
        if len(features) < 256:
            features = np.pad(features, (0, 256 - len(features)))
        else:
            features = features[:256]

        return features

    def _angle_between(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Tính góc giữa 2 vectors"""
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-6)
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.arccos(dot)

    def _compute_simple_hog(self, gray: np.ndarray) -> np.ndarray:
        """Tính HOG-like features đơn giản"""
        # Resize
        img = cv2.resize(gray, (64, 64))

        # Gradient
        gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        mag = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx)

        # Chia thành cells 8x8
        features = []
        cell_size = 8
        n_bins = 9

        for i in range(0, 64, cell_size):
            for j in range(0, 64, cell_size):
                cell_mag = mag[i:i+cell_size, j:j+cell_size]
                cell_angle = angle[i:i+cell_size, j:j+cell_size]

                # Histogram of oriented gradients
                hist, _ = np.histogram(cell_angle.flatten(),
                                       bins=n_bins,
                                       range=(-np.pi, np.pi),
                                       weights=cell_mag.flatten())
                hist = hist / (np.sum(hist) + 1e-6)
                features.extend(hist)

        return np.array(features, dtype=np.float32)
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """
        Tính độ tương đồng giữa 2 embeddings (Cosine Similarity)
        
        Args:
            embedding1: Vector embedding 1
            embedding2: Vector embedding 2
            
        Returns:
            Độ tương đồng từ 0 đến 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Chuyển từ [-1, 1] sang [0, 1]
        return (similarity + 1) / 2
    
    def compute_distance(self, embedding1: np.ndarray,
                        embedding2: np.ndarray) -> float:
        """
        Tính khoảng cách Euclidean giữa 2 embeddings
        """
        if embedding1 is None or embedding2 is None:
            return float('inf')
        
        return np.linalg.norm(embedding1 - embedding2)

