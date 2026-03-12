"""
Module nhận dạng cảm xúc và giới tính
Sử dụng DeepFace hoặc fallback với rule-based
"""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@dataclass
class AnalysisResult:
    """Kết quả phân tích khuôn mặt"""
    gender: str
    gender_confidence: float
    emotion: str
    emotion_confidence: float
    age: int = None


class EmotionDetector:
    """
    Nhận dạng cảm xúc và giới tính từ khuôn mặt
    """
    
    EMOTIONS_VI = {
        "angry": "Tức giận",
        "disgust": "Ghê tởm", 
        "fear": "Sợ hãi",
        "happy": "Vui vẻ",
        "sad": "Buồn",
        "surprise": "Bất ngờ",
        "neutral": "Bình thường"
    }
    
    GENDERS_VI = {
        "Man": "Nam",
        "Woman": "Nữ"
    }
    
    def __init__(self, use_deepface: bool = True):
        self.use_deepface = use_deepface
        self.deepface = None
        
        if use_deepface:
            self._load_deepface()
    
    def _load_deepface(self):
        """Load DeepFace library"""
        try:
            from deepface import DeepFace
            self.deepface = DeepFace
            print("[EmotionDetector] DeepFace loaded successfully")
        except ImportError:
            print("[EmotionDetector] DeepFace not available, using fallback")
            self.deepface = None
    
    def analyze(self, face_image: np.ndarray, 
                landmarks: np.ndarray = None) -> Optional[AnalysisResult]:
        """
        Phân tích khuôn mặt để lấy giới tính và cảm xúc
        
        Args:
            face_image: Ảnh khuôn mặt BGR
            landmarks: 468 landmarks (optional, cho fallback)
            
        Returns:
            AnalysisResult hoặc None
        """
        if face_image is None or face_image.size == 0:
            return None
        
        if self.deepface is not None:
            return self._analyze_deepface(face_image)
        else:
            return self._analyze_fallback(face_image, landmarks)
    
    def _analyze_deepface(self, face_image: np.ndarray) -> Optional[AnalysisResult]:
        """Phân tích sử dụng DeepFace"""
        try:
            # Resize để tăng tốc
            face_resized = cv2.resize(face_image, (224, 224))
            
            result = self.deepface.analyze(
                face_resized,
                actions=['gender', 'emotion', 'age'],
                enforce_detection=False,
                detector_backend='skip',
                silent=True
            )
            
            if result and len(result) > 0:
                analysis = result[0]
                
                # Gender
                gender = analysis.get('dominant_gender', 'Unknown')
                gender_conf = analysis.get('gender', {}).get(gender, 0) / 100
                
                # Emotion
                emotion = analysis.get('dominant_emotion', 'neutral')
                emotion_conf = analysis.get('emotion', {}).get(emotion, 0) / 100
                
                # Age
                age = analysis.get('age', None)
                
                return AnalysisResult(
                    gender=self.GENDERS_VI.get(gender, gender),
                    gender_confidence=gender_conf,
                    emotion=self.EMOTIONS_VI.get(emotion, emotion),
                    emotion_confidence=emotion_conf,
                    age=age
                )
        except Exception as e:
            print(f"[EmotionDetector] DeepFace error: {e}")
        
        return None
    
    def _analyze_fallback(self, face_image: np.ndarray,
                          landmarks: np.ndarray = None) -> AnalysisResult:
        """
        Fallback: Phân tích dựa trên landmarks geometry
        Dùng 478 landmarks từ MediaPipe để đoán giới tính
        """
        gender = "Unknown"
        gender_conf = 0.5
        emotion = "Binh thuong"
        emotion_conf = 0.5

        # Nếu có đủ landmarks (478 điểm), dùng geometry-based detection
        if landmarks is not None and len(landmarks) >= 468:
            gender, gender_conf = self._detect_gender_by_landmarks(landmarks)
            emotion, emotion_conf = self._detect_emotion_by_landmarks(landmarks)
        elif face_image is not None and face_image.size > 0:
            # Fallback về phương pháp cũ nếu không có landmarks
            gender, gender_conf = self._detect_gender_by_image(face_image)
            emotion, emotion_conf = self._detect_emotion_by_image(face_image)

        return AnalysisResult(
            gender=gender,
            gender_confidence=gender_conf,
            emotion=emotion,
            emotion_confidence=emotion_conf
        )

    def _detect_gender_by_landmarks(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Đoán giới tính dựa trên tỉ lệ khuôn mặt từ 478 landmarks

        Key landmarks (MediaPipe Face Landmarker):
        - 10: Đỉnh trán
        - 152: Cằm
        - 234: Má trái
        - 454: Má phải
        - 33: Mắt trái (outer)
        - 263: Mắt phải (outer)
        - 13: Môi trên
        - 14: Môi dưới
        - 172: Hàm trái
        - 397: Hàm phải
        """
        try:
            # Lấy các điểm quan trọng
            forehead = landmarks[10]      # Đỉnh trán
            chin = landmarks[152]         # Cằm
            left_cheek = landmarks[234]   # Má trái
            right_cheek = landmarks[454]  # Má phải
            left_eye = landmarks[33]      # Mắt trái outer
            right_eye = landmarks[263]    # Mắt phải outer
            upper_lip = landmarks[13]     # Môi trên
            lower_lip = landmarks[14]     # Môi dưới
            left_jaw = landmarks[172]     # Hàm trái
            right_jaw = landmarks[397]    # Hàm phải

            # Tính các tỉ lệ
            face_height = np.linalg.norm(forehead - chin)
            face_width = np.linalg.norm(left_cheek - right_cheek)
            jaw_width = np.linalg.norm(left_jaw - right_jaw)
            eye_distance = np.linalg.norm(left_eye - right_eye)
            lip_thickness = np.linalg.norm(upper_lip - lower_lip)

            # Tránh chia cho 0
            if face_height < 1 or face_width < 1:
                return "Unknown", 0.5

            # Tính các feature ratio
            face_ratio = face_width / face_height          # Tỉ lệ mặt (rộng/cao)
            jaw_ratio = jaw_width / face_width             # Tỉ lệ hàm
            eye_ratio = eye_distance / face_width          # Tỉ lệ khoảng cách mắt
            lip_ratio = lip_thickness / face_height        # Tỉ lệ độ dày môi

            # Scoring system:
            # Nam: mặt vuông hơn, hàm rộng hơn, môi mỏng hơn
            # Nữ: mặt thon hơn, hàm hẹp hơn, môi dày hơn

            male_score = 0.0

            # Face ratio: Nam ~0.85-0.95, Nữ ~0.75-0.85
            if face_ratio > 0.88:
                male_score += 0.25
            elif face_ratio < 0.80:
                male_score -= 0.25

            # Jaw ratio: Nam > 0.85, Nữ < 0.80
            if jaw_ratio > 0.85:
                male_score += 0.30
            elif jaw_ratio < 0.78:
                male_score -= 0.30

            # Eye distance ratio: Không khác biệt nhiều
            if eye_ratio > 0.42:
                male_score += 0.10
            elif eye_ratio < 0.38:
                male_score -= 0.10

            # Lip ratio: Nữ có môi dày hơn
            if lip_ratio < 0.04:
                male_score += 0.20
            elif lip_ratio > 0.06:
                male_score -= 0.20

            # Quyết định giới tính
            if male_score > 0.15:
                return "Nam", min(0.85, 0.6 + male_score)
            elif male_score < -0.15:
                return "Nu", min(0.85, 0.6 - male_score)
            else:
                return "Unknown", 0.5

        except Exception as e:
            return "Unknown", 0.5

    def _detect_emotion_by_landmarks(self, landmarks: np.ndarray) -> Tuple[str, float]:
        """
        Đoán cảm xúc dựa trên landmarks

        Key landmarks cho emotion:
        - 61, 291: Khóe miệng trái/phải
        - 13, 14: Môi trên/dưới (độ mở miệng)
        - 159, 145: Mí mắt trên/dưới trái
        - 386, 374: Mí mắt trên/dưới phải
        """
        try:
            # Miệng
            left_mouth = landmarks[61]
            right_mouth = landmarks[291]
            upper_lip = landmarks[13]
            lower_lip = landmarks[14]

            # Mắt
            left_eye_upper = landmarks[159]
            left_eye_lower = landmarks[145]
            right_eye_upper = landmarks[386]
            right_eye_lower = landmarks[374]

            # Tính các metric
            mouth_width = np.linalg.norm(left_mouth - right_mouth)
            mouth_height = np.linalg.norm(upper_lip - lower_lip)
            left_eye_open = np.linalg.norm(left_eye_upper - left_eye_lower)
            right_eye_open = np.linalg.norm(right_eye_upper - right_eye_lower)

            avg_eye_open = (left_eye_open + right_eye_open) / 2
            mouth_ratio = mouth_height / (mouth_width + 1e-6)

            # Phân loại cảm xúc
            if mouth_ratio > 0.5 and avg_eye_open > 8:
                return "Bat ngo", 0.7
            elif mouth_ratio > 0.3 and avg_eye_open > 6:
                return "Vui ve", 0.7
            elif mouth_ratio < 0.1 and avg_eye_open < 4:
                return "Buon", 0.6
            elif mouth_ratio > 0.4:
                return "Vui ve", 0.6
            else:
                return "Binh thuong", 0.6

        except Exception:
            return "Binh thuong", 0.5

    def _detect_gender_by_image(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Fallback: Đoán giới tính từ ảnh (độ tương phản)"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)

        if contrast > 50:
            return "Nam", min(0.6, contrast / 100)
        else:
            return "Nu", min(0.6, (60 - contrast) / 60)

    def _detect_emotion_by_image(self, face_image: np.ndarray) -> Tuple[str, float]:
        """Fallback: Đoán cảm xúc từ ảnh (độ sáng)"""
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        upper_brightness = np.mean(gray[:h//2, :])
        lower_brightness = np.mean(gray[h//2:, :])
        contrast = np.std(gray)

        brightness_ratio = lower_brightness / (upper_brightness + 1e-6)

        if brightness_ratio > 1.1:
            return "Bat ngo", 0.6
        elif brightness_ratio < 0.9:
            return "Vui ve", 0.6
        elif contrast < 30:
            return "Buon", 0.5
        else:
            return "Binh thuong", 0.6
