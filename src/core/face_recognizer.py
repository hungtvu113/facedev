"""
Module nhận dạng khuôn mặt - So sánh với database
"""
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from .face_encoder import FaceEncoder


@dataclass
class RecognitionResult:
    """Kết quả nhận dạng"""
    person_id: int
    name: str
    confidence: float
    is_match: bool
    gender: str = "Unknown"


class FaceRecognizer:
    """
    Nhận dạng khuôn mặt bằng cách so sánh với database
    """
    
    def __init__(self, db_manager: DatabaseManager, encoder: FaceEncoder,
                 threshold: float = 0.6):
        """
        Args:
            db_manager: Database manager instance
            encoder: Face encoder instance
            threshold: Ngưỡng để xác định match (0-1)
        """
        self.db = db_manager
        self.encoder = encoder
        self.threshold = threshold
        
        # Cache embeddings để tăng tốc
        self._embeddings_cache = []
        self._load_embeddings()
    
    def _load_embeddings(self):
        """Load tất cả embeddings từ database vào cache"""
        self._embeddings_cache = self.db.get_all_embeddings()
        print(f"[Recognizer] Loaded {len(self._embeddings_cache)} embeddings from database")
    
    def refresh_cache(self):
        """Refresh cache khi có thay đổi trong database"""
        self._load_embeddings()
    
    def recognize(self, face_image: np.ndarray,
                  landmarks: np.ndarray = None) -> Optional[RecognitionResult]:
        """
        Nhận dạng một khuôn mặt

        Args:
            face_image: Ảnh khuôn mặt đã crop
            landmarks: 478 landmarks (optional, để cải thiện encoding)

        Returns:
            RecognitionResult hoặc None nếu không nhận dạng được
        """
        # Trích xuất embedding (với landmarks nếu có)
        embedding = self.encoder.encode(face_image, landmarks)
        if embedding is None:
            return None

        return self.recognize_by_embedding(embedding)
    
    def recognize_by_embedding(self, embedding: np.ndarray) -> Optional[RecognitionResult]:
        """
        Nhận dạng dựa trên embedding đã có
        """
        if len(self._embeddings_cache) == 0:
            return RecognitionResult(
                person_id=-1,
                name="Unknown",
                confidence=0.0,
                is_match=False
            )
        
        best_match = None
        best_similarity = 0.0
        best_gender = "Unknown"

        # So sánh với tất cả embeddings trong cache
        # Cache format: (person_id, name, gender, embedding)
        for item in self._embeddings_cache:
            if len(item) == 4:
                person_id, name, gender, db_embedding = item
            else:
                # Fallback cho format cũ
                person_id, name, db_embedding = item
                gender = "Unknown"

            similarity = self.encoder.compute_similarity(embedding, db_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (person_id, name)
                best_gender = gender

        if best_match and best_similarity >= self.threshold:
            return RecognitionResult(
                person_id=best_match[0],
                name=best_match[1],
                confidence=best_similarity,
                is_match=True,
                gender=best_gender
            )
        else:
            return RecognitionResult(
                person_id=-1,
                name="Unknown",
                confidence=best_similarity,
                is_match=False,
                gender="Unknown"
            )
    
    def register_face(self, name: str, face_image: np.ndarray,
                      gender: str = None,
                      landmarks: np.ndarray = None) -> Tuple[bool, str]:
        """
        Đăng ký khuôn mặt mới vào database

        Args:
            name: Tên người
            face_image: Ảnh khuôn mặt
            gender: Giới tính (optional)
            landmarks: 478 landmarks (optional)

        Returns:
            (success, message)
        """
        # Trích xuất embedding (với landmarks nếu có)
        embedding = self.encoder.encode(face_image, landmarks)
        if embedding is None:
            return False, "Không thể trích xuất đặc trưng khuôn mặt"
        
        # Thêm người vào database (hoặc lấy ID nếu đã tồn tại)
        person_id = self.db.add_person(name, gender)
        if person_id < 0:
            return False, "Lỗi khi thêm người vào database"
        
        # Thêm embedding
        success = self.db.add_embedding(person_id, embedding)
        if success:
            self.refresh_cache()
            count = len(self.db.get_person_embeddings(person_id))
            return True, f"Đã lưu khuôn mặt cho {name} (Tổng: {count} mẫu)"
        else:
            return False, "Lỗi khi lưu embedding"
    
    def get_all_persons(self) -> List:
        """Lấy danh sách tất cả người trong database"""
        return self.db.get_all_persons()
    
    def delete_person(self, person_id: int) -> bool:
        """Xóa một người khỏi database"""
        success = self.db.delete_person(person_id)
        if success:
            self.refresh_cache()
        return success

