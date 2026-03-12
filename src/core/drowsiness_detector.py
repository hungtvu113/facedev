import numpy as np
from typing import Tuple

class DrowsinessDetector:
    """
    Module phân tích trạng thái buồn ngủ của tài xế dựa trên Landmarks
    Tính toán EAR (Mắt) và MAR (Miệng)
    """
    def __init__(self, ear_thresh=0.22, mar_thresh=0.6, ear_frames=15, mar_frames=10):
        self.ear_thresh = ear_thresh
        self.mar_thresh = mar_thresh
        self.ear_frames = ear_frames
        self.mar_frames = mar_frames
        
        self.counter_sleep = 0
        self.counter_yawn = 0
        self.total_yawns = 0
        self.alarm_on = False
        
        # Chỉ số landmarks của MediaPipe (478 điểm)
        # Mắt trái
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        # Mắt phải
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Miệng (Trái, Trên, Phải, Dưới)
        self.MOUTH = [78, 13, 308, 14]

    def _euclidean_dist(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def calculate_ear(self, landmarks, eye_indices):
        """Tính toán Eye Aspect Ratio (Độ mở của mắt)"""
        # Trích xuất tọa độ
        p1 = landmarks[eye_indices[0]][:2]
        p2 = landmarks[eye_indices[1]][:2]
        p3 = landmarks[eye_indices[2]][:2]
        p4 = landmarks[eye_indices[3]][:2]
        p5 = landmarks[eye_indices[4]][:2]
        p6 = landmarks[eye_indices[5]][:2]

        # Khoảng cách dọc
        vert1 = self._euclidean_dist(p2, p6)
        vert2 = self._euclidean_dist(p3, p5)
        # Khoảng cách ngang
        horiz = self._euclidean_dist(p1, p4)

        ear = (vert1 + vert2) / (2.0 * horiz + 1e-6)
        return ear

    def calculate_mar(self, landmarks):
        """Tính toán Mouth Aspect Ratio (Độ mở của miệng)"""
        left = landmarks[self.MOUTH[0]][:2]
        top = landmarks[self.MOUTH[1]][:2]
        right = landmarks[self.MOUTH[2]][:2]
        bottom = landmarks[self.MOUTH[3]][:2]

        vert = self._euclidean_dist(top, bottom)
        horiz = self._euclidean_dist(left, right)

        mar = vert / (horiz + 1e-6)
        return mar

    def process(self, landmarks: np.ndarray) -> Tuple[str, bool, float, float]:
        """
        Xử lý 1 frame để phát hiện trạng thái
        Returns: (Trạng thái text, Có bật còi không, EAR, MAR)
        """
        if landmarks is None or len(landmarks) < 468:
            return "Khong the phan tich", False, 0.0, 0.0

        # Tính toán
        left_ear = self.calculate_ear(landmarks, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks, self.RIGHT_EYE)
        ear = (left_ear + right_ear) / 2.0
        
        mar = self.calculate_mar(landmarks)

        status = "Tinh tao"
        self.alarm_on = False

        # Kiểm tra Ngủ gật (Nhắm mắt)
        if ear < self.ear_thresh:
            self.counter_sleep += 1
            if self.counter_sleep >= self.ear_frames:
                self.alarm_on = True
                status = "NGU GAT!!!"
        else:
            self.counter_sleep = 0

        # Kiểm tra Ngáp (Mở to miệng)
        if mar > self.mar_thresh:
            self.counter_yawn += 1
            status = "Dang ngap..."
        else:
            if self.counter_yawn >= self.mar_frames:
                self.total_yawns += 1
            self.counter_yawn = 0

        # Nếu vừa ngáp nhiều vừa nhắm mắt
        if self.alarm_on:
            status = "NGUY HIEM: NGU GAT!"

        return status, self.alarm_on, ear, mar
