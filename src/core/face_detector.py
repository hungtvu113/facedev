"""
Module phát hiện khuôn mặt sử dụng MediaPipe Tasks API (mới)
Bao gồm Face Detection + Face Mesh (468 landmarks)
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from dataclasses import dataclass
from typing import List, Optional, Tuple
import urllib.request
import os


@dataclass
class FaceData:
    """Dữ liệu khuôn mặt được phát hiện"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    landmarks: np.ndarray  # 478 landmarks (468 face + 10 iris)
    face_image: np.ndarray  # Ảnh khuôn mặt đã crop
    confidence: float


class FaceDetector:
    """
    Phát hiện và trích xuất khuôn mặt từ ảnh/video
    Sử dụng MediaPipe Tasks API với Face Landmarker (478 điểm)
    """

    # Face Landmarker model (bao gồm cả detection + 478 landmarks)
    LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    LANDMARKER_PATH = "data/face_landmarker.task"

    def __init__(self, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        self.min_confidence = min_detection_confidence

        # Download model nếu chưa có
        self._ensure_model()

        # Khởi tạo Face Landmarker (bao gồm cả detection + mesh)
        base_options = python.BaseOptions(model_asset_path=self.LANDMARKER_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        print(f"[FaceDetector] Initialized with Face Landmarker (478 landmarks)")

    def _ensure_model(self):
        """Download model nếu chưa có"""
        os.makedirs(os.path.dirname(self.LANDMARKER_PATH), exist_ok=True)

        if not os.path.exists(self.LANDMARKER_PATH):
            print(f"[FaceDetector] Downloading Face Landmarker model...")
            urllib.request.urlretrieve(self.LANDMARKER_URL, self.LANDMARKER_PATH)
            print(f"[FaceDetector] Model downloaded to {self.LANDMARKER_PATH}")

    def detect_faces(self, frame: np.ndarray) -> List[FaceData]:
        """
        Phát hiện tất cả khuôn mặt trong frame với 478 landmarks

        Args:
            frame: Ảnh BGR từ OpenCV

        Returns:
            Danh sách FaceData chứa thông tin các khuôn mặt
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Chuyển sang MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Phát hiện khuôn mặt với landmarks
        result = self.landmarker.detect(mp_image)

        faces = []

        # Duyệt qua từng khuôn mặt được phát hiện
        for i, face_landmarks in enumerate(result.face_landmarks):
            # Chuyển landmarks thành numpy array
            landmarks = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face_landmarks])

            # Tính bounding box từ landmarks
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]

            x = int(max(0, np.min(x_coords)))
            y = int(max(0, np.min(y_coords)))
            x_max = int(min(w, np.max(x_coords)))
            y_max = int(min(h, np.max(y_coords)))
            bw = x_max - x
            bh = y_max - y

            # Mở rộng bbox một chút
            padding = int(min(bw, bh) * 0.15)
            x = max(0, x - padding)
            y = max(0, y - padding)
            bw = min(bw + 2 * padding, w - x)
            bh = min(bh + 2 * padding, h - y)

            # Crop khuôn mặt
            face_img = frame[y:y+bh, x:x+bw].copy() if bw > 0 and bh > 0 else None

            if face_img is not None and face_img.size > 0:
                faces.append(FaceData(
                    bbox=(x, y, bw, bh),
                    landmarks=landmarks,
                    face_image=face_img,
                    confidence=0.95  # Face Landmarker không trả về confidence trực tiếp
                ))

        return faces
    
    def align_face(self, face_img: np.ndarray, landmarks: np.ndarray,
                   target_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
        """
        Căn chỉnh khuôn mặt dựa trên vị trí mắt

        Args:
            face_img: Ảnh khuôn mặt
            landmarks: 478 landmarks từ Face Landmarker
            target_size: Kích thước output

        Returns:
            Ảnh khuôn mặt đã căn chỉnh
        """
        if face_img is None or face_img.size == 0:
            return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

        if landmarks is None or len(landmarks) < 468:
            return cv2.resize(face_img, target_size)

        # MediaPipe Face Landmarker: 478 landmarks
        # Mắt trái: 33, 133 | Mắt phải: 362, 263
        left_eye = np.mean(landmarks[[33, 133], :2], axis=0)
        right_eye = np.mean(landmarks[[362, 263], :2], axis=0)

        # Tính góc xoay
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Tâm xoay
        eye_center = ((left_eye[0] + right_eye[0]) / 2,
                      (left_eye[1] + right_eye[1]) / 2)

        # Ma trận xoay
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)

        # Xoay ảnh
        h, w = face_img.shape[:2]
        rotated = cv2.warpAffine(face_img, M, (w, h))

        return cv2.resize(rotated, target_size)

    def draw_detections(self, frame: np.ndarray, faces: List[FaceData],
                        draw_landmarks: bool = True, draw_mesh: bool = True) -> np.ndarray:
        """Vẽ bounding box và landmarks lên frame"""
        output = frame.copy()

        for face in faces:
            x, y, w, h = face.bbox

            # Vẽ bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Vẽ confidence
            cv2.putText(output, f"{face.confidence:.2f}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Vẽ landmarks (478 điểm)
            if draw_landmarks and face.landmarks is not None:
                # Vẽ tất cả các điểm với kích thước nhỏ
                for i, point in enumerate(face.landmarks):
                    # Điểm quan trọng (mắt, mũi, miệng) vẽ to hơn
                    if i in [33, 133, 362, 263, 1, 61, 291, 199]:  # Key points
                        cv2.circle(output, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)
                    else:
                        cv2.circle(output, (int(point[0]), int(point[1])), 1, (0, 255, 255), -1)

                # Vẽ mesh connections (lưới)
                if draw_mesh:
                    self._draw_face_mesh(output, face.landmarks)

        return output

    def _draw_face_mesh(self, frame: np.ndarray, landmarks: np.ndarray):
        """Vẽ lưới kết nối FULL FACE giữa các landmarks (Tesselation)"""

        # FACE MESH TESSELATION - Lưới tam giác full face từ MediaPipe
        # Đây là danh sách các cặp điểm tạo thành lưới trên toàn bộ khuôn mặt
        FACE_MESH_TESSELATION = [
            # Trán và đỉnh đầu
            (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
            (10, 109), (109, 67), (67, 103), (103, 54), (54, 21), (21, 162),
            (162, 127), (127, 234), (234, 93), (93, 132), (132, 58), (58, 172),
            (172, 136), (136, 150), (150, 149), (149, 176), (176, 148), (148, 152),
            (152, 377), (377, 400), (400, 378), (378, 379), (379, 365), (365, 397),
            (397, 288), (288, 361), (361, 323), (323, 454), (454, 356), (356, 389),

            # Vùng trán - lưới ngang
            (10, 151), (151, 9), (9, 8), (8, 168), (168, 6), (6, 197), (197, 195),
            (195, 5), (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (2, 164),
            (151, 337), (337, 299), (299, 333), (333, 298), (298, 301), (301, 368),
            (108, 69), (69, 104), (104, 68), (68, 71), (71, 139), (139, 34),

            # Mắt trái - chi tiết
            (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
            (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
            (159, 160), (160, 161), (161, 246), (246, 33),
            (33, 130), (130, 25), (25, 110), (110, 24), (24, 23), (23, 22),
            (22, 26), (26, 112), (112, 243), (243, 190), (190, 56), (56, 28),

            # Mắt phải - chi tiết
            (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390),
            (390, 249), (249, 263), (263, 466), (466, 388), (388, 387), (387, 386),
            (386, 385), (385, 384), (384, 398), (398, 362),
            (362, 359), (359, 255), (255, 339), (339, 254), (254, 253), (253, 252),
            (252, 256), (256, 341), (341, 463), (463, 414), (414, 286), (286, 258),

            # Lông mày trái
            (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 65),
            (65, 52), (52, 53), (53, 46), (46, 124), (124, 35), (35, 111),

            # Lông mày phải
            (300, 293), (293, 334), (334, 296), (296, 336), (336, 285), (285, 295),
            (295, 282), (282, 283), (283, 276), (276, 353), (353, 265), (265, 340),

            # Mũi - chi tiết
            (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19),
            (19, 94), (94, 2), (2, 98), (98, 97), (97, 99), (99, 100),
            (168, 417), (417, 351), (351, 419), (419, 248), (248, 281), (281, 275),
            (275, 274), (274, 457), (457, 438), (438, 439), (439, 455), (455, 460),
            (2, 326), (326, 327), (327, 328), (328, 329), (329, 330),
            (1, 44), (44, 45), (45, 220), (220, 115), (115, 48), (48, 64),
            (64, 98), (98, 240), (240, 99), (99, 218), (218, 237),

            # Má trái
            (234, 127), (127, 34), (34, 143), (143, 111), (111, 117), (117, 118),
            (118, 119), (119, 120), (120, 121), (121, 128), (128, 245), (245, 193),
            (193, 55), (55, 107), (107, 66), (66, 105), (105, 63), (63, 70),
            (116, 123), (123, 50), (50, 101), (101, 36), (36, 142), (142, 126),

            # Má phải
            (454, 356), (356, 264), (264, 372), (372, 340), (340, 346), (346, 347),
            (347, 348), (348, 349), (349, 350), (350, 357), (357, 465), (465, 417),
            (417, 285), (285, 336), (336, 296), (296, 334), (334, 293), (293, 300),
            (345, 352), (352, 280), (280, 330), (330, 266), (266, 371), (371, 355),

            # Môi trên - chi tiết
            (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
            (267, 269), (269, 270), (270, 409), (409, 291), (291, 375),
            (61, 76), (76, 62), (62, 78), (78, 191), (191, 80), (80, 81),
            (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415),
            (415, 308), (308, 324), (324, 318), (318, 402), (402, 317), (317, 14),

            # Môi dưới - chi tiết
            (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
            (314, 405), (405, 321), (321, 375), (375, 291),
            (61, 77), (77, 90), (90, 180), (180, 85), (85, 16), (16, 315),
            (315, 404), (404, 320), (320, 307), (307, 291),
            (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402),
            (402, 318), (318, 324), (324, 308),

            # Cằm và hàm
            (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
            (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
            (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
            (152, 377), (377, 400), (400, 378), (378, 379), (379, 365), (365, 397),
            (397, 288), (288, 361), (361, 323), (323, 454), (454, 356), (356, 389),
            (389, 251), (251, 284), (284, 332), (332, 297), (297, 338), (338, 10),
            (140, 171), (171, 175), (175, 396), (396, 369), (369, 395),

            # Lưới chéo trên mặt - tạo hiệu ứng 3D
            (151, 108), (108, 69), (69, 67), (67, 109), (109, 10),
            (337, 299), (299, 296), (296, 297), (297, 338),
            (168, 193), (193, 245), (245, 128), (128, 114), (114, 217),
            (168, 417), (417, 465), (465, 357), (357, 343), (343, 437),
            (6, 122), (122, 188), (188, 114), (114, 47), (47, 100),
            (6, 351), (351, 412), (412, 343), (343, 277), (277, 329),
        ]

        # Vẽ lưới tesselation với màu xanh nhạt
        mesh_color = (180, 180, 100)  # Màu xanh olive nhạt
        for connection in FACE_MESH_TESSELATION:
            pt1_idx, pt2_idx = connection
            if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
                pt1 = (int(landmarks[pt1_idx][0]), int(landmarks[pt1_idx][1]))
                pt2 = (int(landmarks[pt2_idx][0]), int(landmarks[pt2_idx][1]))
                cv2.line(frame, pt1, pt2, mesh_color, 1)

        # Vẽ thêm các đường viền quan trọng với màu nổi bật hơn
        # Viền mặt
        face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]

        # Môi
        upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

        # Mắt
        left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
        right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]

        # Lông mày
        left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
        right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

        # Mũi
        nose_bridge = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]
        nose_tip = [98, 97, 2, 326, 327]

        def draw_path(points, color, thickness=1):
            for i in range(len(points) - 1):
                if points[i] < len(landmarks) and points[i+1] < len(landmarks):
                    pt1 = (int(landmarks[points[i]][0]), int(landmarks[points[i]][1]))
                    pt2 = (int(landmarks[points[i+1]][0]), int(landmarks[points[i+1]][1]))
                    cv2.line(frame, pt1, pt2, color, thickness)

        # Vẽ các đường viền nổi bật
        draw_path(face_oval, (150, 150, 150), 1)      # Viền mặt - xám sáng
        draw_path(upper_lip, (0, 220, 220), 1)        # Môi - cyan
        draw_path(lower_lip, (0, 220, 220), 1)
        draw_path(left_eye, (255, 180, 0), 1)         # Mắt - cam
        draw_path(right_eye, (255, 180, 0), 1)
        draw_path(left_eyebrow, (200, 200, 0), 1)     # Lông mày - vàng
        draw_path(right_eyebrow, (200, 200, 0), 1)
        draw_path(nose_bridge, (180, 180, 255), 1)    # Mũi - hồng nhạt
        draw_path(nose_tip, (180, 180, 255), 1)

    def close(self):
        """Giải phóng tài nguyên"""
        # MediaPipe Tasks API tự quản lý resources
        pass

