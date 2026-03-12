"""
Các hàm tiện ích cho Face Recognition System
"""
import cv2
import numpy as np
from typing import Tuple, List
import unicodedata
import re


def remove_vietnamese_accents(text: str) -> str:
    """
    Chuyển tiếng Việt có dấu thành không dấu
    Ví dụ: "Bất ngờ" -> "Bat ngo"
    """
    # Chuẩn hóa Unicode
    text = unicodedata.normalize('NFD', text)
    # Loại bỏ các ký tự dấu
    text = re.sub(r'[\u0300-\u036f]', '', text)
    # Xử lý các ký tự đặc biệt tiếng Việt
    text = text.replace('đ', 'd').replace('Đ', 'D')
    return text


def draw_face_box(frame: np.ndarray, bbox: Tuple[int, int, int, int],
                  label: str = "", color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2) -> np.ndarray:
    """
    Vẽ bounding box và label lên frame
    """
    x, y, w, h = bbox
    
    # Vẽ box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # Vẽ label background
    if label:
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w + 10, y), color, -1)
        cv2.putText(frame, label, (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray,
                   color: Tuple[int, int, int] = (0, 255, 255),
                   radius: int = 1, step: int = 3,
                   draw_mesh: bool = True) -> np.ndarray:
    """
    Vẽ facial landmarks và mesh lên frame
    """
    if landmarks is None:
        return frame

    # Vẽ mesh connections (lưới) nếu có đủ 468 landmarks
    if draw_mesh and len(landmarks) >= 468:
        _draw_face_mesh(frame, landmarks)

    # Vẽ các điểm
    for i in range(0, len(landmarks), step):
        point = landmarks[i]
        cv2.circle(frame, (int(point[0]), int(point[1])), radius, color, -1)

    return frame


def _draw_face_mesh(frame: np.ndarray, landmarks: np.ndarray):
    """Vẽ lưới kết nối FULL FACE giữa các landmarks (Tesselation)"""

    # FACE MESH TESSELATION - Lưới tam giác full face từ MediaPipe
    FACE_MESH_TESSELATION = [
        # Trán và đỉnh đầu
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
        (10, 109), (109, 67), (67, 103), (103, 54), (54, 21), (21, 162),
        (162, 127), (127, 234), (234, 93), (93, 132), (132, 58), (58, 172),
        (172, 136), (136, 150), (150, 149), (149, 176), (176, 148), (148, 152),
        (152, 377), (377, 400), (400, 378), (378, 379), (379, 365), (365, 397),
        (397, 288), (288, 361), (361, 323), (323, 454), (454, 356), (356, 389),

        # Vùng trán
        (10, 151), (151, 9), (9, 8), (8, 168), (168, 6), (6, 197), (197, 195),
        (195, 5), (5, 4), (4, 1), (1, 19), (19, 94), (94, 2), (2, 164),
        (151, 337), (337, 299), (299, 333), (333, 298), (298, 301), (301, 368),
        (108, 69), (69, 104), (104, 68), (68, 71), (71, 139), (139, 34),

        # Mắt trái
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
        (154, 155), (155, 133), (133, 173), (173, 157), (157, 158), (158, 159),
        (159, 160), (160, 161), (161, 246), (246, 33),
        (33, 130), (130, 25), (25, 110), (110, 24), (24, 23), (23, 22),
        (22, 26), (26, 112), (112, 243), (243, 190), (190, 56), (56, 28),

        # Mắt phải
        (362, 382), (382, 381), (381, 380), (380, 374), (374, 373), (373, 390),
        (390, 249), (249, 263), (263, 466), (466, 388), (388, 387), (387, 386),
        (386, 385), (385, 384), (384, 398), (398, 362),
        (362, 359), (359, 255), (255, 339), (339, 254), (254, 253), (253, 252),
        (252, 256), (256, 341), (341, 463), (463, 414), (414, 286), (286, 258),

        # Lông mày
        (70, 63), (63, 105), (105, 66), (66, 107), (107, 55), (55, 65),
        (65, 52), (52, 53), (53, 46), (46, 124), (124, 35), (35, 111),
        (300, 293), (293, 334), (334, 296), (296, 336), (336, 285), (285, 295),
        (295, 282), (282, 283), (283, 276), (276, 353), (353, 265), (265, 340),

        # Mũi
        (168, 6), (6, 197), (197, 195), (195, 5), (5, 4), (4, 1), (1, 19),
        (19, 94), (94, 2), (2, 98), (98, 97), (97, 99), (99, 100),
        (168, 417), (417, 351), (351, 419), (419, 248), (248, 281), (281, 275),
        (2, 326), (326, 327), (327, 328), (328, 329), (329, 330),
        (1, 44), (44, 45), (45, 220), (220, 115), (115, 48), (48, 64),

        # Má trái
        (234, 127), (127, 34), (34, 143), (143, 111), (111, 117), (117, 118),
        (118, 119), (119, 120), (120, 121), (121, 128), (128, 245), (245, 193),
        (116, 123), (123, 50), (50, 101), (101, 36), (36, 142), (142, 126),

        # Má phải
        (454, 356), (356, 264), (264, 372), (372, 340), (340, 346), (346, 347),
        (347, 348), (348, 349), (349, 350), (350, 357), (357, 465), (465, 417),
        (345, 352), (352, 280), (280, 330), (330, 266), (266, 371), (371, 355),

        # Môi trên
        (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
        (267, 269), (269, 270), (270, 409), (409, 291), (291, 375),
        (61, 76), (76, 62), (62, 78), (78, 191), (191, 80), (80, 81),
        (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415),

        # Môi dưới
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
        (314, 405), (405, 321), (321, 375), (375, 291),
        (61, 77), (77, 90), (90, 180), (180, 85), (85, 16), (16, 315),
        (95, 88), (88, 178), (178, 87), (87, 14), (14, 317), (317, 402),

        # Cằm và hàm
        (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
        (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
        (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
        (152, 377), (377, 400), (400, 378), (378, 379), (379, 365), (365, 397),
        (397, 288), (288, 361), (361, 323), (323, 454), (454, 356), (356, 389),
        (389, 251), (251, 284), (284, 332), (332, 297), (297, 338), (338, 10),

        # Lưới chéo
        (151, 108), (108, 69), (69, 67), (67, 109), (109, 10),
        (337, 299), (299, 296), (296, 297), (297, 338),
        (168, 193), (193, 245), (245, 128), (128, 114), (114, 217),
        (168, 417), (417, 465), (465, 357), (357, 343), (343, 437),
    ]

    # Vẽ lưới tesselation
    mesh_color = (180, 180, 100)
    for connection in FACE_MESH_TESSELATION:
        pt1_idx, pt2_idx = connection
        if pt1_idx < len(landmarks) and pt2_idx < len(landmarks):
            pt1 = (int(landmarks[pt1_idx][0]), int(landmarks[pt1_idx][1]))
            pt2 = (int(landmarks[pt2_idx][0]), int(landmarks[pt2_idx][1]))
            cv2.line(frame, pt1, pt2, mesh_color, 1)

    # Vẽ các đường viền nổi bật
    face_oval = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
    upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
    lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    left_eye = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
    right_eye = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
    left_eyebrow = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
    right_eyebrow = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
    nose_bridge = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

    def draw_path(points, color, thickness=1):
        for i in range(len(points) - 1):
            if points[i] < len(landmarks) and points[i+1] < len(landmarks):
                pt1 = (int(landmarks[points[i]][0]), int(landmarks[points[i]][1]))
                pt2 = (int(landmarks[points[i+1]][0]), int(landmarks[points[i+1]][1]))
                cv2.line(frame, pt1, pt2, color, thickness)

    draw_path(face_oval, (150, 150, 150), 1)
    draw_path(upper_lip, (0, 220, 220), 1)
    draw_path(lower_lip, (0, 220, 220), 1)
    draw_path(left_eye, (255, 180, 0), 1)
    draw_path(right_eye, (255, 180, 0), 1)
    draw_path(left_eyebrow, (200, 200, 0), 1)
    draw_path(right_eyebrow, (200, 200, 0), 1)
    draw_path(nose_bridge, (180, 180, 255), 1)


def calculate_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Tính khoảng cách Euclidean giữa 2 điểm"""
    return np.linalg.norm(p1 - p2)


def draw_info_panel(frame: np.ndarray, info: dict,
                    position: Tuple[int, int] = (10, 30),
                    font_scale: float = 0.6) -> np.ndarray:
    """
    Vẽ panel thông tin lên frame
    """
    x, y = position
    line_height = 30
    
    for i, (key, value) in enumerate(info.items()):
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x, y + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
    
    return frame


def draw_stats_box(frame: np.ndarray, stats: dict,
                   position: str = "top-right") -> np.ndarray:
    """
    Vẽ box thống kê ở góc frame
    """
    h, w = frame.shape[:2]
    box_w, box_h = 180, 120
    padding = 10
    
    if position == "top-right":
        x = w - box_w - padding
        y = padding
    elif position == "bottom-right":
        x = w - box_w - padding
        y = h - box_h - padding
    else:
        x, y = padding, padding
    
    # Vẽ background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (50, 50, 50), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Vẽ text
    line_height = 20
    for i, (key, value) in enumerate(stats.items()):
        text = f"{key}: {value}"
        cv2.putText(frame, text, (x + 10, y + 20 + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame


def resize_frame(frame: np.ndarray, max_width: int = 1280,
                 max_height: int = 720) -> np.ndarray:
    """
    Resize frame giữ tỷ lệ
    """
    h, w = frame.shape[:2]
    
    if w > max_width or h > max_height:
        scale = min(max_width / w, max_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h))
    
    return frame


def create_face_grid(faces: List[np.ndarray], cols: int = 4,
                     cell_size: int = 100) -> np.ndarray:
    """
    Tạo grid hiển thị nhiều khuôn mặt
    """
    if not faces:
        return np.zeros((cell_size, cell_size * cols, 3), dtype=np.uint8)
    
    rows = (len(faces) + cols - 1) // cols
    grid = np.zeros((rows * cell_size, cols * cell_size, 3), dtype=np.uint8)
    
    for i, face in enumerate(faces):
        row = i // cols
        col = i % cols
        
        resized = cv2.resize(face, (cell_size, cell_size))
        y1 = row * cell_size
        x1 = col * cell_size
        grid[y1:y1+cell_size, x1:x1+cell_size] = resized
    
    return grid

import threading
import platform

_is_playing_alarm = False

def play_alarm():
    """Phát âm thanh bíp cảnh báo (Chạy trên luồng riêng để không block camera)"""
    global _is_playing_alarm
    if _is_playing_alarm:
        return
        
    def _beep():
        global _is_playing_alarm
        _is_playing_alarm = True
        try:
            if platform.system() == "Windows":
                import winsound
                # Tần số 2500Hz, kéo dài 500ms
                winsound.Beep(2500, 500)
            else:
                print('\a')  # Tiếng chuông terminal cho Linux/Mac
        except:
            pass
        finally:
            _is_playing_alarm = False

    threading.Thread(target=_beep, daemon=True).start()

