"""
Cấu hình hệ thống Face Recognition
"""
import os

# Đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
DB_PATH = os.path.join(DATA_DIR, "faces.db")

# Face Detection
DETECTION_CONFIDENCE = 0.5
TRACKING_CONFIDENCE = 0.5

# Face Recognition
# Ngưỡng để xác định cùng một người (cao hơn = khắt khe hơn)
# 0.6 = dễ nhận nhầm, 0.75-0.8 = cân bằng, 0.85+ = rất khắt khe
# Với fallback encoder: cần threshold cao hơn để tránh nhận nhầm
# 0.98 = an toàn cho bảo mật (chỉ chấp nhận 98%+ similarity)
RECOGNITION_THRESHOLD = 0.98
EMBEDDING_SIZE = 512  # Kích thước vector embedding

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# UI Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_MAGENTA = (255, 0, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)

# Emotions
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
EMOTIONS_VI = ["Tức giận", "Ghê tởm", "Sợ hãi", "Vui vẻ", "Buồn", "Bất ngờ", "Bình thường"]

# Gender
GENDERS = ["Male", "Female"]
GENDERS_VI = ["Nam", "Nữ"]

