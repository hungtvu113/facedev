"""
Face Recognition System - Main Application
==========================================
Hệ thống nhận dạng khuôn mặt với các tính năng:
1. Phát hiện khuôn mặt (Face Detection)
2. Trích xuất đặc trưng (Face Embedding)
3. Nhận dạng danh tính (Face Recognition)
4. Phân tích giới tính và cảm xúc (Gender & Emotion)

Phím tắt:
- R: Đăng ký khuôn mặt mới
- L: Xem danh sách người đã đăng ký
- D: Xóa người (nhập ID)
- S: Chụp và lưu ảnh
- Q/ESC: Thoát
"""
import cv2
import numpy as np
import sys
import os

# Thêm src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.face_detector import FaceDetector
from src.core.face_encoder import FaceEncoder
from src.core.face_recognizer import FaceRecognizer
from src.core.emotion_detector import EmotionDetector
from src.database.db_manager import DatabaseManager
from src.utils.helpers import draw_face_box, draw_landmarks, draw_info_panel, draw_stats_box, remove_vietnamese_accents
from src.config import *


class FaceRecognitionApp:
    """Ứng dụng nhận dạng khuôn mặt chính"""
    
    def __init__(self):
        print("=" * 50)
        print("  FACE RECOGNITION SYSTEM")
        print("=" * 50)
        
        # Khởi tạo các modules
        print("\n[1/5] Initializing Face Detector...")
        self.detector = FaceDetector(
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        
        print("[2/5] Initializing Face Encoder...")
        self.encoder = FaceEncoder(model_name="Facenet")
        
        print("[3/5] Initializing Database...")
        self.db = DatabaseManager(DB_PATH)
        
        print("[4/5] Initializing Face Recognizer...")
        self.recognizer = FaceRecognizer(self.db, self.encoder, threshold=RECOGNITION_THRESHOLD)
        
        print("[5/5] Initializing Emotion Detector...")
        self.emotion_detector = EmotionDetector(use_deepface=True)
        
        # Camera
        self.cap = None
        self.is_running = False
        
        # State
        self.current_mode = "recognition"  # recognition, register
        self.register_name = ""
        self.register_samples = []
        self.samples_needed = 5
        
        print("\n[OK] System ready!")
        print("-" * 50)
    
    def start_camera(self):
        """Khởi động camera"""
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("[ERROR] Cannot open camera!")
            return False
        
        print(f"[Camera] Started (Index: {CAMERA_INDEX})")
        return True
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Xử lý một frame"""
        # Phát hiện khuôn mặt
        faces = self.detector.detect_faces(frame)
        
        for face in faces:
            x, y, w, h = face.bbox

            # Nhận dạng (truyền landmarks để cải thiện encoding)
            result = self.recognizer.recognize(face.face_image, face.landmarks)

            # Phân tích cảm xúc
            analysis = self.emotion_detector.analyze(face.face_image, face.landmarks)
            
            # Xác định màu và label
            if result and result.is_match:
                color = COLOR_GREEN
                label = f"{result.name} ({result.confidence:.0%})"
            else:
                color = COLOR_RED
                label = f"Unknown ({result.confidence:.0%})" if result else "Unknown"
            
            # Vẽ bounding box
            frame = draw_face_box(frame, face.bbox, label, color)
            
            # Vẽ landmarks (optional)
            if face.landmarks is not None:
                frame = draw_landmarks(frame, face.landmarks, COLOR_CYAN, radius=1, step=5)
            
            # Hiển thị thông tin
            info_y = y + h + 20

            # Gender từ database (nếu nhận dạng được) hoặc từ analysis
            if result and result.is_match:
                gender_text = result.gender
            elif analysis:
                gender_text = remove_vietnamese_accents(analysis.gender)
            else:
                gender_text = "Unknown"

            cv2.putText(frame, f"Gender: {gender_text}", (x, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MAGENTA, 1)

            # Emotion từ analysis (vẫn dùng AI/fallback)
            if analysis:
                emotion_text = remove_vietnamese_accents(analysis.emotion)
                cv2.putText(frame, f"Emotion: {emotion_text}", (x, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_YELLOW, 1)
        
        # Hiển thị thống kê
        stats = self.db.get_stats()
        stats_display = {
            "Persons": stats["total_persons"],
            "Samples": stats["total_embeddings"],
            "Mode": self.current_mode.upper()
        }
        frame = draw_stats_box(frame, stats_display)
        
        # Hiển thị hướng dẫn
        self._draw_instructions(frame)
        
        return frame
    
    def _draw_instructions(self, frame: np.ndarray):
        """Vẽ hướng dẫn phím tắt"""
        h = frame.shape[0]
        instructions = "R: Register | L: List | D: Delete | S: Screenshot | Q: Quit"
        cv2.putText(frame, instructions, (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1)
    
    def register_face_interactive(self, frame: np.ndarray, faces: list):
        """Đăng ký khuôn mặt tương tác với hướng dẫn xoay mặt"""
        if not faces:
            print("[Register] No face detected!")
            return

        # Nhập tên
        name = input("\nEnter name for registration: ").strip()
        if not name:
            print("[Register] Cancelled - no name provided")
            return

        # Chọn giới tính
        print("\nSelect gender:")
        print("  1. Nam (Male)")
        print("  2. Nu (Female)")
        gender_choice = input("Enter 1 or 2: ").strip()

        if gender_choice == "1":
            gender = "Nam"
        elif gender_choice == "2":
            gender = "Nu"
        else:
            print("[Register] Invalid choice. Defaulting to 'Unknown'")
            gender = "Unknown"

        # Các góc độ cần chụp
        poses = [
            {"name": "Chinh dien (nhin thang)", "icon": "[O]", "delay": 2.0},
            {"name": "Nghieng TRAI nhe", "icon": "[<]", "delay": 2.0},
            {"name": "Nghieng PHAI nhe", "icon": "[>]", "delay": 2.0},
            {"name": "Ngang len nhe", "icon": "[^]", "delay": 2.0},
            {"name": "Cui xuong nhe", "icon": "[v]", "delay": 2.0},
        ]

        print(f"\n[Register] Registering {name}...")
        print(f"[Register] Will capture {len(poses)} poses. Follow the instructions!")
        print("-" * 50)

        samples_collected = 0

        for pose_idx, pose in enumerate(poses):
            print(f"\n  Pose {pose_idx + 1}/{len(poses)}: {pose['name']}")

            # Đếm ngược và chờ người dùng xoay mặt
            countdown = 3  # 3 giây chuẩn bị
            start_time = cv2.getTickCount()
            captured = False

            while not captured:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                faces = self.detector.detect_faces(frame)

                # Tính thời gian đã trôi qua
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                remaining = countdown - elapsed

                # Vẽ landmarks nếu có mặt
                if faces:
                    frame = draw_landmarks(frame, faces[0].landmarks, COLOR_CYAN, radius=1, step=5)
                    x, y, w, h = faces[0].bbox
                    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GREEN, 2)

                # Hiển thị hướng dẫn
                h_frame = frame.shape[0]
                w_frame = frame.shape[1]

                # Background cho text
                cv2.rectangle(frame, (0, 0), (w_frame, 120), (0, 0, 0), -1)

                cv2.putText(frame, f"Registering: {name}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GREEN, 2)
                cv2.putText(frame, f"Pose {pose_idx + 1}/{len(poses)}: {pose['name']}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_YELLOW, 2)

                if remaining > 0:
                    # Đang đếm ngược
                    cv2.putText(frame, f"Get ready... {int(remaining) + 1}", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CYAN, 2)
                    # Vẽ icon lớn ở giữa màn hình
                    cv2.putText(frame, pose['icon'], (w_frame // 2 - 30, h_frame // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2.0, COLOR_YELLOW, 3)
                else:
                    # Chụp ảnh
                    if faces:
                        face = faces[0]
                        success, msg = self.recognizer.register_face(
                            name, face.face_image, gender, face.landmarks
                        )
                        if success:
                            samples_collected += 1
                            captured = True
                            print(f"    [OK] Captured!")
                            # Hiển thị thông báo đã chụp
                            cv2.putText(frame, "CAPTURED!", (w_frame // 2 - 80, h_frame // 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_GREEN, 3)
                            cv2.imshow("Face Recognition System", frame)
                            cv2.waitKey(500)  # Hiển thị 0.5 giây
                    else:
                        cv2.putText(frame, "No face! Look at camera", (10, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_RED, 2)

                cv2.imshow("Face Recognition System", frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC để hủy
                    print("[Register] Cancelled by user")
                    return

        print("-" * 50)
        print(f"[Register] Completed! {name} ({gender}) registered with {samples_collected} samples.")
        print(f"[Register] Now the system can recognize {name}!")

    def list_persons(self):
        """Hiển thị danh sách người đã đăng ký"""
        persons = self.recognizer.get_all_persons()

        print("\n" + "=" * 40)
        print("  REGISTERED PERSONS")
        print("=" * 40)

        if not persons:
            print("  (No persons registered)")
        else:
            print(f"  {'ID':<5} {'Name':<20} {'Samples':<10}")
            print("-" * 40)
            for p in persons:
                print(f"  {p.id:<5} {p.name:<20} {p.embeddings_count:<10}")

        print("=" * 40 + "\n")

    def delete_person_interactive(self):
        """Xóa người tương tác"""
        self.list_persons()

        try:
            person_id = int(input("Enter person ID to delete (0 to cancel): "))
            if person_id == 0:
                print("[Delete] Cancelled")
                return

            confirm = input(f"Delete person ID {person_id}? (y/n): ").lower()
            if confirm == 'y':
                if self.recognizer.delete_person(person_id):
                    print(f"[Delete] Person ID {person_id} deleted successfully")
                else:
                    print(f"[Delete] Failed to delete person ID {person_id}")
            else:
                print("[Delete] Cancelled")
        except ValueError:
            print("[Delete] Invalid ID")

    def run(self):
        """Chạy ứng dụng chính"""
        if not self.start_camera():
            return

        self.is_running = True
        print("\n[Running] Press 'Q' or 'ESC' to quit\n")

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            # Flip frame (mirror)
            frame = cv2.flip(frame, 1)

            # Xử lý frame
            processed_frame = self.process_frame(frame)

            # Hiển thị
            cv2.imshow("Face Recognition System", processed_frame)

            # Xử lý phím
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                self.is_running = False

            elif key == ord('r'):  # Register
                faces = self.detector.detect_faces(frame)
                self.register_face_interactive(frame, faces)

            elif key == ord('l'):  # List
                self.list_persons()

            elif key == ord('d'):  # Delete
                self.delete_person_interactive()

            elif key == ord('s'):  # Screenshot
                filename = f"screenshot_{len(os.listdir('data'))+1}.jpg"
                cv2.imwrite(os.path.join('data', filename), frame)
                print(f"[Screenshot] Saved: {filename}")

        self.cleanup()

    def cleanup(self):
        """Dọn dẹp tài nguyên"""
        print("\n[Cleanup] Releasing resources...")

        if self.cap:
            self.cap.release()

        self.detector.close()
        cv2.destroyAllWindows()

        print("[Cleanup] Done. Goodbye!")


def main():
    """Entry point"""
    app = FaceRecognitionApp()
    app.run()


if __name__ == "__main__":
    main()

