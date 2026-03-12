import cv2
import mediapipe as mp
import math
import numpy as np

# ===== Khởi tạo mediapipe =====
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
mp_mesh = mp.solutions.face_mesh

face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

face_mesh = mp_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ===== Hàm tính khoảng cách =====
def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

# ===== Hàm chuẩn hóa landmark =====
def normalize_landmarks(lm):
    coords = np.array([[p.x, p.y] for p in lm])
    center = coords[1]  # landmark mũi = 1
    coords -= center
    scale = np.linalg.norm(coords)
    return coords / scale

saved_face = None  # lưu khuôn mặt để xác thực
saved_emotion = None
saved_gender = None

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- B1: Phát hiện khuôn mặt ----
    results = face_detection.process(rgb)
    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(frame, "Khuon mat", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ---- B2: Landmark khuôn mặt ----
    mesh_results = face_mesh.process(rgb)
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:

            mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                mp_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_draw.DrawingSpec(color=(0, 255, 255), thickness=1)
            )

            lm = face_landmarks.landmark

            # ===== B3: Đặc trưng hình dạng =====
            left_eye = lm[33]; right_eye = lm[263]
            mouth_up = lm[13]; mouth_down = lm[14]
            jaw_left = lm[234]; jaw_right = lm[454]
            left_mouth = lm[61]; right_mouth = lm[291]
            left_eyebrow = lm[105]; right_eyebrow = lm[334]
            forehead_left = lm[10]; forehead_right = lm[338]

            eye_dist = distance(left_eye, right_eye)
            mouth_open = distance(mouth_up, mouth_down)
            jaw_dist = distance(jaw_left, jaw_right)
            mouth_width = distance(left_mouth, right_mouth)
            eyebrow_height = (distance(left_eye, left_eyebrow) + distance(right_eye, right_eyebrow))/2
            forehead_width = distance(forehead_left, forehead_right)

            cv2.putText(frame, f"Khoang cach hai mat: {eye_dist:.3f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Do mo mieng: {mouth_open:.3f}", (20, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(frame, f"Khoang cach ham: {jaw_dist:.3f}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # ===== B4: Xác thực khuôn mặt =====
            current_face_norm = normalize_landmarks(lm)
            if saved_face is not None:
                sim = np.dot(saved_face.flatten(), current_face_norm.flatten()) / (
                    np.linalg.norm(saved_face.flatten()) * np.linalg.norm(current_face_norm.flatten()))
                text = "Cung mot nguoi" if sim > 0.95 else "Nguoi khac"
                cv2.putText(frame, text, (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # ===== B5: Phân biệt giới tính nâng cao =====
            ratio1 = jaw_dist / eye_dist
            ratio2 = forehead_width / eye_dist
            gender = "Nam" if ratio1 > 2.1 and ratio2 > 1.1 else "Nu"
            cv2.putText(frame, f"Gioi tinh: {gender}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,255), 2)

            # ===== B6: Nhận dạng cảm xúc nâng cao =====
            if mouth_open > 0.04 and eyebrow_height > 0.03:
                emotion = "Bat ngo"
            elif mouth_width/mouth_open > 1.8:
                emotion = "Cuoi"
            elif mouth_open < 0.02:
                emotion = "Binh thuong"
            else:
                emotion = "Buon"
            cv2.putText(frame, f"Cam xuc: {emotion}", (20, 165),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            # ===== Biểu đồ trực quan =====
            cv2.rectangle(frame, (w-150,20),(w-20,120),(50,50,50),-1)
            cv2.putText(frame, f"Eyes: {eye_dist:.3f}", (w-145,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
            cv2.putText(frame, f"Mouth: {mouth_open:.3f}", (w-145,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
            cv2.putText(frame, f"Jaw: {jaw_dist:.3f}", (w-145,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1)
            cv2.putText(frame, f"Emotion: {emotion}", (w-145,100), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,255),1)

    cv2.putText(frame, "Nhan phim S de luu khuon mat", (20, h-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    cv2.imshow("Demo Nhan dang khuon mat - MediaPipe", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    if key == ord('s') and mesh_results.multi_face_landmarks:
        saved_face = normalize_landmarks(mesh_results.multi_face_landmarks[0].landmark)
        saved_emotion = emotion
        saved_gender = gender
        print("Da luu khuon mat + cam xuc + gioi tinh de demo!")

cap.release()
cv2.destroyAllWindows()