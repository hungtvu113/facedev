"""
Test camera và face detection đơn giản
"""
import cv2
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.face_detector import FaceDetector

print("=" * 50)
print("  TEST CAMERA & FACE DETECTION")
print("=" * 50)

# Test camera
print("\n[1] Testing camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Cannot open camera!")
    print("Please check if your webcam is connected and not used by another app.")
    sys.exit(1)

print("[OK] Camera opened successfully!")

# Test face detector
print("\n[2] Initializing Face Detector...")
try:
    detector = FaceDetector()
    print("[OK] Face Detector initialized!")
except Exception as e:
    print(f"[ERROR] Failed to initialize Face Detector: {e}")
    cap.release()
    sys.exit(1)

print("\n[3] Starting video capture...")
print("Press 'Q' to quit\n")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame")
        break
    
    frame_count += 1
    
    # Flip frame
    frame = cv2.flip(frame, 1)
    
    # Detect faces
    faces = detector.detect_faces(frame)

    # Draw detections với mesh (478 landmarks + lưới kết nối)
    frame = detector.draw_detections(frame, faces, draw_landmarks=True, draw_mesh=True)
    
    # Show info
    cv2.putText(frame, f"Faces: {len(faces)} | Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (10, frame.shape[0] - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Display
    cv2.imshow("Test Camera - Face Detection", frame)
    
    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n[Done] Processed {frame_count} frames")
cap.release()
cv2.destroyAllWindows()
detector.close()
print("[Cleanup] Complete!")

