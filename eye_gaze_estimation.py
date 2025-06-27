import cv2
import mediapipe as mp
import math

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Utility functions
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def midpoint(p1, p2):
    return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

def is_eye_contact(landmarks, w, h):
    def to_px(idx):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    # Eye landmarks
    left_outer, left_inner = to_px(33), to_px(133)
    right_inner, right_outer = to_px(362), to_px(263)
    left_iris, right_iris = to_px(468), to_px(473)

    # Eye centers
    left_center = midpoint(left_outer, left_inner)
    right_center = midpoint(right_outer, right_inner)

    # Offsets
    left_offset = euclidean(left_iris, left_center) / (euclidean(left_outer, left_inner) + 1e-6)
    right_offset = euclidean(right_iris, right_center) / (euclidean(right_outer, right_inner) + 1e-6)

    # Threshold for contact
    threshold = 0.15
    if left_offset < threshold and right_offset < threshold:
        return True
    return False

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks (optional)
            mp_drawing.draw_landmarks(
                frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,0,255), thickness=1)
            )

            # Eye contact logic
            if is_eye_contact(face_landmarks.landmark, w, h):
                cv2.putText(frame, "Eye Contact: YES", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Eye Contact: NO", (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2)

    cv2.imshow("Eye Contact Detector", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
