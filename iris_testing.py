import cv2
import mediapipe as mp
import numpy as np

# Constants for eye indices
LEFT_EYE_INDICES = [33, 133]        # Eye corners
RIGHT_EYE_INDICES = [362, 263]      # Eye corners
LEFT_IRIS_INDICES = [468, 469, 470, 471]  # Approximate iris center = mean of these
RIGHT_IRIS_INDICES = [473, 474, 475, 476]

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

def compute_eye_contact(iris_center, eye_corners):
    left_corner, right_corner = eye_corners
    eye_center = (left_corner + right_corner) / 2
    eye_width = np.linalg.norm(right_corner - left_corner)

    # Distance of iris center from eye center (normalized)
    deviation = np.linalg.norm(iris_center - eye_center) / eye_width
 
    return deviation < 0.15  # Tunable threshold for eye contact

def get_average_point(landmarks, indices, shape):
    h, w = shape
    points = np.array([np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices])
    return np.mean(points, axis=0)

# OpenCV capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        # Get corners and iris centers
        left_eye = np.array([landmarks[i] for i in LEFT_EYE_INDICES])
        right_eye = np.array([landmarks[i] for i in RIGHT_EYE_INDICES])
        left_iris_center = get_average_point(landmarks, LEFT_IRIS_INDICES, (h, w))
        right_iris_center = get_average_point(landmarks, RIGHT_IRIS_INDICES, (h, w))

        left_eye_corners = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in LEFT_EYE_INDICES]
        right_eye_corners = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in RIGHT_EYE_INDICES]

        # Eye contact logic
        left_eye_contact = compute_eye_contact(left_iris_center, left_eye_corners)
        right_eye_contact = compute_eye_contact(right_iris_center, right_eye_corners)

        overall_eye_contact = left_eye_contact and right_eye_contact

        # Draw iris and corners
        for pt in left_eye_corners + right_eye_corners:
            cv2.circle(frame, tuple(np.int32(pt)), 2, (0, 255, 255), -1)
        cv2.circle(frame, tuple(np.int32(left_iris_center)), 3, (0, 0, 255), -1)
        cv2.circle(frame, tuple(np.int32(right_iris_center)), 3, (0, 0, 255), -1)

        # Display result
        text = "Looking at Camera" if overall_eye_contact else "Not Looking"
        color = (0, 255, 0) if overall_eye_contact else (0, 0, 255)
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow('Eye Contact Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
