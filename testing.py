import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Draw utilities (optional)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Landmark indices
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
LEFT_EYE = 33
LEFT_CHEEK = 205
RIGHT_EYE = 263
RIGHT_CHEEK = 425
LEFT_FACE = 234
RIGHT_FACE = 454

# Globals for smile tracking
count = 0
real_smile_active = False
smile_buffer = 0

# Distance helper
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

# Smile detection with buffering
def detect_smile(landmarks, image_w, image_h):
    global count, real_smile_active, smile_buffer

    # Get pixel coordinates
    def get_point(index):
        lm = landmarks[index]
        return int(lm.x * image_w), int(lm.y * image_h)

    # Normalization by face width
    face_width = euclidean(get_point(LEFT_FACE), get_point(RIGHT_FACE)) + 1e-6

    # Normalized features
    mouth_spread = euclidean(get_point(LEFT_MOUTH), get_point(RIGHT_MOUTH)) / face_width
    lip_opening = euclidean(get_point(UPPER_LIP), get_point(LOWER_LIP)) / face_width
    eye_openness_left = euclidean(get_point(LEFT_EYE_TOP), get_point(LEFT_EYE_BOTTOM)) / face_width
    eye_openness_right = euclidean(get_point(RIGHT_EYE_TOP), get_point(RIGHT_EYE_BOTTOM)) / face_width
    cheek_raise_left = euclidean(get_point(LEFT_EYE), get_point(LEFT_CHEEK)) / face_width
    cheek_raise_right = euclidean(get_point(RIGHT_EYE), get_point(RIGHT_CHEEK)) / face_width
    # Compute smile score
    smile_score = 0
    if mouth_spread > 0.35:
        smile_score += 1
        if lip_opening > 0.03:
            smile_score += 0.5
        if eye_openness_left < 0.058 and eye_openness_right < 0.058:
            smile_score += 1
        if 0.26 <= cheek_raise_left <= 0.32 and 0.26 <= cheek_raise_right <= 0.32:
            smile_score += 1

    smiling = smile_score >= 1

    # Frame-based smoothing
    ACTIVATION_THRESHOLD = 5      # min frames to activate smile
    DEACTIVATION_BUFFER = 10      # frames to keep smile after disappearance

    if smiling:
        count += 1
        smile_buffer = DEACTIVATION_BUFFER  # refresh
        if count >= ACTIVATION_THRESHOLD:
            real_smile_active = True
    else:
        count = 0
        smile_buffer -= 1
        if smile_buffer <= 0:
            real_smile_active = False

    return real_smile_active, smile_score


# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            smiling, score = detect_smile(face_landmarks.landmark, w, h)

            # Draw mesh
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec,
            )

            # Draw smile status
            status = f"Smiling 😄 (Score: {score:.2f})" if smiling else f"Not Smiling 😐 (Score: {score:.2f})"
            color = (0, 255, 0) if smiling else (0, 0, 255)
            cv2.putText(frame, status, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow('Smile Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
