import cv2
import mediapipe as mp
import numpy as np

class MediaPipeService:
    def __init__(self, max_faces=1, static_image_mode=False, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_faces,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def process_frame(self, frame):
        """Process a single frame and return face landmarks if detected."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.multi_face_landmarks[0],
                self.mp_face_mesh.FACEMESH_CONTOURS,
                self.drawing_spec,
                self.drawing_spec
            )
            return results.multi_face_landmarks  # list of landmarks
        else:
            return None

    def detect_smile(self, landmarks, image_width, image_height, threshold=0.05):
        """
        Detects a smile using 4 key mouth landmarks.
        Returns (is_smiling: bool, mar: float)
        """
        def get_point(idx):
            lm = landmarks[idx]
            return np.array([lm.x * image_width, lm.y * image_height])

        left_corner = get_point(61)
        right_corner = get_point(291)
        upper_lip = get_point(13)
        lower_lip = get_point(14)

        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        vertical_dist = np.linalg.norm(upper_lip - lower_lip)

        mar = vertical_dist / horizontal_dist if horizontal_dist > 0 else 0

        is_smiling = mar > threshold
        return is_smiling, mar
    



import cv2

mp_service = MediaPipeService()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    landmarks_list = mp_service.process_frame(frame)

    if landmarks_list:
        face_landmarks = landmarks_list[0]
        is_smiling, mar = mp_service.detect_smile(
            face_landmarks.landmark,
            image_width=frame.shape[1],
            image_height=frame.shape[0]
        )

        label = "Smiling 😀" if is_smiling else "Not Smiling 😐"
        color = (0, 255, 0) if is_smiling else (0, 0, 255)
        cv2.putText(frame, f"{label} (MAR: {mar:.2f})", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Smile Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
