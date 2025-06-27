import cv2
import mediapipe as mp
import numpy as np




class PoseDetector:
    def __init__(self, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_conf,
            min_tracking_confidence=min_tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_connections = self.mp_pose.POSE_CONNECTIONS

    def _to_pixel_coords(self, landmark, image_w, image_h):
        return np.array([landmark.x * image_w, landmark.y * image_h])

    def _shoulder_tilt_angle(self, l_shoulder, r_shoulder):
        dx = r_shoulder[0] - l_shoulder[0]
        dy = r_shoulder[1] - l_shoulder[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))

        # Normalize to [-90, 90]
        if angle_deg < -90:
            angle_deg += 180
        elif angle_deg > 90:
            angle_deg -= 180

        return angle_deg
 # in degrees

    def _head_pitch_angle(self, nose, l_shoulder, r_shoulder):
        # Compute shoulder midpoint
        shoulder_center = (np.array(l_shoulder) + np.array(r_shoulder)) / 2
        # Vector from shoulder center to nose
        dx = nose[0] - shoulder_center[0]
        dy = shoulder_center[1] - nose[1]  # flip y because top is 0 in image coords
        # Compute pitch angle
        angle_deg = np.degrees(np.arctan2(dy, dx))
        return angle_deg

    def _posture_score(self, shoulder_angle):
        score = 1.0
        # If shoulders are tilted more than ±10°, reduce score
        if abs(shoulder_angle) > 7:
            score -= 0.4  # or adjust penalty as you like
        return min(max(score, 0.0), 1.0)

    def _head_score(self, head_angle):
        score = 1.0
        if 80 <= head_angle <= 95:
            score += 0.2
        else:
            score -= 0.4
        return min(max(score, 0.0), 1.0)

    def _confidence_score(self, shoulder_angle, head_angle):
        ps = self._posture_score(shoulder_angle)
        hs = self._head_score(head_angle)
        return round((0.5 * ps + 0.5 * hs), 2)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)

        posture_metrics = {
            "shoulder_angle": None,
            "head_angle": None,
            "posture_score": None,
            "head_score": None,
            "confidence_score": None
        }

        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_connections,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            lm = results.pose_landmarks.landmark

            try:
                nose = self._to_pixel_coords(lm[0], w, h)
                l_shoulder = self._to_pixel_coords(lm[11], w, h)
                r_shoulder = self._to_pixel_coords(lm[12], w, h)

                shoulder_angle = self._shoulder_tilt_angle(l_shoulder, r_shoulder)
                
                head_angle = self._head_pitch_angle(nose, l_shoulder, r_shoulder)

                posture_score = self._posture_score(shoulder_angle)
                head_score = self._head_score(head_angle)
                conf_score = self._confidence_score(shoulder_angle, head_angle)

                posture_metrics = {
                    "shoulder_angle": round(shoulder_angle, 2),
                    "head_angle": round(head_angle, 2),
                    "posture_score": round(posture_score, 2),
                    "head_score": round(head_score, 2),
                    "confidence_score": round(conf_score, 2)
                }

            except Exception as e:
                print("Error computing posture metrics:", e)

        return posture_metrics
detector = PoseDetector()
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    metrics = detector.process_frame(frame)

    if metrics["confidence_score"] is not None:
        cv2.putText(frame, f"Confidence Score: {metrics['confidence_score']:.2f}",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Shoulder Tilt: {metrics['shoulder_angle']:.2f}°",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Head Angle: {metrics['head_angle']:.2f}°",
                    (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)

    cv2.imshow("Confidence Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
