import cv2
import mediapipe as mp
import cv2
import mediapipe as mp
import numpy as np
import  time


class MediaPipeService:
    def __init__(self):
        self.count = 0
        self.real_smile_active = False
        self.score=0.0
        self.smile_count = 0
        self.smile_buffer = 0
        self.smile_active = False
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_drawing_spec = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1) 
        self.pose_drawing_spec = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_faces=1,
        
        )
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_results = None
        self.pose_results = None
    
    def process_face(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.face_results = self.face_mesh.process(rgb_image)
        return self.face_results
    
    def process_pose(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.pose_results = self.pose.process(rgb_image)
        return self.pose_results
    
    def draw_face_landmarks(self, image):
        if self.face_results and self.face_results.multi_face_landmarks:
            for face_landmarks in self.face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=self.face_drawing_spec,
                    connection_drawing_spec=self.face_drawing_spec
                )
    
    def draw_pose_landmarks(self, image):
        if self.pose_results and self.pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                image=image,
                landmark_list=self.pose_results.pose_landmarks,
                connections=self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.pose_drawing_spec,
                connection_drawing_spec=self.pose_drawing_spec
            )
 
    def detect_smile(self, landmarks, image_w, image_h,frame):
        
        def get_point(index):
            lm = landmarks[index]
            return (int(lm.x * image_w), int(lm.y * image_h))

        # Key landmarks
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

        face_width = self.euclidean(get_point(LEFT_FACE), get_point(RIGHT_FACE)) + 1e-6
        mouth_spread = self.euclidean(get_point(LEFT_MOUTH), get_point(RIGHT_MOUTH)) / face_width
        lip_opening = self.euclidean(get_point(UPPER_LIP), get_point(LOWER_LIP)) / face_width
        eye_openness_left = self.euclidean(get_point(LEFT_EYE_TOP), get_point(LEFT_EYE_BOTTOM)) / face_width
        eye_openness_right = self.euclidean(get_point(RIGHT_EYE_TOP), get_point(RIGHT_EYE_BOTTOM)) / face_width
        cheek_raise_left = self.euclidean(get_point(LEFT_EYE), get_point(LEFT_CHEEK)) / face_width
        cheek_raise_right = self.euclidean(get_point(RIGHT_EYE), get_point(RIGHT_CHEEK)) / face_width
        lip_corner_left_y = get_point(LEFT_MOUTH)[1]
        lip_corner_right_y = get_point(RIGHT_MOUTH)[1]
        upper_lip_y = get_point(UPPER_LIP)[1]

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
            self.count += 1
            self.smile_buffer = DEACTIVATION_BUFFER  # refresh
            if self.count >= ACTIVATION_THRESHOLD:
                self.real_smile_active = True
        else:
            self.count = 0
            self.smile_buffer -= 1
            if self.smile_buffer <= 0:
                self.real_smile_active = False

        return self.real_smile_active, smile_score
    
    def euclidean(self, p1, p2):
         return np.linalg.norm(np.array(p1) - np.array(p2))
    
    def _to_pixel_coords(self, landmark, image_w, image_h):
        return np.array([landmark.x * image_w, landmark.y * image_h])
   
    def _shoulder_tilt_angle(self, l_shoulder, r_shoulder):
        dx = r_shoulder[0] - l_shoulder[0]
        dy = r_shoulder[1] - l_shoulder[1]
        angle_deg = np.degrees(np.arctan2(dy, dx))
        print("Left shoulder:",dy)
        print("Right Shoulder:",dx)
        if angle_deg < -90:
            angle_deg += 180
        elif angle_deg > 90:
            angle_deg -= 180
        return angle_deg
    
    def _head_pitch_angle(self, nose, l_shoulder, r_shoulder):
        shoulder_center = (np.array(l_shoulder) + np.array(r_shoulder)) / 2
        # Vector from shoulder center to nose
        dx = nose[0] - shoulder_center[0]
        dy = shoulder_center[1] - nose[1]  # flip y because top is 0 in image coords
        # Compute pitch angle
        angle_deg = np.degrees(np.arctan2(dy, dx))
        return angle_deg

    def _posture_score(self, shoulder_angle):
        print(f"[Posture] shoulder_angle: {shoulder_angle}")
        angle = abs(shoulder_angle)
        if angle <= 15:
            return 0.85  # even perfect posture gets max 0.85, not 1.0
        elif angle <= 20:
            return round(1.0 - (angle - 15) * 0.02, 2)  # subtracts 0.02 per degree between 15-20
        else:
            return 0.60  # for bad posture (angle > 20), harsh penalty
    def _head_score(self, head_angle):
        print("Head Angle:,",head_angle)
        angle = abs(head_angle)
        if 75 <= angle <= 100:
            return 0.95  
        elif 60 <= angle < 75 or 100 < angle <= 115:
            return 0.85  
        else:
            return 0.75  
    def _confidence_score(self, shoulder_angle, head_angle, posture_weight=0.5, head_weight=0.5):
        # Get raw scores (0 to 1)
        ps = self._posture_score(shoulder_angle)  # e.g., 0.6
        hs = self._head_score(head_angle)         # e.g., 0.8

        # Combine using weighted average
        confidence = (ps * posture_weight + hs * head_weight) / (posture_weight + head_weight)

        # Clamp result between 0 and 1
        return min(max(confidence, 0.0), 1.0)

    def pose_detection(self,frame):
        POSE_LANDMARKS = {
            0: "nose", 1: "left_eye_inner", 2: "left_eye", 3: "left_eye_outer",
            4: "right_eye_inner", 5: "right_eye", 6: "right_eye_outer",
            7: "left_ear", 8: "right_ear", 9: "mouth_left", 10: "mouth_right",
            11: "left_shoulder", 12: "right_shoulder", 13: "left_elbow", 14: "right_elbow",
            15: "left_wrist", 16: "right_wrist", 17: "left_pinky", 18: "right_pinky",
            19: "left_index", 20: "right_index", 21: "left_thumb", 22: "right_thumb",
            23: "left_hip", 24: "right_hip", 25: "left_knee", 26: "right_knee",
            27: "left_ankle", 28: "right_ankle", 29: "left_heel", 30: "right_heel",
            31: "left_foot_index", 32: "right_foot_index"
        }
        posture_metrics={
            "shoulder_angle":None,
            "head_angle":None,
            "posture_score":None,
            "head_score":None,
            "confidence_score":None
        }
        h,w,_ = frame.shape
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image.flags.writeable=False
        self.pose_results=self.pose.process(image)
        if self.pose_results.pose_landmarks:
            lm=self.pose_results.pose_landmarks.landmark
        else:
            lm=None
        try:
            nose=self._to_pixel_coords(lm[0],w,h)
            l_shoulder=self._to_pixel_coords(lm[11],w,h)
            r_shoulder=self._to_pixel_coords(lm[12],w,h)
            shoulder_angle=self._shoulder_tilt_angle(l_shoulder,r_shoulder)
            head_angle=self._head_pitch_angle(nose,l_shoulder,r_shoulder)
            posture_score=self._posture_score(shoulder_angle)
            head_score=self._head_score(head_angle)
            posture_metrics={
                "shoulder_angle":round(shoulder_angle,2),
                "head_angle":round(head_angle,2),
                "posture_score":round(posture_score,2),
                "head_score":round(head_score,2),
                "confidence_score":round(self._confidence_score(shoulder_angle,head_angle),2)
            }
        except Exception as e:
            print("Error computing posture metrics:",e)


        return posture_metrics
   
    def draw_group(self,landmarks, indices):
        xs = []
        for idx in indices:
            lm = landmarks[idx]
            xs.append(lm.x)  # Normalized x (0–1)
        return sum(xs) / len(xs)
    
    def determine_horizontal_direction(self,avg_left_x, avg_right_x, avg_center_x, tolerance):
        dist_to_left = avg_center_x - avg_left_x    # signed
        dist_to_right = avg_right_x - avg_center_x  # signed
        
        # Extra hint logic: if one of the sides has crossed the nose center
        if dist_to_right < 0:
            return "Looking Right (Crossover)", dist_to_left, dist_to_right
        elif dist_to_left < 0:
            return "Looking Left (Crossover)", dist_to_left, dist_to_right

        # Standard logic when all landmarks are in proper order
        if dist_to_left > 0 and dist_to_right > 0:
            # Center is between left and right → determine based on closeness
            if dist_to_right < tolerance:
                return "Looking Right", dist_to_left, dist_to_right
            elif dist_to_left < tolerance:
                return "Looking Left", dist_to_left, dist_to_right
            else:
                return "Looking Forward", dist_to_left, dist_to_right
        else:
            return "Uncertain", dist_to_left, dist_to_right
   
    def process_head_pose(self, frame):
        LEFT_INDICES = [127, 234, 93, 132, 58]
        RIGHT_INDICES = [356, 454, 323, 361, 288]
        CENTER_INDICES = [1, 168, 13, 152]


        results = self.process_face(frame)
        

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            avg_left_x = self.draw_group(landmarks, LEFT_INDICES)
            avg_right_x = self.draw_group(landmarks, RIGHT_INDICES)
            avg_center_x = self.draw_group(landmarks, CENTER_INDICES)

            horizontal_direction, dist_left, dist_right = self.determine_horizontal_direction(
                avg_left_x, avg_right_x, avg_center_x, 0.03
            )

            if horizontal_direction=="Looking Forward":
                score=0.90
            else:
                score=0.5

            return score, horizontal_direction,dist_left,dist_right
        else:
            # Fallback if no face detected
            return 0.0, "No face detected",0,0
     
    def calculate_final_score(self, smile_score, simile_active, confidence_score, head_pose_score, eye_contact):
        # Normalize smile_score (max 3.5 based on your smile detection logic)
        norm_smile_score = min(smile_score / 1.5, 1.0)

        # Convert simile_active (bool) to 1.0 or 0.0
        smile_active_score = 1.0 if simile_active else 0.0

        # Pose confidence is already normalized (assume it's one of 0.6, 0.8, 1.0)
        confidence_score = min(max(confidence_score, 0.0), 1.0)

        # Head pose score: 1.0 only if looking straight
        
        eye_contact_score = 1.0 if eye_contact else 0.0
        # Weighted final score calculation
        final_score = (
            0.25 * norm_smile_score +
            0.25 * confidence_score +
            0.25 * head_pose_score +
            0.25 * eye_contact_score
        )

        return round(final_score, 2), norm_smile_score, smile_active_score, confidence_score, head_pose_score, eye_contact_score

    def print_smile_status(self,frame, simile_active, smile_score):
        h, w = frame.shape[:2]
        margin_x = int(w * 0.01)  # 1% from left
        margin_y = int(h * 0.04)  # 4% from top
        font_scale = max(0.2, min(0.4, h / 1000.0))  # Small font, scales with height
        status = f"Smiling (Score: {smile_score:.2f})" if simile_active else f"Not Smiling (Score: {smile_score:.2f})"
        color = (0, 255, 0) if simile_active else (0, 0, 255)
        cv2.putText(frame, status, (margin_x, margin_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    def print_pose_stats(self,frame, pose_metrics):
        h, w = frame.shape[:2]
        margin_x = int(w * 0.01)
        margin_y = int(h * 0.04)
        font_scale = max(0.2, min(0.4, h / 1000.0))
        line_height = int(h * 0.042)
        if pose_metrics["confidence_score"] is not None:
            score = pose_metrics["confidence_score"]
            if score==1.0:
                color = (0, 255, 0)
            elif score >= 0.8:
                color = (255, 0, 0)  # Blue
            elif score <= 0.6:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            cv2.putText(frame, f"Posture Score: {score:.2f}",
                        (margin_x, margin_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            # Shoulder Tilt and Head Angle lines are commented out as per your last change
            # cv2.putText(frame, f"Shoulder Tilt: {pose_metrics['shoulder_angle']:.2f}°",
            #            (margin_x, margin_y + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0), 1)
            # cv2.putText(frame, f"Head Angle: {pose_metrics['head_angle']:.2f}°",
            #            (margin_x, margin_y + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 100), 1)

    def print_head_pose_stats(self,frame, head_pose_metrics):
        h, w = frame.shape[:2]
        margin_x = int(w * 0.01)
        margin_y = int(h * 0.01)
        font_scale = max(0.2, min(0.4, h / 1000.0))
        line_height = int(h * 0.045)
        # Color logic: green if straight, red otherwise
        is_straight = head_pose_metrics["head_pose_text"] == "Looking Straight"
        label_color = (0, 255, 0) if is_straight else (0, 0, 255)
        angle_color = (0, 0, 255) if not is_straight else (255, 0, 0)
        # Head pose text below pose stats
        cv2.putText(frame, head_pose_metrics["head_pose_text"], (margin_x, margin_y + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_color, 1)
        x_angle = f"{head_pose_metrics['x_angle']:.2f}" if head_pose_metrics['x_angle'] is not None else "--"
        y_angle = f"{head_pose_metrics['y_angle']:.2f}" if head_pose_metrics['y_angle'] is not None else "--"
        z_angle = f"{head_pose_metrics['z_angle']:.2f}" if head_pose_metrics['z_angle'] is not None else "--"
        #cv2.putText(frame, f"x: {x_angle}", (margin_x, margin_y + 5 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, angle_color, 1)
        #cv2.putText(frame, f"y: {y_angle}", (margin_x, margin_y + 6 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, angle_color, 1)
        #cv2.putText(frame, f"z: {z_angle}", (margin_x, margin_y + 7 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, angle_color, 1)

    def print_final_score(self,frame, final_score):
        h, w = frame.shape[:2]
        margin_x = int(w * 0.01)
        margin_y = int(h * 0.01)
        font_scale = max(0.2, min(0.4, h / 1000.0))
        line_height = int(h * 0.045)
        # Place final score below head pose stats
        cv2.putText(frame, f"Final Score: {final_score:.2f}", (margin_x, margin_y + 5 * line_height), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 1), 1)

    def get_average_point(self, landmarks, indices, shape):
        h, w = shape
        points = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        return np.mean(points, axis=0)

    def compute_eye_aspect_ratio(self, eye_indices, landmarks, shape):
        h, w = shape
        eye = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in eye_indices]
        vertical = np.linalg.norm(eye[0] - eye[1])     # upper to lower
        horizontal = np.linalg.norm(eye[2] - eye[3])   # left to right
        return vertical / (horizontal + 1e-6)

    def compute_eye_contact(self, iris_center, eye_corners):
        left_corner, right_corner = eye_corners
        eye_center = (left_corner + right_corner) / 2
        eye_width = np.linalg.norm(right_corner - left_corner)

        dx = abs(iris_center[0] - eye_center[0]) / eye_width
        dy = abs(iris_center[1] - eye_center[1]) / eye_width

        # Iris distance ratio (left vs right corner)
        dist_left = np.linalg.norm(iris_center - left_corner)
        dist_right = np.linalg.norm(iris_center - right_corner)
        ratio = dist_left / (dist_right + 1e-6)  # avoid division by zero

        print(f"[EyeContact] dx: {dx:.3f}, dy: {dy:.3f}, ratio (L/R): {ratio:.2f}")

        return dx < 0.2 and dy < 0.3, dx, dy
    # You can adjust this threshold

    def compute_ear(self, upper_ids, lower_ids, corner_ids, landmarks, image_shape):
        h, w = image_shape
        upper = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in upper_ids])
        lower = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in lower_ids])
        verticals = np.linalg.norm(upper - lower, axis=1)
        vertical_avg = np.mean(verticals)

        left_corner = np.array([landmarks[corner_ids[0]].x * w, landmarks[corner_ids[0]].y * h])
        right_corner = np.array([landmarks[corner_ids[1]].x * w, landmarks[corner_ids[1]].y * h])
        horizontal = np.linalg.norm(left_corner - right_corner)

        return vertical_avg / horizontal if horizontal != 0 else 0

    def detect_eye_contact(self, frame, landmarks):
        # Landmark indices
        LEFT_EYE_UPPER = [159, 160, 158, 157, 173]
        LEFT_EYE_LOWER = [145, 144, 153, 154, 155]
        LEFT_EYE_CORNERS = [33, 133]

        RIGHT_EYE_UPPER = [386, 387, 385, 384, 398]
        RIGHT_EYE_LOWER = [374, 373, 380, 381, 382]
        RIGHT_EYE_CORNERS = [362, 263]

        LEFT_IRIS_INDICES = [468, 469, 470, 471]
        RIGHT_IRIS_INDICES = [473, 474, 475, 476]

        EYE_CLOSED_THRESHOLD = 0.15
        h, w, _ = frame.shape

        # Check for closed eyes
        left_ear = self.compute_ear(LEFT_EYE_UPPER, LEFT_EYE_LOWER, LEFT_EYE_CORNERS, landmarks, (h, w))
        right_ear = self.compute_ear(RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_EYE_CORNERS, landmarks, (h, w))
        ear_avg = (left_ear + right_ear) / 2
        print(f"[EyeContact] ear_avg: {ear_avg}")
        if ear_avg < EYE_CLOSED_THRESHOLD:
            
            print("Eyes Closed")
            return False, None, None, None, None

        # Get iris centers
        left_iris_center = self.get_average_point(landmarks, LEFT_IRIS_INDICES, (h, w))
        right_iris_center = self.get_average_point(landmarks, RIGHT_IRIS_INDICES, (h, w))

        # Get eye corners
        left_eye_corners = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in LEFT_EYE_CORNERS]
        right_eye_corners = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in RIGHT_EYE_CORNERS]

        # Eye contact logic
        left_eye_contact,dx_left,dy_left = self.compute_eye_contact(left_iris_center, left_eye_corners)
        right_eye_contact,dx_right,dy_right = self.compute_eye_contact(right_iris_center, right_eye_corners)
        overall_eye_contact = left_eye_contact and right_eye_contact
        print(f"[EyeContact] overall: {overall_eye_contact}, left: {left_eye_contact} (dx={dx_left:.3f}, dy={dy_left:.3f}), right: {right_eye_contact} (dx={dx_right:.3f}, dy={dy_right:.3f})")
        return overall_eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center
    def draw_eye_contact_info(self,frame, eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center):
        for pt in left_eye_corners + right_eye_corners:
            cv2.circle(frame, tuple(np.int32(pt)), 2, (0, 255, 255), -1)
        cv2.circle(frame, tuple(np.int32(left_iris_center)), 3, (0, 0, 255), -1)
        cv2.circle(frame, tuple(np.int32(right_iris_center)), 3, (0, 0, 255), -1)

        text = "Looking at Camera" if eye_contact else "Not Looking"
        color = (0, 255, 0) if eye_contact else (0, 0, 255)
        cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    def draw_eye_contact_status(self, frame, eye_contact_result, line_offset=2):
        """
        Dynamically prints eye contact status on the frame with adjustable vertical position.
        `line_offset`: how many lines down to push the text (each line ≈ 5% screen height)
        """
        h, w = frame.shape[:2]
        margin_x = int(w * 0.01)
        margin_y = int(h * 0.02)
        font_scale = max(0.2, min(0.4, h / 1000.0))
        line_height = int(h * 0.05)
        # Status text and color
        if eye_contact_result == "Eyes Closed":
            text = "Eyes Closed"
            color = (255, 0, 0)  # Blue
        elif eye_contact_result == True:
            text = "Looking at Camera"
            color = (0, 255, 0)  
        else:
            text = "Not Looking"
            color = (0, 0, 255)  
        # Dynamic Y-position based on line offset
        y_position = margin_y + line_height * 5
        # Draw text on frame
        cv2.putText(frame, f"Eye Contact: {text}",
                    (margin_x, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
    def draw_live_bars(self,frame, scores_dict, origin=None, bar_width=None, bar_height=None, spacing=None):
        """
        Draws horizontal score bars on the frame, with dynamic sizing and positioning based on frame size.
        """
        h, w = frame.shape[:2]
        if bar_width is None:
            bar_width = int(w * 0.25)
        if bar_height is None:
            bar_height = int(h * 0.03)
        if spacing is None:
            spacing = int(h * 0.05)
        if origin is None:
            x, y = int(w * 0.02), int(h * 0.4)
        else:
            x, y = origin

        num_bars = len(scores_dict)
        total_height = num_bars * bar_height + (num_bars - 1) * spacing
        # Adjust y if bars would overflow
        if y + total_height > h:
            y = max(int(h - total_height - h * 0.02), int(h * 0.02))

        for label, value in scores_dict.items():
            value = max(0.0, min(1.0, value))
            cv2.rectangle(frame, (x, y), (x + bar_width, y + bar_height), (60, 60, 60), -1)
            bar_color = (0, 255, 0) if value >= 0.7 else (0, 165, 255) if value >= 0.4 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + int(bar_width * value), y + bar_height), bar_color, -1)
            cv2.putText(frame, f"{label}: {value:.2f}", (x + bar_width + int(w*0.01), y + bar_height - int(h*0.005)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.4, min(0.7, h / 1000.0)), (13, 94, 166), 1)
            y += spacing
    def draw_single_line_graph(self,frame, values, label, origin=None, size=None, color=(0, 255, 0)):
        """
        Draws a single line graph on the frame at the given origin, with dynamic sizing.
        """
        h, w = frame.shape[:2]
        if origin is None:
            x0, y0 = int(w * 0.7), int(h * 0.05)
        else:
            x0, y0 = origin
        if size is None:
            graph_w, graph_h = int(w * 0.15), int(h * 0.12)
        else:
            graph_w, graph_h = size
        # Draw background
        cv2.rectangle(frame, (x0, y0), (x0 + graph_w, y0 + graph_h), (30, 30, 30), -1)
        # Draw the graph line
        series = values[-graph_w:]
        if len(series) >= 2:
            points = []
            for i, v in enumerate(series):
                x = x0 + i
                y = int(y0 + (1.0 - v) * graph_h)  # Inverted Y
                points.append((x, y))
            for i in range(1, len(points)):
                cv2.line(frame, points[i - 1], points[i], color, 1)
        # Draw label
        cv2.putText(frame, label, (x0 + 5, y0 + int(graph_h * 0.25)), cv2.FONT_HERSHEY_SIMPLEX, max(0.4, min(0.7, h / 1000.0)), color, 1)
    def draw_labeled_line_graph(self,frame, values, label, origin=None, size=None, color=(0, 255, 0)):
        """
        Draws a clean line graph with X/Y axis and labels directly on the OpenCV frame, with dynamic sizing.
        """
        h, w = frame.shape[:2]
        if origin is None:
            x0, y0 = int(w * 0.7), int(h * 0.05)
        else:
            x0, y0 = origin
        if size is None:
            graph_w, graph_h = int(w * 0.15), int(h * 0.15)
        else:
            graph_w, graph_h = size
        # Y-axis (0 to 1 score line)
        cv2.line(frame, (x0, y0), (x0, y0 + graph_h), (200, 200, 200), 1)
        # X-axis (time)
        cv2.line(frame, (x0, y0 + graph_h), (x0 + graph_w, y0 + graph_h), (200, 200, 200), 1)
        # Y-axis labels
        for i in range(0, 11):  # 0.0 to 1.0 with step of 0.1
            y_val = i / 10.0
            y = int(y0 + graph_h - y_val * graph_h)
            label_y = f"{y_val:.1f}"
            cv2.putText(frame, label_y, (x0 - int(w*0.025), y + int(h*0.005)), cv2.FONT_HERSHEY_PLAIN, max(0.5, min(0.8, h / 800.0)), (150, 150, 150), 1)
        # X-axis ticks (up to graph_w values)
        max_points = graph_w - 10
        step = max(1, len(values) // max_points)
        series = values[-max_points::step]
        # Draw the line
        if len(series) >= 2:
            for i in range(1, len(series)):
                x1 = x0 + 10 + (i - 1)
                y1 = int(y0 + graph_h - series[i - 1] * graph_h)
                x2 = x0 + 10 + i
                y2 = int(y0 + graph_h - series[i] * graph_h)
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)
        # Label
        cv2.putText(frame, label, (x0 + 5, y0 - int(h*0.01)), cv2.FONT_HERSHEY_SIMPLEX, max(0.5, min(0.8, h / 800.0)), color, 1)



    import os
    import time
    import cv2
    import numpy as np

    def save_eye_contact_visual(self, frame, eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center, save_dir="eye_contact_outputs"):
        os.makedirs(save_dir, exist_ok=True)

        # Create a copy of the frame to draw on
        annotated = frame.copy()

        # Draw keypoints
        self.draw_eye_contact_info(annotated, eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center)

        # Base filename
        base_name = "eye_contact"
        index = 1
        filename = f"{base_name}.jpg"
        filepath = os.path.join(save_dir, filename)

        # If file exists, add suffix number
        while os.path.exists(filepath):
            filename = f"{base_name}_{index}.jpg"
            filepath = os.path.join(save_dir, filename)
            index += 1

        # Save image
        cv2.imwrite(filepath, annotated)
        print(f"[EyeContact] Saved visualization at: {filepath}")

