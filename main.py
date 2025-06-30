import cv2
import time
import matplotlib.pyplot as plt
from Media_Pipe_Service import MediaPipeService

def print_smile_status(frame, simile_active, smile_score, y_start):
    status = f"Smiling 😄 (Score: {smile_score:.2f})" if simile_active else f"Not Smiling 😐 (Score: {smile_score:.2f})"
    color = (0, 255, 0) if simile_active else (0, 0, 255)
    cv2.putText(frame, status, (30, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def print_pose_stats(frame, pose_metrics, y_start, line_height):
    if pose_metrics["confidence_score"] is not None:
        cv2.putText(frame, f"Confidence Score: {pose_metrics['confidence_score']:.2f}",
                    (30, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Shoulder Tilt: {pose_metrics['shoulder_angle']:.2f}°",
                    (30, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Head Angle: {pose_metrics['head_angle']:.2f}°",
                    (30, y_start + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 100), 2)

def print_head_pose_stats(frame, head_pose_metrics, y_start, line_height):
    cv2.putText(frame, head_pose_metrics["head_pose_text"], (400, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    x_angle = f"{head_pose_metrics['x_angle']:.2f}" if head_pose_metrics['x_angle'] is not None else "--"
    y_angle = f"{head_pose_metrics['y_angle']:.2f}" if head_pose_metrics['y_angle'] is not None else "--"
    z_angle = f"{head_pose_metrics['z_angle']:.2f}" if head_pose_metrics['z_angle'] is not None else "--"
    cv2.putText(frame, f"x: {x_angle}", (500, y_start + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"y: {y_angle}", (500, y_start + 2 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(frame, f"z: {z_angle}", (500, y_start + 3 * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def print_final_score(frame, final_score, y_start):
    cv2.putText(frame, f"Final Score: {final_score:.2f}", (30, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

cap = cv2.VideoCapture(0)
mp_service = MediaPipeService()

cv2.namedWindow("MediaPipe Output", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("MediaPipe Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

LINE_HEIGHT = 30
Y_START_SMILE = 40
Y_START_POSE = 120
Y_START_HEAD = 240
timestamps = []
final_scores = []
smile_scores = []
smile_active_scores = []
confidence_scores = []
head_pose_scores = []
start_time = time.time()
last_logged_second = -1
while True:
    ret, frame = cap.read()
    if not ret:
        break
    mp_service.process_face(frame)
    mp_service.process_pose(frame)
    if mp_service.face_results and mp_service.face_results.multi_face_landmarks:
        for face_landmarks in mp_service.face_results.multi_face_landmarks:
            simile_active,smile_score=mp_service.detect_smile(face_landmarks.landmark, frame.shape[1], frame.shape[0],frame)
        print_smile_status(frame, simile_active, smile_score, Y_START_SMILE)
    pose_metrics=mp_service.pose_detection(frame)    
    print_pose_stats(frame, pose_metrics, Y_START_POSE, LINE_HEIGHT)
    head_pose_metrics = mp_service.process_head_pose(frame)
    print_head_pose_stats(frame, head_pose_metrics, Y_START_HEAD, LINE_HEIGHT)
    final_score,norm_smile_score,smile_active_score,confidence_score,head_pose_score= mp_service.calculate_final_score(smile_score, simile_active, pose_metrics["confidence_score"], head_pose_metrics["head_pose_text"])
    print_final_score(frame, final_score, Y_START_HEAD + 4 * LINE_HEIGHT)
    mp_service.draw_pose_landmarks(frame)
    mp_service.draw_face_landmarks(frame)
    cv2.imshow("MediaPipe Output", frame)

    current_time = time.time() - start_time
    current_second = int(current_time)

    if current_second != last_logged_second:
        last_logged_second = current_second  # Update tracker
        final_score, norm_smile_score, smile_active_score, confidence_score, head_pose_score = mp_service.calculate_final_score(
            smile_score, simile_active, pose_metrics["confidence_score"], head_pose_metrics["head_pose_text"]
        )
        timestamps.append(current_second)
        final_scores.append(final_score)
        smile_scores.append(norm_smile_score)
        smile_active_scores.append(smile_active_score)
        confidence_scores.append(confidence_score)
        head_pose_scores.append(head_pose_score)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# Plotting the scores
plt.figure(figsize=(12, 6))
plt.plot(timestamps, final_scores, label='Final Score', color='blue')
plt.legend()
plt.xlabel('Time (seconds)')
plt.ylabel('Score')
plt.title('Scores Over Time')
plt.show()

