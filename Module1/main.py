import cv2
import time
import statistics
import matplotlib.pyplot as plt
from testing_media_pipe import MediaPipeService
# Initialize video capture and MediaPipe service
cap = cv2.VideoCapture(0)
mp_service = MediaPipeService()
# Score trackers
timestamps = []
final_scores = []
smile_scores = []
smile_active_scores = []
confidence_scores = []
head_pose_scores = []
eye_contact_scores = []
start_time = time.time()
last_logged_second = -1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ==================== ✅ STAGE 1: DETECTION & COMPUTATION ====================

    # Process face and pose
    mp_service.process_face(frame)
    mp_service.process_pose(frame)

    simile_active, smile_score = 0, 0
    eye_contact = 0
    # Face landmark-based detections
    if mp_service.face_results and mp_service.face_results.multi_face_landmarks:
        for face_landmarks in mp_service.face_results.multi_face_landmarks:
            simile_active, smile_score = mp_service.detect_smile(
                face_landmarks.landmark,
                frame.shape[1],
                frame.shape[0],
                frame
            )
            eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center = mp_service.detect_eye_contact(
                frame, face_landmarks.landmark
            )

    # Pose and head pose metrics
    pose_metrics = mp_service.pose_detection(frame)
   
    head_pose_metrics = mp_service.process_head_pose(frame)
    print(head_pose_metrics)
    # Normalize posture confidence
    pose_confidence = pose_metrics["confidence_score"] or 0.0
    confidence_score = min(max(pose_confidence, 0.0), 1.0)

    # Final combined score calculation
    final_score, norm_smile_score, smile_active_score, confidence_score, head_pose_score, eye_contact = mp_service.calculate_final_score(
        smile_score, simile_active, confidence_score, head_pose_metrics["head_pose_text"], eye_contact
    )
    
    # Logging scores per second
    current_time = time.time() - start_time
    current_second = (current_time)

    if current_second != last_logged_second:
        last_logged_second = current_second
        timestamps.append(current_second)
        final_scores.append(final_score)
        smile_scores.append(norm_smile_score)
        smile_active_scores.append(smile_active_score)
        confidence_scores.append(confidence_score)
        head_pose_scores.append(head_pose_score)
        eye_contact_scores.append(eye_contact)

    # ==================== 🎨 STAGE 2: DRAWING & VISUALIZATION ====================
    # ==================== 🎨 STAGE 2: DRAWING & VISUALIZATION ====================

    # Draw face mesh and pose skeleton
    mp_service.draw_face_landmarks(frame)
    mp_service.draw_pose_landmarks(frame)

    # Print status on frame
    mp_service.print_smile_status(frame, simile_active, smile_score)
    # Print status on frame
    mp_service.print_smile_status(frame, simile_active, smile_score)
    mp_service.draw_eye_contact_status(frame, eye_contact)
    mp_service.print_pose_stats(frame, pose_metrics)
    mp_service.print_head_pose_stats(frame, head_pose_metrics)
    mp_service.print_final_score(frame, final_score)
  
    # Live bar drawing
    mp_service.draw_live_bars(frame, {
        "Final": (final_score),
        "Smile": smile_score,
        "Active Smile": smile_active_score,
        "Posture Score": confidence_score,
        "Head Straight": head_pose_score,
        "Eye Contact": eye_contact
    }, origin=(30, 300))

 
    # Show final frame
    cv2.imshow("MediaPipe Output", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Cleanup
cap.release()
cv2.destroyAllWindows()

# ==================== 📊 FINAL PLOT ====================

average_confidence = (statistics.mean(final_scores))
total_duration = timestamps[-1] if timestamps else 0

plt.figure(figsize=(12, 6))
plt.plot(timestamps, final_scores, label='Final Score', color='blue')
plt.axhline(y=average_confidence, color='gray', linestyle='--', label=f'Avg: {average_confidence:.2f}')
plt.text(timestamps[-1], average_confidence + 0.01, f'Avg: {average_confidence:.2f}', fontsize=10, color='gray')
plt.text(0, (min(final_scores)) - 0.05, f'Total Time: {total_duration:.2f} sec', fontsize=10, color='black')

plt.title('Final Score Over Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Final Score')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()
