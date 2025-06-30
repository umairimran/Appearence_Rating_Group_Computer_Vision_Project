import cv2
import time
import matplotlib.pyplot as plt
from Media_Pipe__Service import MediaPipeService



cap = cv2.VideoCapture("video2.mp4")


mp_service = MediaPipeService()

#cv2.namedWindow("MediaPipe Output", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("MediaPipe Output", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
def draw_live_bars(frame, scores_dict, origin=None, bar_width=None, bar_height=None, spacing=None):
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
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, min(0.7, h / 1000.0)), (255, 255, 255), 1)
        y += spacing
def draw_single_line_graph(frame, values, label, origin=None, size=None, color=(0, 255, 0)):
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
def draw_labeled_line_graph(frame, values, label, origin=None, size=None, color=(0, 255, 0)):
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
        mp_service.print_smile_status(frame, simile_active, smile_score)
        eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center = mp_service.detect_eye_contact(frame, face_landmarks.landmark)
        #mp_service.draw_eye_contact_info(frame, eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center)
    pose_metrics=mp_service.pose_detection(frame)    
    mp_service.print_pose_stats(frame, pose_metrics)
    head_pose_metrics = mp_service.process_head_pose(frame)
    mp_service.print_head_pose_stats(frame, head_pose_metrics)
    if pose_metrics["confidence_score"] is None:
        pose_confidence = 0.0
    else:
        pose_confidence = pose_metrics["confidence_score"]
    confidence_score = min(max(pose_confidence, 0.0), 1.0)
    final_score,norm_smile_score,smile_active_score,confidence_score,head_pose_score= mp_service.calculate_final_score(smile_score, simile_active, confidence_score, head_pose_metrics["head_pose_text"])
    mp_service.print_final_score(frame, final_score)
    draw_live_bars(frame, {
        "Final": final_score,
        "Smile": norm_smile_score,
        "Active Smile": smile_active_score,
        "Posture Score": confidence_score,
        "Head Straight": head_pose_score
    }, origin=(30, 300))
# Graph dimensions and positions
    graph_width = 140
    graph_height = 60
    gap = 10

    # Top-right corner anchors
    frame_h, frame_w = frame.shape[:2]
    x_base = frame_w - graph_width - 20
    y_base = 20

    # Draw each graph separately
    draw_labeled_line_graph(frame, final_scores, "Final", (x_base, y_base), size=(graph_width, graph_height), color=(0, 255, 255))
    draw_labeled_line_graph(frame, smile_scores, "Smile", (x_base, y_base + graph_height + gap), size=(graph_width, graph_height), color=(0, 255, 0))
    draw_labeled_line_graph(frame, confidence_scores, "Posture", (x_base, y_base + 2 * (graph_height + gap)), size=(graph_width, graph_height), color=(255, 0, 0))

    mp_service.draw_pose_landmarks(frame)
    mp_service.draw_face_landmarks(frame)
    cv2.imshow("MediaPipe Output", frame)

    current_time = time.time() - start_time
    current_second = int(current_time)

    if current_second != last_logged_second:
        last_logged_second = current_second  # Update tracker
        final_score, norm_smile_score, smile_active_score, confidence_score, head_pose_score = mp_service.calculate_final_score(
            smile_score, simile_active, confidence_score, head_pose_metrics["head_pose_text"]
        )
        timestamps.append(current_second)
        final_scores.append(final_score)
        smile_scores.append(norm_smile_score)
        smile_active_scores.append(smile_active_score)
        confidence_scores.append(confidence_score)
        head_pose_scores.append(head_pose_score)
    # Draw real-time score bars on screen
  # Lower this if it overlaps with text

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
# Plotting the scores

#plt.figure(figsize=(14, 10))

#plt.subplot(5, 1, 1)
#  plt.plot(timestamps, final_scores, label='Final Score', color='blue')
#plt.ylabel('Final Score')
#plt.legend(loc='upper right')
#plt.grid(True)

#plt.subplot(5, 1, 2)
#plt.plot(timestamps, smile_scores, label='Smile Score', color='green')
#plt.ylabel('Smile Score')
#plt.legend(loc='upper right')
#plt.grid(True)

#plt.subplot(5, 1, 3)
#plt.plot(timestamps, smile_active_scores, label='Active Smile', color='orange')
#plt.ylabel('Active Smile')
#plt.legend(loc='upper right')
#plt.grid(True)

#plt.subplot(5, 1, 4)
#plt.plot(timestamps, confidence_scores, label='Posture Score', color='purple')
#plt.ylabel('Posture Score')
#plt.legend(loc='upper right')
#plt.grid(True)

#plt.subplot(5, 1, 5)
#plt.plot(timestamps, head_pose_scores, label='Head Pose Score', color='red')
#plt.xlabel('Time (seconds)')
#plt.ylabel('Head Pose Score')
#plt.legend(loc='upper right')
#plt.grid(True)

#plt.suptitle('Scores Over Time', fontsize=18)
#plt.tight_layout(rect=[0, 0.03, 1, 0.97])
#plt.show()

