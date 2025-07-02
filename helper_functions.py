import cv2
import time
import statistics
import os
import json
import re
from Module1.Media_Pipe__Service import MediaPipeService
def run_video_analysis(path: str):
    result = process_video(path)
    print("Analysis complete.")
    # You can store to file or DB here if needed
    output_path = path.replace(".mp4", "_results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)


def normalize_filename(filename):
        # Remove emojis and non-ASCII characters
        filename = filename.encode('ascii', 'ignore').decode('ascii')
        # Remove special characters except dot, underscore, and dash
        filename = re.sub(r'[^\w\.-]', '', filename)
        # Remove spaces
        filename = filename.replace(' ', '')
        return filename
def process_video(video_path: str):
    """
    Process the uploaded video file, compute visual scores, overlay stats, 
    save processed video, and store results in a JSON file.
    """

    # Setup filenames
    video_name = os.path.basename(video_path)                   # e.g., "test.mp4"
    video_base = os.path.splitext(video_name)[0]                # e.g., "test"
    processed_video_name = f"{video_base}_analyzed.mp4"         # ✅ New processed video
    processed_video_path = os.path.join("processed", processed_video_name)
    result_json_path = os.path.join("uploaded_videos", f"{video_base}_results.json")

    # Prepare video reader and writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(
        processed_video_path,
        fourcc,
        fps,
        (width, height)
    )

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

        # ======= DETECTION & COMPUTATION =======
        mp_service.process_face(frame)
        mp_service.process_pose(frame)

        simile_active, smile_score = 0, 0
        eye_contact = 0

        if mp_service.face_results and mp_service.face_results.multi_face_landmarks:
            for face_landmarks in mp_service.face_results.multi_face_landmarks:
                simile_active, smile_score = mp_service.detect_smile(
                    face_landmarks.landmark,
                    frame.shape[1], frame.shape[0], frame
                )
                eye_contact, *_ = mp_service.detect_eye_contact(frame, face_landmarks.landmark)

        pose_metrics = mp_service.pose_detection(frame)
        head_pose_metrics = mp_service.process_head_pose(frame)

        pose_confidence = pose_metrics["confidence_score"] or 0.0
        confidence_score = min(max(pose_confidence, 0.0), 1.0)

        final_score, norm_smile_score, smile_active_score, confidence_score, head_pose_score, eye_contact = mp_service.calculate_final_score(
            smile_score, simile_active, confidence_score, head_pose_metrics["head_pose_text"], eye_contact
        )

        # Log scores every new second
        current_time = time.time() - start_time
        current_second = int(current_time)

        if current_second != last_logged_second:
            last_logged_second = current_second
            timestamps.append(round(current_time, 2))
            final_scores.append(final_score)
            smile_scores.append(norm_smile_score)
            smile_active_scores.append(smile_active_score)
            confidence_scores.append(confidence_score)
            head_pose_scores.append(head_pose_score)
            eye_contact_scores.append(eye_contact)

        mp_service.print_smile_status(frame, simile_active, smile_score)
        mp_service.draw_eye_contact_status(frame, eye_contact)
        mp_service.print_pose_stats(frame, pose_metrics)
        mp_service.print_head_pose_stats(frame, head_pose_metrics)
        mp_service.print_final_score(frame, final_score)


        # ======= DRAW & SAVE FRAME =======
        mp_service.draw_live_bars(frame, {
            "Final": final_score,
            "Smile": norm_smile_score,
            "Active Smile": smile_active_score,
            "Posture Score": confidence_score,
            "Head Straight": head_pose_score,
            "Eye Contact": eye_contact
        }, origin=(30, 300))
        
        out.write(frame)

    cap.release()
    out.release()

    # ======= SAVE METRICS =======
    result = {
        "timestamps": timestamps,
        "final_scores": final_scores,
        "smile_scores": smile_scores,
        "smile_active_scores": smile_active_scores,
        "confidence_scores": confidence_scores,
        "head_pose_scores": head_pose_scores,
        "eye_contact_scores": eye_contact_scores,
        "summary": {
            "avg_confidence": statistics.mean(confidence_scores) if confidence_scores else 0,
            "duration_sec": timestamps[-1] if timestamps else 0
        },
        "processed_video_path": processed_video_path
    }

    with open(result_json_path, "w") as f:
        json.dump(result, f)

    return result
