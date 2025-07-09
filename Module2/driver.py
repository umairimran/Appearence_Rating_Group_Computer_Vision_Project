import cv2
import os
import json
import sys
import time

from Media_Pipe__Service import MediaPipeService
from main import run_pipeline_from_image


# Initialize the MediaPipeService
mp_service = MediaPipeService()

# Run pipeline with path relative to script location
image_path = "groupPhotos/image3.jpg"
run_pipeline_from_image(image_path)
#print("pipe line is done runned")

#time.sleep(10)
# Define paths relative to script location
people_folder = "cropped_people"
faces_folder = "cropped_faces"
output_folder = "scored_people"
os.makedirs(output_folder, exist_ok=True)

# List files
image_files = sorted(os.listdir(people_folder))

for index, person_file in enumerate(image_files):
    person_path = os.path.join(people_folder, person_file)
    face_path = os.path.join(faces_folder, os.path.splitext(person_file)[0] + "_bestface.jpg")

    if not os.path.exists(face_path):
        print(f"❌ Face not found for {person_file}, skipping...")
        continue

    # Read images
    frame = cv2.imread(person_path)
    face_frame = cv2.imread(face_path)

    if frame is None or face_frame is None:
        print(f"❌ Failed to load image pair {person_file}")
        continue

    # ========== STAGE 1: DETECTION & COMPUTATION ==========

    # Face and pose processing
    mp_service.process_face(face_frame)
    mp_service.process_pose(frame)

    # Face-based metrics
    simile_active, smile_score = 0, 0
    eye_contact = 0
    if mp_service.face_results and mp_service.face_results.multi_face_landmarks:
        for face_landmarks in mp_service.face_results.multi_face_landmarks:
            simile_active, smile_score = mp_service.detect_smile(
                face_landmarks.landmark,
                face_frame.shape[1],
                face_frame.shape[0],
                face_frame
            )
            eye_contact, _, _, _, _ = mp_service.detect_eye_contact(
                face_frame, face_landmarks.landmark
            )

    # Pose and head pose metrics
    pose_metrics = mp_service.pose_detection(frame)
    head_pose_score,head_pose_text = mp_service.process_head_pose(face_frame)  ## willl return 0.5 for left right and 1 for straight

    # Normalize posture confidence
    pose_confidence = pose_metrics["confidence_score"] or 0.0
    confidence_score = min(max(pose_confidence, 0.0), 1.0)

    # Final score calculation
    final_score, norm_smile_score, smile_active_score, confidence_score, head_pose_score, eye_contact = mp_service.calculate_final_score(
        smile_score, simile_active, confidence_score, head_pose_score, eye_contact
    )
    
        # ========== STAGE 2: OUTPUT ==========

        # Store result in JSON per person
    result = {
            "person_file": person_file,
            "final_score": final_score,
            "smile_score": norm_smile_score,
            "smile_active": smile_active_score,
            "confidence_score": confidence_score,
            "head_pose_score": head_pose_score,
            "eye_contact": eye_contact,
            "pose_stats": pose_metrics,
            "head_pose_text": head_pose_text
        }

    json_path = os.path.join(output_folder, person_file.replace(".jpg", ".json"))
    with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

    print(f"✅ Scored and saved: {json_path}")
