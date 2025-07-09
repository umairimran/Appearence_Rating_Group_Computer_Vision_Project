import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import json

from Media_Pipe__Service import MediaPipeService
from main import run_pipeline_from_image

# Helper: Convert cv2 image to RGB for Gradio
def cv2_to_rgb(img):
    if img is None:
        return None
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Helper: Load all cropped people and faces after pipeline
def load_cropped_images(people_folder, faces_folder):
    people_files = sorted([
        f for f in os.listdir(people_folder)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    results = []
    for idx, person_file in enumerate(people_files):
        person_path = os.path.join(people_folder, person_file)
        face_path = os.path.join(faces_folder, os.path.splitext(person_file)[0] + "_bestface.jpg")
        person_img = cv2.imread(person_path)
        face_img = cv2.imread(face_path) if os.path.exists(face_path) else None
        results.append({
            "index": idx + 1,
            "person_file": person_file,
            "person_img": person_img,
            "face_img": face_img,
            "person_path": person_path,
            "face_path": face_path if os.path.exists(face_path) else None
        })
    return results

# Main Gradio pipeline function
def analyze_group_photo(input_image):
    # Save uploaded image to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.jpg")
        cv2.imwrite(input_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

        # Run pipeline (creates cropped_people/ and cropped_faces/)
        run_pipeline_from_image(input_path)

        # Folders (output folders)
        people_folder = "cropped_people"
        faces_folder = "cropped_faces"

        # Load all detected people and faces
        detected = load_cropped_images(people_folder, faces_folder)

        # Initialize MediaPipeService
        mp_service = MediaPipeService()
        print("I am here")

        # Prepare outputs
        person_images = []
        face_images = []
        json_results = []

        for person in detected:
            idx = person["index"]
            person_img = person["person_img"]
            face_img = person["face_img"]
            person_file = person["person_file"]

            if person_img is None:
                continue
            # Process face and pose
            if face_img is not None:
                mp_service.process_face(face_img)
            mp_service.process_pose(person_img)
            # Face-based metrics
            smile_active, smile_score = 0, 0
            eye_contact = 0
            if mp_service.face_results and mp_service.face_results.multi_face_landmarks and face_img is not None:
                for face_landmarks in mp_service.face_results.multi_face_landmarks:
                    smile_active, smile_score = mp_service.detect_smile(
                        face_landmarks.landmark,
                        face_img.shape[1],
                        face_img.shape[0],
                        face_img
                    )
                    eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center = mp_service.detect_eye_contact(
                        face_img, face_landmarks.landmark
                    )
            # Pose and head pose metrics
            pose_metrics = mp_service.pose_detection(person_img)
            head_pose_score,head_pose_text,dist_left,dist_right = mp_service.process_head_pose(face_img if face_img is not None else person_img)
            
            pose_confidence = pose_metrics.get("confidence_score", 0.0) or 0.0
            confidence_score = min(max(pose_confidence, 0.0), 1.0)

            final_score, norm_smile_score, smile_active_score, confidence_score, head_pose_score, eye_contact = mp_service.calculate_final_score(
                smile_score, smile_active, confidence_score, head_pose_score, eye_contact
            )

            result_json = {
                "person_index": idx,
                "person_file": person_file,
                "final_score": final_score,
                "smile_score": norm_smile_score,
                "smile_active": smile_active_score,
                "confidence_score": confidence_score,
                "head_pose_score": head_pose_score,
                "eye_contact": eye_contact,
                "pose_stats": pose_metrics,
                "head_pose_text": head_pose_text,
                "dist_left": dist_left,
                "dist_right": dist_right
            }
            json_results.append(result_json)

            # Add to our image lists
            person_images.append(cv2_to_rgb(person_img))
            if face_img is not None:
                face_images.append(cv2_to_rgb(face_img))
            else:
                face_images.append(None)

        # Convert the input image to RGB for Gradio
        rgb_input = cv2_to_rgb(input_image)
        
        # Return all required data
        return [rgb_input, person_images, face_images, json_results, json_results]

def show_selected_person(evt: gr.SelectData, faces, jsons):
    """Handle person selection in the gallery"""
    if not faces or not jsons or evt.index >= len(faces) or evt.index >= len(jsons):
        return None, None
    return faces[evt.index], jsons[evt.index]

# ===================== GRADIO APP ===========================
with gr.Blocks() as demo:
    gr.Markdown("# 🧑‍🤝‍🧑 Group Photo Analyzer")
    gr.Markdown("Upload a group photo. The app will detect all people, crop their images and faces, and show scores.")

    # Store state for faces and results
    state_faces = gr.State([])
    state_jsons = gr.State([])

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="Upload Group Photo")
            analyze_btn = gr.Button("🔍 Analyze")
        with gr.Column():
            output_image = gr.Image(label="📷 Original Image")

    gr.Markdown("## Detected People")
    people_gallery = gr.Gallery(label="People", show_label=True)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Selected Person Details")
            selected_face = gr.Image(label="Selected Person's Face")
            selected_json = gr.JSON(label="Selected Person's Results")
        with gr.Column(scale=1):
            gr.Markdown("## All Results")
            all_json = gr.JSON(label="📊 All Analysis Results")

    # Set up event handlers
    analyze_outputs = [
        output_image,    # Original image
        people_gallery,  # Gallery of detected people
        state_faces,     # State for faces (hidden)
        state_jsons,     # State for JSON results (hidden)
        all_json,        # Display all results
    ]

    analyze_btn.click(
        fn=analyze_group_photo,
        inputs=[input_image],
        outputs=analyze_outputs
    )

    # When a person is selected in the gallery
    people_gallery.select(
        fn=show_selected_person,
        inputs=[state_faces, state_jsons],  # Pass the stored faces and results
        outputs=[selected_face, selected_json]  # Update the selected person's details
    )

if __name__ == "__main__":
    demo.launch()
