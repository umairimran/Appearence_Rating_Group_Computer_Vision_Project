import gradio as gr
import cv2
import numpy as np
import os
import tempfile
import json
from dotenv import load_dotenv
from Media_Pipe__Service import MediaPipeService
from main import run_pipeline_from_image
load_dotenv()
API_KEY = "AIzaSyA-gsPJ5tLoF7Ok5-83Llab6U5oWI3Xe5E"


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Convert hex to RGB tuple
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Compute color similarity using cosine similarity between RGB vectors
def color_similarity_score(color_list):
    if len(color_list) <= 1:
        return 100
    rgb_vectors = [np.array(hex_to_rgb(color)) for color in color_list]
    sims = []
    for i in range(len(rgb_vectors)):
        for j in range(i+1, len(rgb_vectors)):
            sim = cosine_similarity([rgb_vectors[i]], [rgb_vectors[j]])[0][0]
            sims.append(sim)
    avg_sim = np.mean(sims)
    return int(avg_sim * 100)

# Main group summary generation function
def generate_group_summary(json_results, image):
    print(json_results)
    if not json_results:
        return {}

    num_people = len(json_results)
    smile_scores = []
    eye_contacts = 0
    posture_scores = []
    final_scores = []
    confidences = []
    head_pose_scores = []
    top_colors = []

    for person in json_results:
        # Smile score
        smile_score = person.get("smile_score", 0)
        smile_scores.append(smile_score)

        # Eye contact
        if person.get("eye_contact", 0) == 1:
            eye_contacts += 1

        # Scores
        posture_scores.append(person.get("pose_stats", {}).get("posture_score", 0))
        final_scores.append(person.get("final_score", 0))
        confidences.append(person.get("confidence_score", 0))
        head_pose_scores.append(person.get("head_pose_score", 0))

        # Colors
        top_hex = person.get("cloth_colors", {}).get("top_color", {}).get("hex", "#ffffff")
        top_colors.append(top_hex)

    # Averages
    avg_posture = np.mean(posture_scores) if posture_scores else 0
    avg_final_score = np.mean(final_scores) if final_scores else 0
    avg_confidence = np.mean(confidences) if confidences else 0
    avg_head_pose = np.mean(head_pose_scores) if head_pose_scores else 0
    avg_smile = np.mean(smile_scores) if smile_scores else 0
    eye_contact_percent = int((eye_contacts / num_people) * 100)
    smile_percent = int(avg_smile * 100)
    color_sim = color_similarity_score(top_colors)

    # Group synergy score (average-based estimate)
    synergy_score = int((avg_posture + avg_confidence + avg_smile + (eye_contact_percent / 100) + (color_sim / 100)) / 5 * 100)

    # Overall aesthetic score (simplified average)
    overall_aesthetics = int((avg_final_score + avg_confidence + avg_posture + color_sim / 100) / 4 * 100)

    # Human-readable summary
    mp_service = MediaPipeService()
    summary_text = mp_service.generate_gemini_summary(
        avg_posture,
        eye_contact_percent,
        smile_percent,
        color_sim,
        synergy_score,
        API_KEY,
        image
    )

    return {
        "group_synergy_score": synergy_score,
        "color_similarity": color_sim,
        "posture_scores": int(avg_posture * 100),
        "eye_contact": eye_contact_percent,
        "active_smiles": smile_percent,
        "overall_aesthetics": overall_aesthetics,
        "summary": summary_text
    }

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
            cloth_colors = mp_service.extract_dress_colors(person_img,API_KEY)
            result_json = {
                "person_index": idx,
                "person_file": person_file,
                "final_score": final_score,
                "smile_score": norm_smile_score,
                
                "confidence_score": confidence_score,
                "head_pose_score": head_pose_score,
                "eye_contact": eye_contact,
                "pose_stats": pose_metrics,
                "head_pose_text": head_pose_text,
                "dist_left": dist_left,
                "dist_right": dist_right,
                "cloth_colors": cloth_colors
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
        group_summary = generate_group_summary(json_results,rgb_input)
        print("Group Summary Here Printing")
        print(group_summary)
        # Return all required data
        summary_value = group_summary.get("summary", "")
        if isinstance(summary_value, dict):
            summary_value = summary_value.get("summary") or summary_value.get("error") or str(summary_value)
        elif not isinstance(summary_value, str):
            summary_value = str(summary_value)
        return [rgb_input, person_images, face_images, json_results, json_results, summary_value, group_summary]

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
        with gr.Column(scale=1):
            gr.Markdown("## Group Summary (JSON)")
            group_summary_json = gr.JSON(label="📊 Group Summary (Full)")
        with gr.Column(scale=1):
            gr.Markdown("## Group Summary")
            group_summary = gr.Markdown(label="📊 Group Summary")
    # Set up event handlers
    analyze_outputs = [
        output_image,    # Original image
        people_gallery,  # Gallery of detected people
        state_faces,     # State for faces (hidden)
        state_jsons,     # State for JSON results (hidden)
        all_json,        # Display all results
        group_summary,   # Display group summary (Markdown)
        group_summary_json, # Display group summary (JSON)
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
