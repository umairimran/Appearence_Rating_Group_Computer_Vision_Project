import cv2
import numpy as np
import tempfile
import json
import os
import shutil
from dotenv import load_dotenv
from Media_Pipe__Service import MediaPipeService
from gradio_complete import generate_group_summary, load_cropped_images
from main import run_pipeline_from_image

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

def clean_and_create_results_folder():
    """Delete existing results folder and create a fresh one"""
    results_folder = "results"
    if os.path.exists(results_folder):
        shutil.rmtree(results_folder)
        print(f"🗑️ Deleted existing results folder")
    os.makedirs(results_folder)
    print(f"📁 Created fresh results folder: {results_folder}")
    return results_folder

def analyze_group_collective(image_path):
    """Analyze the group collectively - EXACTLY like FastAPI"""
    
    print("🔍 Starting group collective analysis...")
    results_folder = clean_and_create_results_folder()
    
    try:
        # Step 1: Read image (like FastAPI)
        print("📸 Reading image...")
        input_image = cv2.imread(image_path)
        if input_image is None:
            raise Exception("Could not read image file")
        
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        print("✅ Image loaded successfully")
        
        # Step 2: Pipeline execution (like FastAPI)
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jpg")
            cv2.imwrite(input_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
            
            print("🔄 Running person detection pipeline...")
            run_pipeline_from_image(input_path)
            print("✅ Pipeline completed")
            
            # Step 3: Load results (like FastAPI)
            people_folder = "cropped_people"
            faces_folder = "cropped_faces"
            detected = load_cropped_images(people_folder, faces_folder)
            print(f"✅ Found {len(detected)} people")
            
            # Step 4: Process each person (EXACTLY like FastAPI)
            mp_service = MediaPipeService()
            json_results = []
            
            for i, person in enumerate(detected):
                print(f"👤 Processing person {i+1}/{len(detected)}...")
                
                idx = person["index"]
                person_img = person["person_img"]
                face_img = person["face_img"]
                person_file = person["person_file"]

                if person_img is None:
                    print(f"⚠️ Skipping person {idx} - no image data")
                    continue

                try:
                    if face_img is not None:
                        mp_service.process_face(face_img)
                    mp_service.process_pose(person_img)

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
                            eye_contact, *_ = mp_service.detect_eye_contact(
                                face_img, face_landmarks.landmark
                            )

                    pose_metrics = mp_service.pose_detection(person_img)
                    head_pose_score, head_pose_text, dist_left, dist_right = mp_service.process_head_pose(
                        face_img if face_img is not None else person_img
                    )

                    pose_confidence = pose_metrics.get("confidence_score", 0.0) or 0.0
                    confidence_score = min(max(pose_confidence, 0.0), 1.0)

                    final_score, norm_smile_score, *_ = mp_service.calculate_final_score(
                        smile_score, smile_active, confidence_score, head_pose_score, eye_contact
                    )

                    cloth_colors = mp_service.extract_dress_colors(person_img, API_KEY)

                    # EXACTLY like FastAPI - no data cleaning
                    json_results.append({
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
                    })
                    
                except Exception as e:
                    print(f"❌ Error processing person {idx}: {str(e)}")
                    continue

            # Step 5: Generate group summary (EXACTLY like FastAPI)
            print("📊 Generating group summary...")
            
            # Clean the data to remove None values before group summary
            cleaned_json_results = []
            for person in json_results:
                cleaned_person = {}
                for key, value in person.items():
                    if value is None:
                        cleaned_person[key] = 0  # Replace None with 0
                    else:
                        cleaned_person[key] = value
                cleaned_json_results.append(cleaned_person)

            rgb_input = input_image
            group_summary = generate_group_summary(cleaned_json_results, rgb_input)
            
            # Save group summary
            group_summary_filename = os.path.join(results_folder, "group_summary.json")
            with open(group_summary_filename, "w") as f:
                json.dump(group_summary, f, indent=2)
            print(f"✅ Saved {group_summary_filename}")
            
            return {"success": True, "group_summary": group_summary}
            
    except Exception as e:
        import traceback
        error_msg = f"Group analysis failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"❌ Full traceback: {traceback.format_exc()}")
        return {"error": error_msg}

def main():
    print("🧑‍🤝‍🧑 Group Photo Analysis Tool")
    print("=" * 40)
    
    image_path = input("Enter the path to your group photo: ").strip()
    
    if not os.path.exists(image_path):
        print("❌ Error: Image file not found!")
        return
    
    print(f"\n Analyzing image: {image_path}")
    print("⏳ This may take a few moments...")
    
    print("\n🔍 Analyzing group collectively...")
    result = analyze_group_collective(image_path)
    
    if result.get("success"):
        print("✅ Group analysis complete!")
    else:
        print(f"❌ Error: {result.get('error')}")
    
    print(f"\n Check the 'results' folder for JSON result files!")
    print("🎉 Analysis complete!")

if __name__ == "__main__":
    main()