from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import shutil
import os
from helper_functions import run_video_analysis, normalize_filename
import json
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import cv2
import base64
import tempfile
from Module2.main import run_pipeline_from_image
from Module2.gradio_complete import generate_group_summary,load_cropped_images
from Module2.Media_Pipe__Service import MediaPipeService
app = FastAPI()
API_KEY = "AIzaSyA-gsPJ5tLoF7Ok5-83Llab6U5oWI3Xe5E"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
UPLOAD_DIR = "uploaded_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)
def get_indexed_filename(base_name: str, ext: str, directory: str) -> str:
    index = 0
    while True:
        candidate = f"{base_name}_{index}{ext}"
        if not os.path.exists(os.path.join(directory, candidate)):
            return candidate
        index += 1
@app.post("/upload/")
async def upload_video(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    original_name, ext = os.path.splitext(normalize_filename(file.filename))

    unique_filename = get_indexed_filename(original_name, ext, UPLOAD_DIR)
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Background task
    background_tasks.add_task(run_video_analysis, file_path)

    # Return base filename (no extension)
    return {"status": "processing started", "filename": os.path.splitext(unique_filename)[0]}

@app.get("/status/{video_id}")
async def get_status(video_id: str):
    file_path = os.path.join(UPLOAD_DIR, f"{video_id}_results.json")
    print(file_path)
    if not os.path.exists(file_path):
        print("not found")
        return {"status": "not found"}
    print("found")
    return {"status": "done"}

@app.get("/results/{filename}")
async def get_results(filename: str):
    if not filename.endswith(".json"):
        filename += "_results.json"
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Results file not found")
    with open(file_path, "r") as f:
        data = json.load(f)
    return JSONResponse(content=data)



@app.get("/video/{filename}")
async def get_video(filename: str):
    # Convert uploaded filename (e.g., video.mp4) to analyzed filename (video_analyzed.mp4)
    if filename.endswith(".mp4"):
        analyzed_name = filename.replace(".mp4", "_analyzed.mp4")
    else:
        analyzed_name = filename + "_analyzed.mp4"

    file_path = os.path.join("processed", analyzed_name)
    print("Serving video file:", file_path)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(path=file_path, media_type="video/mp4", filename=analyzed_name)
def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')

@app.post("/analyze_photo/")
async def analyze_photo(file: UploadFile = File(...)):
    try:
        # Step 1: Read uploaded image
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        input_image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.jpg")
            cv2.imwrite(input_path, cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))

            run_pipeline_from_image(input_path)

            people_folder = "cropped_people"
            faces_folder = "cropped_faces"
            detected = load_cropped_images(people_folder, faces_folder)

            mp_service = MediaPipeService()
            json_results = []

            for person in detected:
                idx = person["index"]
                person_img = person["person_img"]
                face_img = person["face_img"]
                person_file = person["person_file"]

                if person_img is None:
                    continue

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

            rgb_input = input_image
            group_summary = generate_group_summary(json_results, rgb_input)

            image_b64 = image_to_base64(rgb_input)
            
            return {
                "image_base64": image_b64,
                "group_summary": group_summary
            }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)