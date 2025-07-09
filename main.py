from fastapi import FastAPI, UploadFile, File, BackgroundTasks
import shutil
import os
from helper_functions import run_video_analysis, normalize_filename
import json
from fastapi.responses import JSONResponse
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
app = FastAPI()
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



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)