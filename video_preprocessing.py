import cv2
import numpy as np
from Media_Pipe_Service import MediaPipeService

# ========== Configuration ==========
MIN_WIDTH = 640
MIN_HEIGHT = 480
BRIGHTNESS_THRESHOLD = 50       # lower = too dark
SHARPNESS_THRESHOLD = 100       # lower = blurry
MIN_VALID_FRAMES = 5
OUTPUT_PATH = "cleaned_output.mp4"
# ===================================

# Initialize MediaPipe service
mediapipe_service = MediaPipeService()

# ========== Helper Functions ==========

def is_resolution_valid(frame):
    h, w = frame.shape[:2]
    return w >= MIN_WIDTH and h >= MIN_HEIGHT

def is_bright_enough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = cv2.mean(gray)[0]
    return brightness > BRIGHTNESS_THRESHOLD

def is_sharp_enough(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness > SHARPNESS_THRESHOLD

def has_face(frame):
    # Use the MediaPipe service for face detection
    return mediapipe_service.has_face(frame)

def is_frame_valid(frame):
    return (is_resolution_valid(frame) and
            is_bright_enough(frame) and
            is_sharp_enough(frame) and
            has_face(frame))

# ========== Main Processing ==========

def clean_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    valid_frames = []
    total_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        if is_frame_valid(frame):
            valid_frames.append(frame)

    cap.release()
    return valid_frames, fps, total_frames

def save_cleaned_video(frames, output_path, fps):
    if not frames:
        print("❌ No valid frames to save.")
        return
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"✅ Cleaned video saved to: {output_path}")

# ========== Run Script ==========

if __name__ == "__main__":
    input_path = "video.mp4"  # CHANGE THIS TO YOUR VIDEO PATH

    print(f"🧼 Cleaning video: {input_path}")
    valid_frames, fps, total = clean_video_frames(input_path)

    print(f"✅ Valid frames: {len(valid_frames)} / {total}")
    if len(valid_frames) < MIN_VALID_FRAMES:
        print("⚠️ Not enough valid frames. Please upload a better quality video.")
    else:
        save_cleaned_video(valid_frames, OUTPUT_PATH, fps)
    
    # Clean up MediaPipe service
    mediapipe_service.cleanup()
