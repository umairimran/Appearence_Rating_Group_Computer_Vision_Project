import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import shutil

# Load models once
PERSON_MODEL = YOLO("yolov8n.pt")
FACE_MODEL_PATH = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
FACE_MODEL = YOLO(FACE_MODEL_PATH)

def convert_cv2_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_faces(cv2_image, min_conf=0.4):
    pil_image = Image.fromarray(convert_cv2_to_rgb(cv2_image))
    results = FACE_MODEL(pil_image)[0]
    faces = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        faces.append((x1, y1, x2, y2))
    return faces

def face_inside_person(face_box, person_box):
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = person_box
    return px1 <= fx1 and py1 <= fy1 and px2 >= fx2 and py2 >= fy2

def crop_people_with_faces(folder_path, save_dir="peoplefor_color_analysis"):
    """
    Processes all images in the folder, detects persons with faces, and saves cropped results.
    """
    print(f"[INFO] Starting processing for folder: {folder_path}")
    allowed_exts = {'.jpg', '.jpeg', '.png'}

    # Cleanup previous results
    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                   if os.path.splitext(f)[1].lower() in allowed_exts]

    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Skipped unreadable image: {img_path}")
            continue

        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # Detect
        person_results = PERSON_MODEL(image)
        person_boxes = [box for box in person_results[0].boxes if int(box.cls[0]) == 0]
        face_boxes = detect_faces(image)

        used_faces = set()
        crop_index = 0

        for i, face_box in enumerate(face_boxes):
            for j, person_box_data in enumerate(person_boxes):
                x1, y1, x2, y2 = map(int, person_box_data.xyxy[0])
                person_box = (x1, y1, x2, y2)
                if face_inside_person(face_box, person_box) and i not in used_faces:
                    used_faces.add(i)
                    # Crop and save
                    cropped = image[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                    out_name = f"{base_filename}_{crop_index}.jpg"
                    cv2.imwrite(os.path.join(save_dir, out_name), cropped)
                    crop_index += 1
                    break  # Only assign each face once

    print(f"[DONE] All valid cropped people saved to: {save_dir}")
