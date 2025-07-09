from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import gradio as gr

# Download YOLOv8 face detection model from Hugging Face
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# Ensure output folder exists
os.makedirs("face_crops", exist_ok=True)

def detect_faces(image_rgb):
    if image_rgb is None:
        return "❌ No image uploaded.", None, []

    # Convert RGB (Gradio input) to PIL and OpenCV BGR
    image_pil = Image.fromarray(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Run face detection
    results = model(image_pil)[0]
    face_images = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        # Draw box and label
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, f"{conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop face and save
        face_crop = image_bgr[y1:y2, x1:x2]
        crop_path = f"face_crops/face_{i+1}.jpg"
        cv2.imwrite(crop_path, face_crop)

        # Add to gallery (convert to RGB for display)
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_images.append(face_rgb)

        print(f"✅ Saved: {crop_path}")

    # Convert back to RGB for display
    annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    status = f"✅ {len(results.boxes)} face(s) detected."
    return status, annotated_rgb, face_images


# Gradio Interface
demo = gr.Interface(
    fn=detect_faces,
    inputs=gr.Image(type="numpy", label="Upload Person Image"),
    outputs=[
        gr.Textbox(label="Status"),
        gr.Image(type="numpy", label="Annotated Image"),
        gr.Gallery(label="Detected Faces", columns=4)
    ],
    title="YOLOv8 Face Detection (Hugging Face)",
    description="Upload a cropped person image to detect and extract faces using YOLOv8.",
    allow_flagging="never"
)

demo.launch()
