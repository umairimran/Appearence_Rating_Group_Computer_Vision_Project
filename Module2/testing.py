from ultralytics import YOLO
import cv2
import os
import numpy as np
import gradio as gr
from huggingface_hub import hf_hub_download
from PIL import Image

# ========== Constants & Global Models ==========
os.makedirs("cropped_people", exist_ok=True)
# Add this flag at the top of your script
DEBUG_VERTICAL_EXPANSION = True
PERSON_MODEL = YOLO("yolov8x.pt")
FACE_MODEL_PATH = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
FACE_MODEL = YOLO(FACE_MODEL_PATH)

# ========== Utility Functions ==========

def convert_pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def convert_cv2_to_rgb(cv2_image):
    return cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

def detect_people(cv2_image):
    results = PERSON_MODEL(cv2_image)
    return [det for det in results[0].boxes if int(det.cls) == 0]  # class 0 = person
def detect_faces_in_crop(person_crop_bgr, face_model):
    # Convert BGR (OpenCV) to RGB (PIL)
    pil_image = Image.fromarray(cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB))

    # Run face detection
    result = face_model(pil_image)[0]

    # Extract boxes with confidence > threshold
    faces = []
    for box in result.boxes:
        if float(box.conf[0]) > 0.4:  # Adjust threshold if needed
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            faces.append({
                'bbox': (x1, y1, x2, y2),
                'conf': float(box.conf[0])
            })

    return faces
def detect_faces(cv2_image, min_conf=0.4):
    pil_image = Image.fromarray(convert_cv2_to_rgb(cv2_image))
    results = FACE_MODEL(pil_image)[0]
    faces = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        faces.append({
            'bbox': (x1, y1, x2, y2),
            'conf': conf
        })
    return faces

def face_inside_person(face_bbox, person_bbox):
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    cx = (fx1 + fx2) // 2
    cy = (fy1 + fy2) // 2
    return px1 <= cx <= px2 and py1 <= cy <= py2

def draw_box_with_label(image, bbox, label, color):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def process_person(image, vis_image, person_box, person_conf, faces, person_id):
    x1, y1, x2, y2 = person_box
    matched_faces = []

    for face in faces:
        if face_inside_person(face['bbox'], (x1, y1, x2, y2)):
            matched_faces.append(face)
            draw_box_with_label(vis_image, face['bbox'], f"Face {face['conf']:.2f}", (255, 0, 0))

    if not matched_faces:
        return None

    draw_box_with_label(vis_image, (x1, y1, x2, y2), f"Person {person_conf:.2f}", (0, 255, 0))

    person_crop = image[y1:y2, x1:x2]
    cv2.imwrite(f"cropped_people/person_{person_id}.jpg", person_crop)
    return convert_cv2_to_rgb(person_crop)
def has_significant_overlap(bbox, approved_face_bboxes, iou_threshold=0.05):
    """
    Returns True if:
    - The bbox exactly matches any approved face bbox (pixel-perfect), or
    - The bbox overlaps significantly (IoU > threshold) with any approved face bbox
    """
    x1, y1, x2, y2 = bbox
    for ax1, ay1, ax2, ay2 in approved_face_bboxes:
        
        # 1. ✅ Exact match
        if (x1, y1, x2, y2) == (ax1, ay1, ax2, ay2):
            print("Exact match")
            return True

        # 2. ✅ Significant overlap
        inter_x1 = max(x1, ax1)
        inter_y1 = max(y1, ay1)
        inter_x2 = min(x2, ax2)
        inter_y2 = min(y2, ay2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        if inter_area == 0:
            continue

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (ax2 - ax1) * (ay2 - ay1)

        iou = inter_area / float(area1 + area2 - inter_area)
        if iou > iou_threshold:
            return True

    return False
def split_person_box_by_faces(image, person_box, face_boxes, margin=20):
    """
    Given a person bounding box and list of faces inside it, split into sub-regions per face.

    Args:
        image: original image (numpy array)
        person_box: torch box with .xyxy[0]
        face_boxes: list of face dicts inside this person
        margin: how much extra margin to include in cropped output

    Returns:
        List of cropped images (one per face in that person box)
    """
    px1, py1, px2, py2 = map(int, person_box.xyxy[0])
    sub_images = []

    for face in face_boxes:
        fx1, fy1, fx2, fy2 = face['x1'], face['y1'], face['x2'], face['y2']

        # Expand box around face but clip to person region
        cx1 = max(px1, fx1 - margin)
        cy1 = max(py1, fy1 - margin)
        cx2 = min(px2, fx2 + margin)
        cy2 = min(py2, fy2 + margin)

        crop = image[cy1:cy2, cx1:cx2]
        sub_images.append(crop)

    return sub_images

def get_faces_in_person_box(person_box, faces):
    
    px1, py1, px2, py2 = map(int, person_box.xyxy[0])

    matching_faces = []
    for face in faces:
        fx1, fy1, fx2, fy2 = face['bbox']
        # Check if the face lies fully inside the person box
        if fx1 >= px1 and fy1 >= py1 and fx2 <= px2 and fy2 <= py2:
            matching_faces.append(face)

    return matching_faces, len(matching_faces)


MAX_WIDTH = 1280
MAX_HEIGHT = 720

def resize_to_fit_screen(image, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    if scale < 1.0:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image
##################################
import cv2
import os
def boxes_overlap(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    return xB > xA and yB > yA

def box_contains_face(big_box, face_box, overlap_thresh=0.01):
    """Returns True if face_box overlaps big_box by a small fraction."""
    xA = max(big_box[0], face_box[0])
    yA = max(big_box[1], face_box[1])
    xB = min(big_box[2], face_box[2])
    yB = min(big_box[3], face_box[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    face_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])

    overlap_ratio = inter_area / (face_area + 1e-5)
    return overlap_ratio > overlap_thresh
def safe_expand_crop_vertical_then_horizontal(
    image,
    face_box,
    other_faces,
    step=10,
    overlap_thresh=0.01,
    padding=15,
):
    img_h, img_w = image.shape[:2]
    fx1, fy1, fx2, fy2 = face_box

    # === 1. Vertical Expansion ===
    top, bottom = fy1, fy2
    can_expand_top = True
    can_expand_bottom = True

    while can_expand_top or can_expand_bottom:
        reached_top_edge = top <= 0
        reached_bottom_edge = bottom >= img_h

        new_top = max(0, top - step) if can_expand_top and not reached_top_edge else top
        new_bottom = min(img_h, bottom + step) if can_expand_bottom and not reached_bottom_edge else bottom

        if top == new_top:
            can_expand_top = False
        if bottom == new_bottom:
            can_expand_bottom = False

        for other in other_faces:
            if other == face_box:
                continue
            if can_expand_top and box_contains_face((fx1, new_top, fx2, top), other, overlap_thresh):
                can_expand_top = False
            if can_expand_bottom and box_contains_face((fx1, bottom, fx2, new_bottom), other, overlap_thresh):
                can_expand_bottom = False

        if can_expand_top:
            top = new_top
        if can_expand_bottom:
            bottom = new_bottom

        if not can_expand_top and not can_expand_bottom:
            break

    # === 2. Horizontal Expansion ===
    left, right = fx1, fx2
    can_expand_left = True
    can_expand_right = True

    while can_expand_left or can_expand_right:
        reached_left_edge = left <= 0
        reached_right_edge = right >= img_w

        new_left = max(0, left - step) if can_expand_left and not reached_left_edge else left
        new_right = min(img_w, right + step) if can_expand_right and not reached_right_edge else right

        if left == new_left:
            can_expand_left = False
        if right == new_right:
            can_expand_right = False

        for other in other_faces:
            if other == face_box:
                continue
            if can_expand_left and box_contains_face((new_left, top, left, bottom), other, overlap_thresh):
                can_expand_left = False
            if can_expand_right and box_contains_face((right, top, new_right, bottom), other, overlap_thresh):
                can_expand_right = False

        if can_expand_left:
            left = new_left
        if can_expand_right:
            right = new_right

        if not can_expand_left and not can_expand_right:
            break

    # === Final crop with padding ===
    left = max(0, left - padding)
    right = min(img_w, right + padding)
    top = max(0, top - padding)
    bottom = min(img_h, bottom + padding)

    crop = image[top:bottom, left:right]
    return crop if crop.shape[0] > 30 and crop.shape[1] > 30 else None




def isolate_people_from_faces(image, face_boxes, save_dir="cropped_people"):
    os.makedirs(save_dir, exist_ok=True)
    processed = set()
    count = 0

    for i, face in enumerate(face_boxes):
        if i in processed:
            continue

        current_bbox = face['bbox']
        other_faces = [f['bbox'] for j, f in enumerate(face_boxes) if j != i]

        crop = safe_expand_crop_vertical_then_horizontal(image, current_bbox, other_faces,overlap_thresh=0.0001)

        if crop is not None:
            filename = os.path.join(save_dir, f"person_{count}.jpg")
            cv2.imwrite(filename, crop)
            processed.add(i)
            count += 1

    print(f"✅ Isolated and saved {count} people.")


def detect_people_and_faces(input_image):
    image = convert_pil_to_cv2(input_image)
    faces = detect_faces(image)  # returns list of {'bbox': (x1, y1, x2, y2), 'conf': ...}

    # Draw detected faces on a copy of the image and save, including confidence in the label
    image_with_faces = image.copy()
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        conf = face.get('conf', 0)
        label = f"{conf:.2f}"
        cv2.rectangle(image_with_faces, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for face
        cv2.putText(image_with_faces, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.imwrite("detected_faces.jpg", image_with_faces)

    isolate_people_from_faces(image, faces, save_dir="cropped_people")



# =====p===== Gradio App ==========

with gr.Blocks(title="People with Faces Detection App") as demo:
    gr.Markdown("# People with Faces Detection App")
    gr.Markdown("Upload an image to detect people with visible faces. People are shown with green boxes, faces with blue boxes.")
    
    with gr.Row():
        input_image = gr.Image(label="Input Image", type="pil")
        output_image = gr.Image(label="Detected People & Faces")
    
    with gr.Row():
        submit_btn = gr.Button("Detect People with Faces")
    
    gallery = gr.Gallery(label="Cropped People with Faces", show_label=True, columns=5, height=400)
    output_text = gr.Textbox(label="Results")
    
    submit_btn.click(
        fn=detect_people_and_faces,
        inputs=input_image,
        outputs=[output_image, gallery, output_text]
    )

# ========== Entry Point ==========
if __name__ == "__main__":
    # Load the image first
    input_image = Image.open("groupPhotos/image1.jpg")
    detect_people_and_faces(input_image)
    #filter_people_to_find_single("cropped_people")
    #demo.launch()


