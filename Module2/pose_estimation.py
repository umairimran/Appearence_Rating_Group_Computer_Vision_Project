from ultralytics import YOLO
import cv2

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")  # You can use yolov8s-pose.pt, yolov8m-pose.pt, etc.

# Load an image
image_path = "your_image.jpg"
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Draw poses on the image
annotated_frame = results[0].plot()

# Show or save the result
cv2.imshow("Pose Estimation", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Print keypoints and confidence
for i, kp in enumerate(results[0].keypoints.data):
    print(f"Person {i+1}:")
    for j, point in enumerate(kp):
        x, y, conf = point.tolist()
        print(f"  Keypoint {j}: x={x:.1f}, y={y:.1f}, confidence={conf:.2f}")
