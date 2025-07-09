# Pose Estimation with Gradio

This module provides a user-friendly interface for pose estimation using YOLOv8 and Gradio.

## Setup

1. Make sure you have all the required dependencies installed:

   ```
   pip install -r ../requirements.txt
   ```
2. If the YOLOv8 pose model doesn't download automatically, you can manually download it:

   ```
   pip install ultralytics
   yolo download pose
   ```

## Running the Application

To run the Gradio application:

```bash
python gradio_pose_estimation.py
```

This will start a local web server (typically at http://127.0.0.1:7860) where you can:

- Upload images for pose estimation
- View the detected poses with keypoints
- See detailed keypoint coordinates and confidence values

## Features

- Simple web interface for pose detection
- Upload your own images or use example images
- Real-time pose detection and visualization
- Detailed keypoints data for each detected person

## Example Usage

1. Launch the application
2. Upload an image or select one of the example images
3. Click "Detect Poses"
4. View the results with pose keypoints highlighted
5. Check the text output for detailed keypoint coordinates

## Notes

- The application uses the YOLOv8 nano pose estimation model by default
- For better accuracy but slower performance, you can modify the code to use yolov8s-pose.pt, yolov8m-pose.pt, etc.
