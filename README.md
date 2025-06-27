# Appearance Rating Group - Computer Vision Project

This project provides a comprehensive MediaPipe service for facial landmark extraction and video preprocessing for appearance rating applications.

## 🎯 Features

- **Facial Landmark Extraction**: Extract 468 facial landmarks using MediaPipe Face Mesh
- **Video Preprocessing**: Clean videos by filtering frames based on quality criteria
- **Multiple Input Formats**: Support for both image files and video frames
- **JSON Export**: Save landmark data in structured JSON format
- **Face Detection**: Quick face presence detection for video processing
- **Quality Assessment**: Evaluate frame brightness, sharpness, and resolution

## 📁 Project Structure

```
Appearence_Rating_Group_Computer_Vision_Project/
├── Media_Pipe_Service.py      # Main MediaPipe service class
├── video_preprocessing.py     # Video cleaning and preprocessing
├── example_usage.py          # Usage examples and demonstrations
├── video.mp4                 # Input video file
├── README.md                 # This file
└── venv/                     # Python virtual environment
```

## 🚀 Quick Start

### 1. Setup Environment

Make sure you have the required dependencies installed:

```bash
# Activate virtual environment (if using one)
# On Windows:
venv\Scripts\activate

# On Linux/Mac:
source venv/bin/activate
```

### 2. Basic Usage

```python
from Media_Pipe_Service import MediaPipeService

# Initialize the service
service = MediaPipeService()

# Process an image
result = service.process_image("path/to/your/image.jpg")

# Check if successful
if result["success"]:
    landmarks = result["landmarks"][0]  # First face
    print(f"Found {len(landmarks)} landmarks")
    
    # Get specific landmarks (e.g., nose tip)
    nose_landmark = service.get_specific_landmarks(landmarks, [4])
    print(f"Nose tip: {nose_landmark[0]}")

# Clean up
service.cleanup()
```

### 3. Video Processing

```python
# Process video frames
service = MediaPipeService(static_image_mode=False)

cap = cv2.VideoCapture("video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    result = service.process_frame(frame)
    if result["success"]:
        landmarks = result["landmarks"][0]
        # Use landmarks for your analysis

cap.release()
service.cleanup()
```

## 🔧 MediaPipe Service API

### Initialization

```python
MediaPipeService(
    static_image_mode=True,      # True for images, False for video
    max_num_faces=1,             # Maximum faces to detect
    min_detection_confidence=0.5, # Detection confidence threshold
    min_tracking_confidence=0.5   # Tracking confidence threshold
)
```

### Main Methods

#### `process_image(image_path: str) -> Dict`
Process an image file and return landmarks.

**Returns:**
```json
{
    "image_path": "path/to/image.jpg",
    "image_dimensions": {"width": 1920, "height": 1080},
    "faces_detected": 1,
    "landmarks": [[{"x": 0.5, "y": 0.3, "z": 0.1}, ...]],
    "face_bbox": [{"x": 100, "y": 50, "width": 200, "height": 250}],
    "success": true
}
```

#### `process_frame(frame: np.ndarray) -> Dict`
Process a video frame (numpy array) and return landmarks.

#### `has_face(frame: np.ndarray) -> bool`
Quick check if a frame contains a face (faster than full landmark extraction).

#### `get_specific_landmarks(landmarks: List[Dict], indices: List[int]) -> List[Dict]`
Extract specific landmarks by their indices.

#### `get_face_landmarks_summary(landmarks: List[Dict]) -> Dict`
Get key facial features summary.

**Key Features Available:**
- `nose_tip`: Landmark 4
- `left_eye`: Landmark 33
- `right_eye`: Landmark 263
- `mouth_center`: Landmark 13
- `chin`: Landmark 152
- And more...

#### `save_landmarks_to_json(data: Dict, output_path: str) -> bool`
Save landmark data to JSON file.

#### `draw_landmarks_on_image(image: np.ndarray, landmarks: List[Dict]) -> np.ndarray`
Draw landmarks on an image for visualization.

## 🎬 Video Preprocessing

The `video_preprocessing.py` script filters video frames based on:

- **Resolution**: Minimum 640x480 pixels
- **Brightness**: Above threshold (configurable)
- **Sharpness**: Laplacian variance above threshold
- **Face Detection**: Must contain a detectable face

### Usage

```python
# Run the preprocessing script
python video_preprocessing.py

# The script will:
# 1. Read video.mp4
# 2. Filter frames based on quality criteria
# 3. Save cleaned video as cleaned_output.mp4
```

### Configuration

Edit the configuration section in `video_preprocessing.py`:

```python
MIN_WIDTH = 640              # Minimum frame width
MIN_HEIGHT = 480             # Minimum frame height
BRIGHTNESS_THRESHOLD = 50    # Minimum brightness
SHARPNESS_THRESHOLD = 100    # Minimum sharpness
MIN_VALID_FRAMES = 5         # Minimum valid frames required
OUTPUT_PATH = "cleaned_output.mp4"
```

## 📊 Example Usage

Run the example script to see the service in action:

```bash
python example_usage.py
```

This will demonstrate:
- Image processing with landmark extraction
- Video frame processing
- Quick face detection
- Landmark data export

## 🔍 Landmark Indices

MediaPipe Face Mesh provides 468 facial landmarks. Here are some key indices:

| Feature | Index | Description |
|---------|-------|-------------|
| Nose tip | 4 | Tip of the nose |
| Left eye | 33 | Center of left eye |
| Right eye | 263 | Center of right eye |
| Mouth center | 13 | Center of mouth |
| Left ear | 234 | Left ear |
| Right ear | 454 | Right ear |
| Chin | 152 | Bottom of chin |

## 🛠️ Troubleshooting

### Common Issues

1. **"Could not read image"**: Check file path and format
2. **No faces detected**: Ensure image contains clear, front-facing faces
3. **Performance issues**: Use `static_image_mode=False` for video processing
4. **Memory issues**: Process frames in batches or reduce `max_num_faces`

### Performance Tips

- Use `has_face()` for quick face detection before full landmark extraction
- Set `static_image_mode=False` for video processing
- Process every Nth frame for long videos
- Clean up resources with `cleanup()` method

## 📝 Dependencies

- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)
- JSON (built-in)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Appearance Rating Group Computer Vision Project.

---

**Note**: This service is designed for appearance rating applications and provides comprehensive facial landmark extraction capabilities. The landmarks can be used for various computer vision tasks including facial analysis, emotion detection, and appearance assessment.
