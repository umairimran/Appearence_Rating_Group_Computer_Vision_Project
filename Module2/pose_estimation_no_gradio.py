from ultralytics import YOLO
import cv2
import os
import numpy as np
import argparse

# Create output directory if it doesn't exist
os.makedirs("pose_output", exist_ok=True)

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt")  # You can use yolov8s-pose.pt, yolov8m-pose.pt, etc.

def process_image(image_path, save_output=True):
    """
    Process an image with the YOLO pose estimation model
    
    Args:
        image_path (str): Path to the input image
        save_output (bool): Whether to save output image to disk
        
    Returns:
        None
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Run inference
    results = model(image)
    
    # Draw poses on the image
    annotated_frame = results[0].plot()
    
    # Print keypoints and confidence to console
    print(f"\nResults for {image_path}:")
    for i, kp in enumerate(results[0].keypoints.data):
        print(f"Person {i+1}:")
        for j, point in enumerate(kp):
            x, y, conf = point.tolist()
            print(f"  Keypoint {j}: x={x:.1f}, y={y:.1f}, confidence={conf:.2f}")
    
    # Save the output image if requested
    if save_output:
        filename = os.path.basename(image_path)
        output_path = os.path.join("pose_output", f"pose_{filename}")
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved output image to {output_path}")
    
    # Display the result
    cv2.imshow("Pose Estimation", annotated_frame)
    cv2.waitKey(0)

def process_directory(directory_path):
    """
    Process all images in a directory
    
    Args:
        directory_path (str): Path to the directory containing images
    """
    # Image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Loop through all files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        
        # Check if it's an image file
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            process_image(file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 Pose Estimation')
    parser.add_argument('--image', type=str, help='Path to a single image')
    parser.add_argument('--dir', type=str, help='Path to directory containing images')
    args = parser.parse_args()
    
    if args.image:
        # Process a single image
        process_image(args.image)
    elif args.dir:
        # Process all images in a directory
        process_directory(args.dir)
    else:
        # Default to processing some example images
        example_dir = "Module2/groupPhotos"
        if os.path.exists(example_dir):
            print(f"Processing example images in {example_dir}")
            process_directory(example_dir)
        else:
            print("Please provide an image path (--image) or directory path (--dir)")
    
    cv2.destroyAllWindows() 