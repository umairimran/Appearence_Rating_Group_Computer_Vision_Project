from Media_Pipe_Service import MediaPipeService
import cv2
import json

def example_image_processing():
    """Example of processing an image file"""
    print("=== Image Processing Example ===")
    
    # Initialize the service
    mediapipe_service = MediaPipeService()
    
    # Process an image (replace with your image path)
    image_path = "test_image.jpg"  # Change this to your image path
    
    try:
        result = mediapipe_service.process_image(image_path)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print(f"✅ Successfully processed image: {result['image_path']}")
        print(f"📏 Image dimensions: {result['image_dimensions']['width']}x{result['image_dimensions']['height']}")
        print(f"👥 Faces detected: {result['faces_detected']}")
        
        if result['success'] and result['landmarks']:
            landmarks = result['landmarks'][0]  # First face
            print(f"📍 Total landmarks: {len(landmarks)}")
            
            # Get key facial features
            summary = mediapipe_service.get_face_landmarks_summary(landmarks)
            print("\n🎯 Key facial features:")
            for feature, coords in summary.items():
                print(f"  {feature}: ({coords['x']:.3f}, {coords['y']:.3f}, {coords['z']:.3f})")
            
            # Save landmarks to JSON
            output_path = "landmarks_output.json"
            if mediapipe_service.save_landmarks_to_json(result, output_path):
                print(f"💾 Landmarks saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        mediapipe_service.cleanup()

def example_video_frame_processing():
    """Example of processing video frames"""
    print("\n=== Video Frame Processing Example ===")
    
    # Initialize the service
    mediapipe_service = MediaPipeService(static_image_mode=False)
    
    # Open video file (replace with your video path)
    video_path = "video.mp4"  # Change this to your video path
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 10th frame for efficiency
            if frame_count % 10 == 0:
                result = mediapipe_service.process_frame(frame)
                
                if result['success']:
                    processed_frames += 1
                    landmarks = result['landmarks'][0]  # First face
                    
                    print(f"Frame {frame_count}: {len(landmarks)} landmarks detected")
                    
                    # Example: Get nose tip landmark
                    nose_landmarks = mediapipe_service.get_specific_landmarks(landmarks, [4])
                    if nose_landmarks:
                        nose = nose_landmarks[0]
                        print(f"  Nose tip: ({nose['x']:.3f}, {nose['y']:.3f})")
        
        print(f"\n📊 Summary:")
        print(f"  Total frames: {frame_count}")
        print(f"  Processed frames: {processed_frames}")
        print(f"  Frames with faces: {processed_frames}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        cap.release()
        mediapipe_service.cleanup()

def example_quick_face_check():
    """Example of quick face detection"""
    print("\n=== Quick Face Detection Example ===")
    
    # Initialize the service
    mediapipe_service = MediaPipeService()
    
    # Open video file
    video_path = "video.mp4"  # Change this to your video path
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return
        
        frame_count = 0
        frames_with_faces = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Quick face check (faster than full landmark extraction)
            if mediapipe_service.has_face(frame):
                frames_with_faces += 1
                print(f"Frame {frame_count}: Face detected ✅")
            else:
                print(f"Frame {frame_count}: No face detected ❌")
            
            # Process only first 50 frames for demo
            if frame_count >= 50:
                break
        
        print(f"\n📊 Summary:")
        print(f"  Frames checked: {frame_count}")
        print(f"  Frames with faces: {frames_with_faces}")
        print(f"  Face detection rate: {frames_with_faces/frame_count*100:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        cap.release()
        mediapipe_service.cleanup()

if __name__ == "__main__":
    print("🎯 MediaPipe Service Examples")
    print("=" * 40)
    
    # Run examples (uncomment the ones you want to test)
    
    # example_image_processing()
    # example_video_frame_processing()
    # example_quick_face_check()
    
    print("\n💡 To run examples:")
    print("1. Uncomment the example functions above")
    print("2. Update the file paths to point to your images/videos")
    print("3. Run this script")
    
    print("\n🔧 Quick test - Initializing service...")
    try:
        service = MediaPipeService()
        print("✅ MediaPipe service initialized successfully!")
        service.cleanup()
    except Exception as e:
        print(f"❌ Error initializing service: {str(e)}") 