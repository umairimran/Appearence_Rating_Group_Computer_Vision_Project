import cv2
import numpy as np
from Media_Pipe_Service import MediaPipeService
import time

def main():
    # Initialize
    service = MediaPipeService()
    cap = cv2.VideoCapture(0)  # Use default camera
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Initialize tracking variables
    frame_count = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    scores_history = {
        'confidence': [],
        'warmth': [],
        'smile': [],
        'posture': [],
        'eye_contact': []
    }
    
    print("🎥 Real-time Appearance Analysis Started!")
    print("📋 Controls:")
    print("   - Press 'q' to quit")
    print("   - Press 's' to save scores")
    print("   - Press 'r' to reset scores")
    print("   - Press 'h' to hide/show landmarks")
    
    show_landmarks = True
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read camera frame")
            break
            
        frame_count += 1
        fps_counter += 1
        
        # Calculate FPS every second
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        # Analyze frame with all metrics
        analysis = service.analyze_frame(frame)
        
        if analysis['success']:
            # Extract scores
            confidence_score = analysis['confidence_score']
            warmth_score = analysis['warmth_score']
            smile_score = analysis['smile_score']
            posture_score = analysis['posture_score']
            eye_contact_score = analysis['eye_contact_score']
            
            # Store scores for tracking (keep last 50 frames for performance)
            scores_history['confidence'].append(confidence_score)
            scores_history['warmth'].append(warmth_score)
            scores_history['smile'].append(smile_score)
            scores_history['posture'].append(posture_score)
            scores_history['eye_contact'].append(eye_contact_score)
            
            # Keep only last 50 frames for smooth performance
            max_history = 50
            for key in scores_history:
                if len(scores_history[key]) > max_history:
                    scores_history[key] = scores_history[key][-max_history:]
            
            # Draw landmarks and information
            h, w = frame.shape[:2]
            
            if show_landmarks:
                # Draw key landmarks
                landmarks = analysis['landmarks']
                
                # Draw face mesh points (selected key points)
                key_points = [4, 33, 263, 61, 291, 13, 152, 10, 234, 454]  # nose, eyes, mouth, chin, forehead, ears
                for point_idx in key_points:
                    if point_idx < len(landmarks):
                        point = landmarks[point_idx]
                        point_x, point_y = int(point["x"] * w), int(point["y"] * h)
                        cv2.circle(frame, (point_x, point_y), 2, (0, 255, 255), -1)
                
                # Highlight important landmarks with different colors
                # Nose tip (green)
                nose = landmarks[4]
                nose_x, nose_y = int(nose["x"] * w), int(nose["y"] * h)
                cv2.circle(frame, (nose_x, nose_y), 4, (0, 255, 0), -1)
                
                # Eyes (blue)
                left_eye = landmarks[33]
                left_eye_x, left_eye_y = int(left_eye["x"] * w), int(left_eye["y"] * h)
                cv2.circle(frame, (left_eye_x, left_eye_y), 3, (255, 0, 0), -1)
                
                right_eye = landmarks[263]
                right_eye_x, right_eye_y = int(right_eye["x"] * w), int(right_eye["y"] * h)
                cv2.circle(frame, (right_eye_x, right_eye_y), 3, (255, 0, 0), -1)
                
                # Mouth corners (red)
                mouth_left = landmarks[61]
                mouth_right = landmarks[291]
                mouth_left_x, mouth_left_y = int(mouth_left["x"] * w), int(mouth_left["y"] * h)
                mouth_right_x, mouth_right_y = int(mouth_right["x"] * w), int(mouth_right["y"] * h)
                cv2.circle(frame, (mouth_left_x, mouth_left_y), 3, (0, 0, 255), -1)
                cv2.circle(frame, (mouth_right_x, mouth_right_y), 3, (0, 0, 255), -1)
                
                # Draw face outline
                face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
                for i in range(len(face_outline) - 1):
                    if face_outline[i] < len(landmarks) and face_outline[i+1] < len(landmarks):
                        pt1 = landmarks[face_outline[i]]
                        pt2 = landmarks[face_outline[i+1]]
                        pt1_x, pt1_y = int(pt1["x"] * w), int(pt1["y"] * h)
                        pt2_x, pt2_y = int(pt2["x"] * w), int(pt2["y"] * h)
                        cv2.line(frame, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 255, 255), 1)
            
            # Create overlay for scores
            overlay = frame.copy()
            
            # Semi-transparent background for text
            cv2.rectangle(overlay, (10, 10), (300, 180), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Display real-time information
            y_offset = 35
            cv2.putText(frame, f"FPS: {current_fps} | Frame: {frame_count}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            y_offset += 25
            # Confidence (Green)
            confidence_color = (0, 255, 0) if confidence_score > 0.7 else (0, 200, 0) if confidence_score > 0.4 else (0, 100, 0)
            cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, confidence_color, 1)
            
            y_offset += 20
            # Warmth (Yellow)
            warmth_color = (0, 255, 255) if warmth_score > 0.7 else (0, 200, 200) if warmth_score > 0.4 else (0, 100, 100)
            cv2.putText(frame, f"Warmth: {warmth_score:.2f}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, warmth_color, 1)
            
            y_offset += 20
            # Smile (Red)
            smile_emoji = "😊" if analysis['is_smiling'] else "😐"
            smile_color = (0, 0, 255) if analysis['is_smiling'] else (0, 0, 150)
            cv2.putText(frame, f"Smile: {smile_score:.2f} {smile_emoji}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, smile_color, 1)
            
            y_offset += 20
            # Posture (Purple)
            posture_color = (255, 0, 255) if posture_score > 0.7 else (200, 0, 200) if posture_score > 0.4 else (100, 0, 100)
            cv2.putText(frame, f"Posture: {posture_score:.2f} ({analysis['head_angle']:.0f}°)", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, posture_color, 1)
            
            y_offset += 20
            # Eye Contact (Cyan)
            eye_emoji = "👁️" if analysis['looking_forward'] else "👀"
            eye_color = (255, 255, 0) if analysis['looking_forward'] else (200, 200, 0)
            cv2.putText(frame, f"Eye Contact: {eye_contact_score:.2f} {eye_emoji}", (15, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
            
            # Draw score bars at bottom
            bar_width = 200
            bar_height = 12
            bar_y = h - 80
            
            # Confidence bar
            cv2.rectangle(frame, (15, bar_y), (15 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, bar_y), (15 + int(bar_width * confidence_score), bar_y + bar_height), confidence_color, -1)
            cv2.putText(frame, "Confidence", (220, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Warmth bar
            bar_y += 20
            cv2.rectangle(frame, (15, bar_y), (15 + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            cv2.rectangle(frame, (15, bar_y), (15 + int(bar_width * warmth_score), bar_y + bar_height), warmth_color, -1)
            cv2.putText(frame, "Warmth", (220, bar_y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Overall score
            overall_score = (confidence_score + warmth_score) / 2
            overall_color = (0, 255, 0) if overall_score > 0.7 else (0, 200, 0) if overall_score > 0.4 else (0, 100, 0)
            
            # Display overall score prominently
            cv2.putText(frame, f"Overall: {overall_score:.2f}", (w - 150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, overall_color, 2)
            
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (w//2 - 100, h//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, "Please look at the camera", (w//2 - 120, h//2 + 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("🎥 Real-time Appearance Analysis", frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_scores(scores_history, frame_count)
        elif key == ord('r'):
            # Reset scores
            scores_history = {key: [] for key in scores_history}
            print("🔄 Scores reset!")
        elif key == ord('h'):
            show_landmarks = not show_landmarks
            print(f"🎯 Landmarks {'hidden' if not show_landmarks else 'shown'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    service.cleanup()
    
    # Print final summary
    if scores_history['confidence']:
        print("\n" + "="*50)
        print("📊 FINAL ANALYSIS SUMMARY")
        print("="*50)
        print(f"📹 Total frames analyzed: {frame_count}")
        print(f"⏱️  Average FPS: {current_fps}")
        print(f"🎯 Average confidence: {np.mean(scores_history['confidence']):.3f}")
        print(f"❤️  Average warmth: {np.mean(scores_history['warmth']):.3f}")
        print(f"😊 Average smile: {np.mean(scores_history['smile']):.3f}")
        print(f"🧍 Average posture: {np.mean(scores_history['posture']):.3f}")
        print(f"👁️  Average eye contact: {np.mean(scores_history['eye_contact']):.3f}")
        
        overall_avg = (np.mean(scores_history['confidence']) + np.mean(scores_history['warmth'])) / 2
        print(f"🌟 Overall appearance score: {overall_avg:.3f}")
        
        # Performance rating
        if overall_avg > 0.8:
            print("🏆 Excellent appearance!")
        elif overall_avg > 0.6:
            print("👍 Good appearance!")
        elif overall_avg > 0.4:
            print("😐 Average appearance")
        else:
            print("📈 Room for improvement")

def save_scores(scores_history, frame_count):
    """Save scores to file"""
    import json
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"appearance_scores_{timestamp}.json"
    
    data = {
        'timestamp': timestamp,
        'frame_count': frame_count,
        'scores': scores_history,
        'summary': {
            'avg_confidence': np.mean(scores_history['confidence']) if scores_history['confidence'] else 0,
            'avg_warmth': np.mean(scores_history['warmth']) if scores_history['warmth'] else 0,
            'avg_smile': np.mean(scores_history['smile']) if scores_history['smile'] else 0,
            'avg_posture': np.mean(scores_history['posture']) if scores_history['posture'] else 0,
            'avg_eye_contact': np.mean(scores_history['eye_contact']) if scores_history['eye_contact'] else 0,
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Scores saved to {filename}")

if __name__ == "__main__":
    main() 