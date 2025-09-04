import cv2
import time
import statistics
import matplotlib.pyplot as plt
import json
import os
import shutil
from testing_media_pipe import MediaPipeService

def clean_and_create_output_folder(output_folder="video_results"):
    """
    Delete existing output folder and create a fresh one
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        print(f"🗑️ Deleted existing output folder: {output_folder}")
    
    os.makedirs(output_folder)
    print(f"📁 Created fresh output folder: {output_folder}")
    return output_folder

def analyze_video(video_path, output_folder="video_results"):
    """
    Analyze a video file and save results
    """
    print(f"🎥 Starting video analysis: {video_path}")
    
    # Clean and create output folder
    output_folder = clean_and_create_output_folder(output_folder)
    
    # Initialize video capture and MediaPipe service
    cap = cv2.VideoCapture(video_path)
    mp_service = MediaPipeService()
    
    # Score trackers
    timestamps = []
    final_scores = []
    smile_scores = []
    smile_active_scores = []
    confidence_scores = []
    head_pose_scores = []
    eye_contact_scores = []
    
    start_time = time.time()
    last_logged_second = -1
    frame_count = 0
    
    print("🔄 Processing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ==================== ✅ STAGE 1: DETECTION & COMPUTATION ====================
        
        # Process face and pose
        mp_service.process_face(frame)
        mp_service.process_pose(frame)
        
        simile_active, smile_score = 0, 0
        eye_contact = 0
        
        # Face landmark-based detections
        if mp_service.face_results and mp_service.face_results.multi_face_landmarks:
            for face_landmarks in mp_service.face_results.multi_face_landmarks:
                simile_active, smile_score = mp_service.detect_smile(
                    face_landmarks.landmark,
                    frame.shape[1],
                    frame.shape[0],
                    frame
                )
                eye_contact, left_eye_corners, right_eye_corners, left_iris_center, right_iris_center = mp_service.detect_eye_contact(
                    frame, face_landmarks.landmark
                )
        
        # Pose and head pose metrics
        pose_metrics = mp_service.pose_detection(frame)
        head_pose_metrics = mp_service.process_head_pose(frame)
        
        # Normalize posture confidence
        pose_confidence = pose_metrics.get("confidence_score", 0.0) or 0.0
        confidence_score = min(max(pose_confidence, 0.0), 1.0)
        
        # Final combined score calculation
        final_score, norm_smile_score, smile_active_score, confidence_score, head_pose_score, eye_contact = mp_service.calculate_final_score(
            smile_score, simile_active, confidence_score, head_pose_metrics.get("head_pose_text", 0), eye_contact
        )
        
        # Logging scores per second
        current_time = time.time() - start_time
        current_second = int(current_time)
        
        if current_second != last_logged_second:
            last_logged_second = current_second
            timestamps.append(current_second)
            final_scores.append(final_score)
            smile_scores.append(norm_smile_score)
            smile_active_scores.append(smile_active_score)
            confidence_scores.append(confidence_score)
            head_pose_scores.append(head_pose_score)
            eye_contact_scores.append(eye_contact)
            
            print(f"⏱️ Second {current_second}: Final Score = {final_score:.2f}")
        
        # ==================== �� STAGE 2: DRAWING & VISUALIZATION ====================
        
        # Print status on frame
        mp_service.print_smile_status(frame, simile_active, smile_score)
        mp_service.draw_eye_contact_status(frame, eye_contact)
        mp_service.print_pose_stats(frame, pose_metrics)
        mp_service.print_head_pose_stats(frame, head_pose_metrics)
        mp_service.print_final_score(frame, final_score)
        
        # Live bar drawing
        mp_service.draw_live_bars(frame, {
            "Final": final_score,
            "Smile": smile_score,
            "Active Smile": smile_active_score,
            "Posture Score": confidence_score,
            "Head Straight": head_pose_score,
            "Eye Contact": eye_contact
        }, origin=(30, 300))
        
        # Show final frame (optional - comment out for headless processing)
        cv2.imshow("MediaPipe Output", frame)
        
        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # ==================== 📊 FINAL ANALYSIS ====================
    
    if not timestamps:
        print("❌ No data collected from video")
        return None
    
    # Calculate statistics
    average_final_score = statistics.mean(final_scores)
    average_smile_score = statistics.mean(smile_scores)
    average_confidence_score = statistics.mean(confidence_scores)
    average_head_pose_score = statistics.mean(head_pose_scores)
    average_eye_contact_score = statistics.mean(eye_contact_scores)
    
    total_duration = timestamps[-1] if timestamps else 0
    
    # Create results dictionary
    results = {
        "video_path": video_path,
        "total_duration_seconds": total_duration,
        "total_frames": frame_count,
        "average_scores": {
            "final_score": round(average_final_score, 3),
            "smile_score": round(average_smile_score, 3),
            "confidence_score": round(average_confidence_score, 3),
            "head_pose_score": round(average_head_pose_score, 3),
            "eye_contact_score": round(average_eye_contact_score, 3)
        },
        "score_timeline": {
            "timestamps": timestamps,
            "final_scores": [round(score, 3) for score in final_scores],
            "smile_scores": [round(score, 3) for score in smile_scores],
            "confidence_scores": [round(score, 3) for score in confidence_scores],
            "head_pose_scores": [round(score, 3) for score in head_pose_scores],
            "eye_contact_scores": [round(score, 3) for score in eye_contact_scores]
        }
    }
    
    # Save results to JSON
    results_file = os.path.join(output_folder, "video_analysis_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved results to: {results_file}")
    
    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, final_scores, label='Final Score', color='blue')
    plt.axhline(y=average_final_score, color='gray', linestyle='--', label=f'Avg: {average_final_score:.2f}')
    plt.text(timestamps[-1], average_final_score + 0.01, f'Avg: {average_final_score:.2f}', fontsize=10, color='gray')
    plt.text(0, min(final_scores) - 0.05, f'Total Time: {total_duration:.2f} sec', fontsize=10, color='black')
    
    plt.title('Final Score Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Final Score')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    
    plot_file = os.path.join(output_folder, "score_timeline.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved plot to: {plot_file}")
    #plt.show()
    
    # Print summary
    print("\n" + "="*50)
    print("📊 VIDEO ANALYSIS SUMMARY")
    print("="*50)
    print(f"�� Video: {video_path}")
    print(f"⏱️ Duration: {total_duration} seconds")
    print(f"📊 Total Frames: {frame_count}")
    print(f"🎯 Average Final Score: {average_final_score:.2f}")
    print(f"😊 Average Smile Score: {average_smile_score:.2f}")
    print(f"�� Average Confidence Score: {average_confidence_score:.2f}")
    print(f"👤 Average Head Pose Score: {average_head_pose_score:.2f}")
    print(f"👀 Average Eye Contact Score: {average_eye_contact_score:.2f}")
    print("="*50)
    
    return results

def main():
    print("🎥 Video Analysis Tool")
    print("="*40)
    
    # Get video path
    video_path = input("Enter the path to your video file: ").strip()
    
    if not os.path.exists(video_path):
        print("❌ Error: Video file not found!")
        return
    
    # Analyze video
    results = analyze_video(video_path)
    
    if results:
        print("✅ Video analysis complete!")
        print(f"📁 Check the 'video_results' folder for detailed results and plots!")
    else:
        print("❌ Video analysis failed!")

if __name__ == "__main__":
    main()