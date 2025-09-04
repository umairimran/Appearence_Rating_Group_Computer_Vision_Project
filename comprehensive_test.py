#!/usr/bin/env python3
"""
Comprehensive Testing Script for Appearance Rating and Group Synergy AI
Tests all components of both Module 1 and Module 2
"""

import os
import sys
import time
import json
import numpy as np
import cv2
from pathlib import Path

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def print_success(message):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print error message"""
    print(f"❌ {message}")

def print_warning(message):
    """Print warning message"""
    print(f"⚠️  {message}")

def print_info(message):
    """Print info message"""
    print(f"ℹ️  {message}")

def test_python_version():
    """Test Python version compatibility"""
    print_header("Python Version Check")
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and 8 <= version.minor <= 11:
        print_success("Python version is compatible")
        return True
    else:
        print_error("Python version is not compatible (need 3.8-3.11)")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print_header("Dependencies Check")
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("ultralytics", "Ultralytics YOLO"),
        ("gradio", "Gradio"),
        ("numpy", "NumPy"),
        ("matplotlib", "Matplotlib"),
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("sklearn", "Scikit-learn"),
        ("google.generativeai", "Google Generative AI")
    ]
    
    failed_deps = []
    for module_name, display_name in dependencies:
        try:
            __import__(module_name)
            print_success(f"{display_name} imported successfully")
        except ImportError as e:
            print_error(f"{display_name} import failed: {e}")
            failed_deps.append(display_name)
    
    if failed_deps:
        print_warning(f"Failed to import: {', '.join(failed_deps)}")
        return False
    
    return True

def test_environment_variables():
    """Test environment variables setup"""
    print_header("Environment Variables Check")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            print_success("GEMINI_API_KEY found")
            return True
        else:
            print_warning("GEMINI_API_KEY not found in .env file")
            print_info("Module 2 color analysis features will be limited")
            return False
    except ImportError:
        print_error("python-dotenv not installed")
        return False

def test_module1_components():
    """Test Module 1 components"""
    print_header("Module 1 - Video Analysis Components")
    
    try:
        # Test MediaPipe service import
        sys.path.append("Module1")
        from testing_media_pipe import MediaPipeService
        print_success("MediaPipe service imported")
        
        # Test service initialization
        mp_service = MediaPipeService()
        print_success("MediaPipe service initialized")
        
        # Test with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_service.process_face(dummy_image)
        mp_service.process_pose(dummy_image)
        print_success("Basic processing test passed")
        
        # Test smile detection
        landmarks = [type('Landmark', (), {'x': 0.5, 'y': 0.5})() for _ in range(500)]
        smile_active, smile_score = mp_service.detect_smile(landmarks, 640, 480, dummy_image)
        print_success("Smile detection test passed")
        
        # Test score calculation
        final_score = mp_service.calculate_final_score(0.5, True, 0.8, 0.9, 1)
        print_success("Score calculation test passed")
        
        return True
        
    except ImportError as e:
        print_error(f"Module 1 import failed: {e}")
        return False
    except Exception as e:
        print_error(f"Module 1 test failed: {e}")
        return False

def test_module2_components():
    """Test Module 2 components"""
    print_header("Module 2 - Group Photo Analysis Components")
    
    try:
        # Test YOLO import
        from ultralytics import YOLO
        print_success("YOLO imported successfully")
        
        # Test MediaPipe service import
        sys.path.append("Module2")
        from Media_Pipe__Service import MediaPipeService
        print_success("Module 2 MediaPipe service imported")
        
        # Test service initialization
        mp_service = MediaPipeService()
        print_success("Module 2 MediaPipe service initialized")
        
        # Test with dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_service.process_face(dummy_image)
        mp_service.process_pose(dummy_image)
        print_success("Basic processing test passed")
        
        # Test YOLO model loading
        try:
            model = YOLO("yolov8n.pt")
            print_success("YOLO model loaded successfully")
        except Exception as e:
            print_warning(f"YOLO model not found (will download on first run): {e}")
        
        # Test main pipeline import
        from main import run_pipeline_from_image
        print_success("Main pipeline imported")
        
        # Test gradio components
        from gradio_complete import generate_group_summary, load_cropped_images
        print_success("Gradio components imported")
        
        return True
        
    except ImportError as e:
        print_error(f"Module 2 import failed: {e}")
        return False
    except Exception as e:
        print_error(f"Module 2 test failed: {e}")
        return False

def test_file_structure():
    """Test required file structure"""
    print_header("File Structure Check")
    
    required_files = [
        "requirements.txt",
        "Module1/simple_script_for_analysis.py",
        "Module1/testing_media_pipe.py",
        "Module1/Media_Pipe__Service.py",
        "Module2/simple_script_for_analysis.py",
        "Module2/gradio_complete.py",
        "Module2/main.py",
        "Module2/Media_Pipe__Service.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"{file_path} exists")
        else:
            print_error(f"{file_path} missing")
            missing_files.append(file_path)
    
    if missing_files:
        print_warning(f"Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def test_output_directories():
    """Test output directory creation"""
    print_header("Output Directory Test")
    
    test_dirs = ["Module1/video_results", "Module2/results", "cropped_people", "cropped_faces"]
    
    for dir_path in test_dirs:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print_success(f"Created/verified directory: {dir_path}")
        except Exception as e:
            print_error(f"Failed to create directory {dir_path}: {e}")
            return False
    
    return True

def test_sample_data():
    """Test with sample data if available"""
    print_header("Sample Data Test")
    
    # Check for sample video in Module1
    sample_video = "Module1/v.mp4"
    if os.path.exists(sample_video):
        print_success(f"Sample video found: {sample_video}")
        print_info("You can test Module 1 with this video")
    else:
        print_warning("No sample video found in Module1")
    
    # Check for YOLO models in Module2
    yolo_models = ["Module2/yolov8n.pt", "Module2/yolov8n-pose.pt"]
    for model_path in yolo_models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print_success(f"YOLO model found: {model_path} ({size_mb:.1f}MB)")
        else:
            print_warning(f"YOLO model not found: {model_path} (will download on first run)")
    
    return True

def test_performance():
    """Test basic performance metrics"""
    print_header("Performance Test")
    
    try:
        # Test image processing speed
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        sys.path.append("Module1")
        from testing_media_pipe import MediaPipeService
        
        mp_service = MediaPipeService()
        
        start_time = time.time()
        for _ in range(10):
            mp_service.process_face(dummy_image)
            mp_service.process_pose(dummy_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        fps = 1 / avg_time if avg_time > 0 else 0
        
        print_success(f"Average processing time: {avg_time:.3f}s per frame")
        print_success(f"Estimated FPS: {fps:.1f}")
        
        if fps > 20:
            print_success("Performance is good for real-time processing")
        elif fps > 10:
            print_warning("Performance is acceptable but may be slow for real-time")
        else:
            print_warning("Performance may be too slow for real-time processing")
        
        return True
        
    except Exception as e:
        print_error(f"Performance test failed: {e}")
        return False

def generate_test_report(results):
    """Generate comprehensive test report"""
    print_header("Test Report Summary")
    
    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests == 0:
        print_success("All tests passed! Your installation is ready.")
    else:
        print_warning(f"{failed_tests} tests failed. Check the details above.")
    
    # Save report to file
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "results": results,
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": (passed_tests/total_tests)*100
        }
    }
    
    with open("test_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print_success("Test report saved to test_report.json")

def main():
    """Main testing function"""
    print("="*60)
    print("🧪 Comprehensive Testing Suite")
    print("Appearance Rating and Group Synergy AI")
    print("="*60)
    
    results = {}
    
    # Run all tests
    results["python_version"] = test_python_version()
    results["dependencies"] = test_dependencies()
    results["environment_variables"] = test_environment_variables()
    results["file_structure"] = test_file_structure()
    results["output_directories"] = test_output_directories()
    results["module1_components"] = test_module1_components()
    results["module2_components"] = test_module2_components()
    results["sample_data"] = test_sample_data()
    results["performance"] = test_performance()
    
    # Generate report
    generate_test_report(results)
    
    # Print next steps
    print_header("Next Steps")
    
    if all(results.values()):
        print_success("🎉 All tests passed! You're ready to use both modules.")
        print("\nTo get started:")
        print("1. cd Module1 && python simple_script_for_analysis.py")
        print("2. cd Module2 && python simple_script_for_analysis.py")
        print("3. cd Module2 && python gradio_complete.py (for web interface)")
    else:
        print_warning("⚠️ Some tests failed. Please address the issues above.")
        print("\nCommon solutions:")
        print("1. Run: python install_setup.py")
        print("2. Check: pip install -r requirements.txt")
        print("3. Verify: .env file contains GEMINI_API_KEY")
    
    print("\nFor detailed documentation, see: Research_Document_Module1_Module2.md")
    print("For quick start guide, see: QUICK_START.md")

if __name__ == "__main__":
    main() 