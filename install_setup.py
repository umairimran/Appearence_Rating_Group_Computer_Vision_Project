#!/usr/bin/env python3
"""
Automated Installation Script for Appearance Rating and Group Synergy AI
This script automates the setup process for both Module 1 and Module 2
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def print_banner():
    """Print installation banner"""
    print("=" * 60)
    print("🎥 Appearance Rating and Group Synergy AI - Setup")
    print("=" * 60)
    print("This script will set up your environment for both modules:")
    print("• Module 1: Video Analysis System")
    print("• Module 2: Group Photo Analysis System")
    print("=" * 60)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and 8 <= version.minor <= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Not compatible")
        print("   Please use Python 3.8-3.11")
        return False

def create_virtual_environment():
    """Create and activate virtual environment"""
    print("\n🐍 Setting up virtual environment...")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("   Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            shutil.rmtree(venv_path)
            print("🗑️  Removed existing virtual environment")
        else:
            print("✅ Using existing virtual environment")
            return True
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def get_activation_command():
    """Get the appropriate activation command for the OS"""
    system = platform.system().lower()
    if system == "windows":
        return "venv\\Scripts\\activate"
    else:
        return "source venv/bin/activate"

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Upgrade pip first
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print("✅ Upgraded pip")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Failed to upgrade pip: {e}")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
            print("✅ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    else:
        print("❌ requirements.txt not found")
        return False

def create_env_file():
    """Create .env file for API keys"""
    print("\n🔑 Setting up environment variables...")
    
    env_file = Path(".env")
    if env_file.exists():
        print("⚠️  .env file already exists")
        response = input("   Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("✅ Using existing .env file")
            return True
    
    print("📝 Creating .env file...")
    print("   You'll need a Google Gemini API key for Module 2")
    print("   Get it from: https://makersuite.google.com/app/apikey")
    
    api_key = input("Enter your Gemini API key (or press Enter to skip): ").strip()
    
    with open(".env", "w") as f:
        f.write(f"GEMINI_API_KEY={api_key}\n")
    
    if api_key:
        print("✅ .env file created with API key")
    else:
        print("⚠️  .env file created without API key")
        print("   You can add it later by editing the .env file")
    
    return True

def verify_installation():
    """Verify that all components are properly installed"""
    print("\n🔍 Verifying installation...")
    
    # Test imports
    test_imports = [
        "cv2",
        "mediapipe", 
        "ultralytics",
        "gradio",
        "numpy",
        "matplotlib"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All core dependencies verified")
    return True

def check_models():
    """Check if required models are present"""
    print("\n🤖 Checking AI models...")
    
    # Check Module2 models
    module2_models = [
        "Module2/yolov8n.pt",
        "Module2/yolov8n-pose.pt"
    ]
    
    missing_models = []
    for model_path in module2_models:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✅ {model_path} ({size_mb:.1f}MB)")
        else:
            print(f"⚠️  {model_path} - Will be downloaded on first run")
            missing_models.append(model_path)
    
    if missing_models:
        print("   Models will be automatically downloaded when you first run Module 2")
    
    return True

def create_test_scripts():
    """Create simple test scripts for users"""
    print("\n🧪 Creating test scripts...")
    
    # Test script for Module 1
    module1_test = '''#!/usr/bin/env python3
"""
Quick test for Module 1 - Video Analysis
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Module1.testing_media_pipe import MediaPipeService
    import cv2
    import numpy as np
    
    print("✅ Module 1 test successful!")
    print("   MediaPipe service can be imported")
    
    # Test service initialization
    mp_service = MediaPipeService()
    print("✅ MediaPipe service initialized")
    
    # Create a dummy image for testing
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_service.process_face(dummy_image)
    mp_service.process_pose(dummy_image)
    print("✅ Basic processing test passed")
    
except ImportError as e:
    print(f"❌ Module 1 test failed: {e}")
except Exception as e:
    print(f"❌ Module 1 test failed: {e}")
'''
    
    # Test script for Module 2
    module2_test = '''#!/usr/bin/env python3
"""
Quick test for Module 2 - Group Photo Analysis
"""
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from Module2.Media_Pipe__Service import MediaPipeService
    from ultralytics import YOLO
    import cv2
    import numpy as np
    
    print("✅ Module 2 test successful!")
    print("   All core dependencies can be imported")
    
    # Test YOLO model loading
    try:
        model = YOLO("yolov8n.pt")
        print("✅ YOLO model loaded successfully")
    except Exception as e:
        print(f"⚠️  YOLO model not found (will download on first run): {e}")
    
    # Test MediaPipe service
    mp_service = MediaPipeService()
    print("✅ MediaPipe service initialized")
    
    # Test with dummy image
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    mp_service.process_face(dummy_image)
    mp_service.process_pose(dummy_image)
    print("✅ Basic processing test passed")
    
except ImportError as e:
    print(f"❌ Module 2 test failed: {e}")
except Exception as e:
    print(f"❌ Module 2 test failed: {e}")
'''
    
    # Write test scripts
    with open("test_module1.py", "w") as f:
        f.write(module1_test)
    
    with open("test_module2.py", "w") as f:
        f.write(module2_test)
    
    print("✅ Test scripts created:")
    print("   • test_module1.py - Test Module 1 components")
    print("   • test_module2.py - Test Module 2 components")

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "=" * 60)
    print("🎉 Installation Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Activate your virtual environment:")
    print(f"   {get_activation_command()}")
    print("\n2. Test the installation:")
    print("   python test_module1.py")
    print("   python test_module2.py")
    print("\n3. Run Module 1 (Video Analysis):")
    print("   cd Module1")
    print("   python simple_script_for_analysis.py")
    print("\n4. Run Module 2 (Group Photo Analysis):")
    print("   cd Module2")
    print("   python simple_script_for_analysis.py")
    print("\n5. Launch Gradio Web Interface (Module 2):")
    print("   cd Module2")
    print("   python gradio_complete.py")
    print("\n📖 For detailed instructions, see: Research_Document_Module1_Module2.md")
    print("=" * 60)

def main():
    """Main installation function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check the error messages above.")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Some components may not be properly installed.")
        print("   Please check the error messages above.")
    
    # Check models
    check_models()
    
    # Create test scripts
    create_test_scripts()
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main() 