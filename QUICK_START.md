# Quick Start Guide - Appearance Rating and Group Synergy AI

## 🚀 Get Started in 5 Minutes

This guide will help you set up and run both modules quickly.

---

## Prerequisites

- **Python 3.8-3.11** (3.9 recommended)
- **8GB RAM** minimum
- **5GB free disk space**
- **Google Gemini API key** (for Module 2)

---

## Step 1: Automated Setup (Recommended)

Run the automated installation script:

```bash
python install_setup.py
```

This script will:
- ✅ Check Python version
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set up environment variables
- ✅ Verify installation
- ✅ Create test scripts

---

## Step 2: Manual Setup (Alternative)

If you prefer manual setup:

### 2.1 Create Virtual Environment
```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on macOS/Linux:
source venv/bin/activate
```

### 2.2 Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 Set Up API Key
Create `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

Get API key from: https://makersuite.google.com/app/apikey

---

## Step 3: Test Installation

Run the test scripts to verify everything works:

```bash
# Test Module 1
python test_module1.py

# Test Module 2
python test_module2.py
```

You should see ✅ success messages for both tests.

---

## Step 4: Run the Modules

### Module 1: Video Analysis

```bash
cd Module1
python simple_script_for_analysis.py
```

**What it does:**
- Analyzes video files for appearance metrics
- Provides real-time scoring
- Generates detailed reports

**Input:** Video file path
**Output:** 
- `video_results/video_analysis_results.json`
- `video_results/score_timeline.png`
- Real-time display

### Module 2: Group Photo Analysis

```bash
cd Module2
python simple_script_for_analysis.py
```

**What it does:**
- Analyzes group photos for collective synergy
- Detects multiple people and faces
- Evaluates color coordination
- Generates group summary

**Input:** Group photo path
**Output:**
- `results/group_summary.json`
- Individual person analysis
- Color coordination metrics

---

## Step 5: Web Interface (Module 2)

Launch the Gradio web interface:

```bash
cd Module2
python gradio_complete.py
```

Open your browser to: `http://localhost:7860`

**Features:**
- Drag & drop photo upload
- Real-time analysis
- Interactive results display
- Multiple analysis modes

---

## 🎯 Sample Usage

### Video Analysis Example
```bash
cd Module1
python simple_script_for_analysis.py
# Enter: v.mp4 (included in Module1)
```

### Group Photo Example
```bash
cd Module2
python simple_script_for_analysis.py
# Enter: path/to/your/group_photo.jpg
```

---

## 📊 Understanding Results

### Module 1 Metrics
- **Final Score**: Overall appearance (0.0-1.0)
- **Smile Score**: Facial expression quality (0.0-1.0)
- **Confidence Score**: Posture and body language (0.0-1.0)
- **Head Pose Score**: Head positioning (0.0-1.0)
- **Eye Contact**: Eye contact detection (0 or 1)

### Module 2 Metrics
- **Group Synergy Score**: Overall coordination (0-100)
- **Color Similarity**: Clothing coordination (0-100)
- **Posture Scores**: Average group posture (0-100)
- **Eye Contact**: Percentage with eye contact (0-100)
- **Active Smiles**: Percentage smiling (0-100)

---

## 🔧 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. CUDA/GPU Issues**
```bash
# Install CPU version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**3. API Key Issues**
```bash
# Check .env file
cat .env
# Ensure GEMINI_API_KEY is set
```

**4. Model Download Issues**
```bash
# Models download automatically on first run
# If issues persist, check internet connection
```

### Performance Tips

**For Better Performance:**
- Use GPU if available
- Close other applications
- Use smaller video files for testing
- Ensure good lighting in videos/photos

**For CPU Users:**
- Use smaller YOLO models
- Process shorter videos
- Reduce image resolution if needed

---

## 📁 File Structure

```
Appearence_Rating_Group_Computer_Vision_Project/
├── Module1/                          # Video Analysis
│   ├── simple_script_for_analysis.py # Client script
│   ├── testing_media_pipe.py         # MediaPipe service
│   ├── Media_Pipe__Service.py        # Core service
│   └── video_results/                # Output folder
├── Module2/                          # Group Photo Analysis
│   ├── simple_script_for_analysis.py # Client script
│   ├── gradio_complete.py            # Web interface
│   ├── main.py                       # Pipeline
│   ├── Media_Pipe__Service.py        # Core service
│   └── results/                      # Output folder
├── requirements.txt                  # Dependencies
├── install_setup.py                  # Auto-installer
├── test_module1.py                   # Module 1 test
├── test_module2.py                   # Module 2 test
└── .env                              # API keys
```

---

## 🆘 Need Help?

1. **Check the full documentation**: `Research_Document_Module1_Module2.md`
2. **Run test scripts**: `python test_module1.py` or `python test_module2.py`
3. **Check error messages**: Look for specific error details
4. **Verify installation**: Ensure all dependencies are installed

---

## 🎉 Success!

You're now ready to use both modules! 

- **Module 1**: Perfect for video analysis and individual appearance assessment
- **Module 2**: Ideal for group photos and collective synergy analysis

Both modules provide comprehensive analysis with detailed metrics and visual feedback.

---

**Next Steps:**
- Try different videos and photos
- Experiment with the web interface
- Explore the detailed documentation
- Customize analysis parameters

Happy analyzing! 🎥📸 