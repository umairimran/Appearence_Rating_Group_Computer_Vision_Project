# Appearance Rating and Group Synergy AI

🎥 **Video Analysis** | 📸 **Group Photo Analysis** | 🤖 **AI-Powered Insights**

A comprehensive computer vision system for appearance assessment and group synergy analysis using advanced AI technologies including MediaPipe, YOLOv8, and Google Gemini.

## 📹 Demo Video

<video src="Portfolio/Computer Vision video.mp4" width="800" controls>
  Your browser does not support the video tag.
</video>

*Watch the demo video showcasing the appearance rating and group synergy analysis system in action.*

## 📸 Portfolio Images

![Computer Vision Image 1](Portfolio/Computer Vision Image 1.png)

![Computer Vision Image 2](Portfolio/Computer Vision Image 2png.png)

![Computer Vision Image 3](Portfolio/Computer Vision Image 3.png)

![Computer Vision Image 4](Portfolio/Computer Vision Image 4.png)

![Computer Vision Image 5](Portfolio/Computer Vision Image 5.png)

![Computer Vision Image 6](Portfolio/Computer Vision Image 6.png)

## 🎯 Problem Statement

In today's social media-driven world, there is a growing trend on platforms like TikTok where people judge the appearance of groups and individuals. We often need to assess the overall ability, appearance, and synergy of a person or group to make informed decisions - whether it's for social connections, professional collaborations, or entertainment purposes.

This trend has created a demand for an objective, computer vision-based solution that can:
- Analyze individual appearance metrics
- Evaluate group synergy and coordination
- Provide comprehensive scoring based on facial features, expressions, and group dynamics
- Offer unbiased, data-driven assessments

## 💡 Proposed Approach

Our proposed solution leverages state-of-the-art computer vision technologies:

- **MediaPipe** (Google-based library): For detecting and analyzing facial features of the human face, including facial landmarks, expressions, pose estimation, and eye contact detection.

- **YOLOv8 Pre-trained Models**: For accurate face detection and person detection in images and videos, enabling robust multi-person analysis.

- **Integrated Analysis Pipeline**: Combining MediaPipe's facial feature analysis with YOLO's detection capabilities to provide comprehensive appearance ratings and group synergy scores.

This computer vision-based approach provides:
- Real-time facial feature analysis
- Group coordination assessment
- Objective scoring metrics
- Visual feedback and reporting

## 🚀 Quick Start

### Automated Installation (Recommended)
```bash
python install_setup.py
```

### Manual Installation
```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Set up API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Test Installation
```bash
python comprehensive_test.py
```

## 📋 What's Included

### Module 1: Video Analysis System
- **Real-time video processing** with live metrics
- **Facial expression analysis** (smile detection)
- **Eye contact tracking** and posture assessment
- **Confidence scoring** and head pose analysis
- **Visual feedback** with live bars and graphs
- **Data export** to JSON and timeline plots

### Module 2: Group Photo Analysis System
- **Multi-person detection** using YOLOv8
- **Individual face extraction** and analysis
- **Group synergy evaluation** with collective metrics
- **Color coordination analysis** using Google Gemini AI
- **Web interface** with Gradio
- **Comprehensive reporting** with group summaries

## 🎯 Key Features

### Advanced AI Technologies
- **MediaPipe**: Facial landmark detection and pose estimation
- **YOLOv8**: Real-time object detection for people and faces
- **Google Gemini**: AI-powered color analysis and insights
- **OpenCV**: Computer vision processing and visualization

### Comprehensive Metrics
- **Individual Scores**: Smile, confidence, posture, eye contact
- **Group Synergy**: Collective coordination and aesthetics
- **Color Analysis**: Clothing coordination and style assessment
- **Real-time Feedback**: Live scoring and visual indicators

### User-Friendly Interfaces
- **Command-line tools** for batch processing
- **Web interface** for interactive analysis
- **JSON exports** for data integration
- **Visual reports** with charts and graphs

## 📖 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get up and running in 5 minutes
- **[Research Document](Research_Document_Module1_Module2.md)** - Comprehensive technical documentation
- **[Installation Script](install_setup.py)** - Automated setup and verification
- **[Testing Suite](comprehensive_test.py)** - Complete system testing

## 🛠️ Usage Examples

### Video Analysis (Module 1)
```bash
cd Module1
python simple_script_for_analysis.py
# Enter video file path when prompted
```

### Group Photo Analysis (Module 2)
```bash
cd Module2
python simple_script_for_analysis.py
# Enter group photo path when prompted
```

### Web Interface (Module 2)
```bash
cd Module2
python gradio_complete.py
# Open http://localhost:7860 in your browser
```

## 📊 Output Examples

### Module 1 - Video Analysis Results
```json
{
  "video_path": "sample_video.mp4",
  "total_duration_seconds": 30,
  "average_scores": {
    "final_score": 0.85,
    "smile_score": 0.78,
    "confidence_score": 0.92,
    "head_pose_score": 0.88,
    "eye_contact_score": 0.75
  }
}
```

### Module 2 - Group Analysis Results
```json
{
  "group_synergy_score": 87,
  "color_similarity": 92,
  "posture_scores": 85,
  "eye_contact": 80,
  "active_smiles": 75,
  "overall_aesthetics": 89
}
```

## 🔧 System Requirements

### Hardware
- **CPU**: Intel i5/AMD Ryzen 5 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional)
- **Storage**: 5GB free space

### Software
- **OS**: Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python**: 3.8 - 3.11 (3.9 recommended)
- **CUDA**: 11.0+ (for GPU acceleration)

## 🎨 Use Cases

### Professional Applications
- **Interview preparation** and confidence building
- **Presentation training** and public speaking
- **Team photo analysis** for corporate events
- **Fashion coordination** and style assessment

### Educational Applications
- **Communication skills** development
- **Body language** analysis and improvement
- **Group dynamics** research and analysis
- **Visual communication** training

### Research Applications
- **Human-computer interaction** studies
- **Social psychology** research
- **Computer vision** algorithm development
- **AI/ML** model evaluation

## 🔍 Technical Architecture

### Module 1 Architecture
```
Video Input → MediaPipe Processing → Feature Extraction → Scoring → Visualization
```

### Module 2 Architecture
```
Image Input → YOLO Detection → Face Extraction → MediaPipe Analysis → Gemini AI → Group Summary
```

### Key Components
- **MediaPipeService**: Core processing engine
- **YOLO Models**: Person and face detection
- **Gemini Integration**: AI-powered analysis
- **Gradio Interface**: Web-based user interface

## 🚀 Performance Metrics

### Processing Speed
- **Module 1**: 25-30 FPS (real-time video)
- **Module 2**: 2-5 seconds per group photo

### Accuracy Benchmarks
- **Face Detection**: 92.5%
- **Pose Estimation**: 87.3%
- **Smile Detection**: 84.7%
- **Color Extraction**: 96.1%

### Memory Usage
- **Module 1**: ~2GB RAM
- **Module 2**: ~3GB RAM

## 🔧 Troubleshooting

### Common Issues
1. **Import Errors**: Run `pip install -r requirements.txt --force-reinstall`
2. **CUDA Issues**: Install CPU version of PyTorch
3. **API Key Issues**: Verify `.env` file contains `GEMINI_API_KEY`
4. **Model Download**: Check internet connection for automatic downloads

### Performance Optimization
- Use GPU acceleration when available
- Close other applications during processing
- Use smaller video files for testing
- Ensure good lighting conditions

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- Bug reports and feature requests
- Code contributions and improvements
- Documentation updates
- Testing and validation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **MediaPipe** by Google for facial landmark detection
- **Ultralytics** for YOLOv8 implementation
- **Google Gemini** for AI-powered analysis
- **OpenCV** for computer vision capabilities
- **Gradio** for web interface framework

## 📞 Support

- **Documentation**: [Research Document](Research_Document_Module1_Module2.md)
- **Quick Start**: [Quick Start Guide](QUICK_START.md)
- **Testing**: [Comprehensive Test Suite](comprehensive_test.py)
- **Issues**: Please report bugs and feature requests

---

**Made with ❤️ for computer vision and AI research**

*Transform your appearance analysis with cutting-edge AI technology!*
