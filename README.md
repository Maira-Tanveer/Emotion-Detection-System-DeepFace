# Real-Time Emotion Detection with DeepFace

A Python application that performs real-time emotion detection using your webcam and the DeepFace library. The application analyzes facial expressions to detect emotions and categorizes them into mood groups for enhanced emotional state tracking.

## Features

- **Real-time emotion detection** from webcam feed
- **Seven emotion categories**: angry, disgust, fear, happy, sad, surprise, neutral
- **Mood categorization**: Groups emotions into Negative, Neutral, and Positive moods
- **Emotion history tracking**: Maintains last 10 emotion predictions for stable mood assessment
- **Performance optimization**: Processes every nth frame for better performance
- **Confidence thresholding**: Only displays emotions above 40% confidence
- **FPS monitoring**: Real-time and average FPS display
- **Face detection visualization**: Draws rectangles around detected faces
- **Detailed emotion scores**: Shows confidence percentages for all emotions

## Requirements

### Dependencies

```bash
pip install opencv-python
pip install deepface
pip install numpy
```

### System Requirements

- Python 3.7 or higher
- Webcam/Camera device
- Sufficient processing power for real-time analysis

## Installation

1. Clone or download the script
2. Install required dependencies:
   ```bash
   pip install opencv-python deepface numpy
   ```
3. Ensure your webcam is connected and accessible

## Usage

### Running the Application

```bash
python emotion_detection.py
```

### Controls

- **Press 'q'** to quit the application
- **Ctrl+C** for emergency exit

### Configuration Options

You can modify these parameters in the code:

```python
# Performance settings
skip_frames = 2  # Process every 2nd frame (increase for better performance)
confidence_threshold = 0.40  # Minimum confidence to display emotion (0.0-1.0)

# Detection backend options
detector_backend = 'opencv'  # Options: opencv, ssd, mtcnn, retinaface, mediapipe, dlib

# History tracking
emotion_history = deque(maxlen=10)  # Number of recent emotions to track

# Safety timeout
max_run_time = 300  # Maximum runtime in seconds (5 minutes)
```

## Emotion Categories

### Primary Emotions
- **Angry**: Frustration, irritation, rage
- **Disgust**: Revulsion, distaste
- **Fear**: Anxiety, worry, terror
- **Happy**: Joy, contentment, pleasure
- **Sad**: Sorrow, melancholy, grief
- **Surprise**: Astonishment, amazement
- **Neutral**: Calm, expressionless state

### Mood Classification
- **Negative Mood**: angry, disgust, fear, sad
- **Positive Mood**: happy, surprise
- **Neutral Mood**: neutral

## Performance Features

### Optimization Techniques
- **Frame skipping**: Processes every 2nd frame by default
- **Confidence filtering**: Only displays high-confidence predictions
- **Efficient face detection**: Uses OpenCV backend for speed
- **Memory management**: Limited emotion history buffer

### Performance Monitoring
- Real-time FPS counter
- Average FPS calculation
- Processing ratio display (e.g., "1 in 2 frames")

## Troubleshooting

### Common Issues

**Webcam not opening:**
```
Error: Could not open webcam
```
- Check if webcam is connected
- Ensure no other applications are using the camera
- Try changing the camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

**Low performance/choppy video:**
- Increase `skip_frames` value (process fewer frames)
- Lower the confidence threshold
- Close other resource-intensive applications

**No face detected:**
- Ensure adequate lighting
- Position face clearly in frame
- Check if `enforce_detection=False` is set

**Analysis errors:**
- Update DeepFace library: `pip install --upgrade deepface`
- Check internet connection (first-time model download)
- Verify all dependencies are installed correctly

### Performance Tips

1. **Better Performance:**
   - Increase `skip_frames` to 3 or 4
   - Use `detector_backend = 'opencv'` for fastest detection
   - Close unnecessary applications

2. **Better Accuracy:**
   - Decrease `skip_frames` to 1
   - Use `detector_backend = 'mtcnn'` or `'retinaface'`
   - Ensure good lighting conditions
   - Keep face centered and well-visible

## Technical Details

### Architecture
- **Computer Vision**: OpenCV for video capture and display
- **Emotion Analysis**: DeepFace library with pre-trained models
- **Face Detection**: Configurable backends (OpenCV, MTCNN, etc.)
- **Data Processing**: NumPy for numerical operations

### Output Information
- Primary emotion with confidence percentage
- Overall mood classification
- All emotion scores breakdown
- Real-time performance metrics
- Face bounding boxes

### Safety Features
- Maximum runtime limit (5 minutes default)
- Graceful error handling
- Resource cleanup on exit
- Multiple exit methods (keyboard interrupt, 'q' key)

## File Structure

```
emotion_detection.py    # Main application script
README.md              # This documentation
```

## Contributing

Feel free to fork and improve this project. Some areas for enhancement:

- GUI interface for parameter adjustment
- Emotion logging and analytics
- Multiple face tracking
- Integration with other applications
- Additional emotion categories
- Video file processing support

## License

This project uses the DeepFace library and OpenCV. Please check their respective licenses for commercial use.

---

**Note**: First-time usage may require internet connection for DeepFace to download pre-trained models automatically.
