import cv2
import numpy as np
import time
from deepface import DeepFace
from collections import deque
import os
import sys

# Define emotion categories
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Define mood categories
moods = ['Negative', 'Neutral', 'Positive']

# Mapping from emotions to mood categories
emotion_to_mood = {
    'angry': 'Negative',
    'disgust': 'Negative',
    'fear': 'Negative',
    'sad': 'Negative',
    'happy': 'Positive',
    'surprise': 'Positive',
    'neutral': 'Neutral'
}

# For face detection speed
detector_backend = 'opencv'  # Options: opencv, ssd, mtcnn, retinaface, mediapipe, dlib

print("Starting emotion detection using DeepFace...")

# Initialize webcam
print("Starting webcam...")
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit(1)

print("Starting real-time emotion detection. Press 'q' to quit.")

# Initialize variables for emotion tracking
emotion_history = deque(maxlen=10)  # Store last 10 emotion predictions
for _ in range(10):
    emotion_history.append('neutral')  # Initialize with neutral

# For FPS calculation
prev_frame_time = 0
frame_count = 0
total_fps = 0
skip_frames = 2  # Process every n frames for better performance
confidence_threshold = 0.40  # Minimum confidence to display emotion

# Set a max run time in case quit doesn't work (in seconds)
max_run_time = 300  # 5 minutes
start_time = time.time()

try:
    while True:
        # Check if we've exceeded max run time
        if time.time() - start_time > max_run_time:
            print(f"Maximum run time ({max_run_time}s) exceeded. Exiting...")
            break
            
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_frame_time) if prev_frame_time > 0 else 0
        prev_frame_time = current_time
        
        # Skip processing some frames for better performance
        frame_count += 1
        if frame_count % skip_frames != 0:
            # Display current emotion from history
            if emotion_history:
                current_emotion = max(set(emotion_history), key=emotion_history.count)
                cv2.putText(frame, f"Emotion: {current_emotion}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calculate mood based on recent emotions
                mood_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
                for emotion in emotion_history:
                    if emotion in emotion_to_mood:
                        mood = emotion_to_mood[emotion]
                        mood_counts[mood] += 1
                
                current_mood = max(mood_counts, key=mood_counts.get)
                cv2.putText(frame, f"Mood: {current_mood}", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display quit instructions
            cv2.putText(frame, "Press 'q' to quit", 
                      (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Emotion Detection (DeepFace)', frame)
            
            # Check for 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quit key pressed. Exiting...")
                break
            
            continue
        
        # Analyze the frame with DeepFace
        try:
            # Analyze the full frame
            analysis = DeepFace.analyze(
                frame, 
                actions=['emotion'],
                enforce_detection=False,  # Don't enforce face detection to handle cases where face might not be perfectly detected
                detector_backend=detector_backend
            )
            
            # If no face was found
            if len(analysis) == 0:
                cv2.putText(frame, "No face detected", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Process each detected face (usually just one)
                for i, face_analysis in enumerate(analysis):
                    # Extract emotion information
                    emotion_label = face_analysis['dominant_emotion']
                    emotion_scores = face_analysis['emotion']
                    emotion_confidence = emotion_scores[emotion_label] / 100.0
                    
                    # Add to emotion history
                    emotion_history.append(emotion_label)
                    
                    # Get face rectangle coordinates
                    face_rect = face_analysis.get('region', {})
                    x, y, w, h = (
                        face_rect.get('x', 0),
                        face_rect.get('y', 0),
                        face_rect.get('w', 0),
                        face_rect.get('h', 0)
                    )
                    
                    # Draw rectangle around face if coordinates are available
                    if w > 0 and h > 0:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display emotion and confidence if above threshold
                    if emotion_confidence > confidence_threshold:
                        emotion_text = f"Emotion: {emotion_label} ({emotion_confidence*100:.1f}%)"
                        cv2.putText(frame, emotion_text, 
                                  (x if w > 0 else 10, y-10 if h > 0 else 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Calculate mood based on recent emotions
                        mood_counts = {'Negative': 0, 'Neutral': 0, 'Positive': 0}
                        for emotion in emotion_history:
                            if emotion in emotion_to_mood:
                                mood = emotion_to_mood[emotion]
                                mood_counts[mood] += 1
                        
                        current_mood = max(mood_counts, key=mood_counts.get)
                        mood_text = f"Mood: {current_mood}"
                        cv2.putText(frame, mood_text, 
                                  (x if w > 0 else 10, y+h+25 if h > 0 else 90), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Low confidence", 
                                  (x if w > 0 else 10, y-10 if h > 0 else 60), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Display all emotion scores (optional)
                    y_offset = y+h+60 if h > 0 else 120
                    for emotion, score in emotion_scores.items():
                        score_text = f"{emotion}: {score:.1f}%"
                        cv2.putText(frame, score_text, 
                                  (x if w > 0 else 10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        y_offset += 20
        
        except Exception as e:
            print(f"Error in analysis: {e}")
            cv2.putText(frame, "Analysis error", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate and track FPS
        total_fps += fps
        avg_fps = total_fps / (frame_count // skip_frames) if frame_count // skip_frames > 0 else 0
        
        # Display FPS
        cv2.putText(frame, f"FPS: {int(fps)} (Avg: {int(avg_fps)})", 
                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display processing info
        cv2.putText(frame, f"Processing: 1 in {skip_frames} frames", 
                  (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display quit instructions
        cv2.putText(frame, "Press 'q' to quit", 
                  (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Emotion Detection (DeepFace)', frame)
        
        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit key pressed. Exiting...")
            break

except KeyboardInterrupt:
    print("Interrupted by user (Ctrl+C). Exiting...")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    # Release the capture and close windows
    print("Cleaning up resources...")
    cap.release()
    cv2.destroyAllWindows()
    # Try more aggressive window closing
    for i in range(5):
        cv2.waitKey(1)
    
    print(f"Emotion detection stopped. Average FPS: {avg_fps:.2f}")

