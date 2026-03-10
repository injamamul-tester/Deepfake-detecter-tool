"""
Video Deepfake Detection Module
Extracts frames, detects faces, aggregates image results
"""
import cv2
from image_detector import detect_image_deepfake
from utils.video_utils import extract_frames
from utils.face_utils import detect_landmarks, detect_eye_blink

def detect_video_deepfake(video_path):
    """
    Detects deepfake in a video file.
    Returns (is_deepfake, confidence_score)
    """
    frames = extract_frames(video_path, max_frames=20)
    scores = []
    blink_count = 0
    for frame in frames:
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, frame)
        is_fake, prob = detect_image_deepfake(temp_path)
        scores.append(prob)
        # Facial landmark and blink analysis
        landmarks = detect_landmarks(frame)
        if landmarks:
            if detect_eye_blink(landmarks):
                blink_count += 1
    avg_score = sum(scores) / len(scores) if scores else 0.0
    # Heuristic: abnormal blink count (too low or too high) may indicate fake
    abnormal_blink = blink_count < 2 or blink_count > 15
    is_deepfake = avg_score > 0.5 or abnormal_blink
    return is_deepfake, avg_score
