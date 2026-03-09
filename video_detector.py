"""
Video Deepfake Detection Module
Extracts frames, detects faces, aggregates image results
"""
import cv2
from image_detector import detect_image_deepfake
from utils.video_utils import extract_frames

def detect_video_deepfake(video_path):
    """
    Detects deepfake in a video file.
    Returns (is_deepfake, confidence_score)
    """
    frames = extract_frames(video_path, max_frames=20)
    scores = []
    for frame in frames:
        temp_path = 'temp_frame.jpg'
        cv2.imwrite(temp_path, frame)
        is_fake, prob = detect_image_deepfake(temp_path)
        scores.append(prob)
    avg_score = sum(scores) / len(scores) if scores else 0.0
    is_deepfake = avg_score > 0.5
    return is_deepfake, avg_score
