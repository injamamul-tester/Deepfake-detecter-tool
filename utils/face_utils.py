"""
Face detection utility using dlib or mediapipe
"""
import cv2
import mediapipe as mp

def detect_face(img):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.detections:
            # Get bounding box of first detected face
            bbox = results.detections[0].location_data.relative_bounding_box
            h, w, _ = img.shape
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            return img[y1:y2, x1:x2]
    return None

# Facial landmark detection using mediapipe
def detect_landmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0].landmark
    return None

# Eye blink detection (simple heuristic)
def detect_eye_blink(landmarks):
    # Use mediapipe landmark indices for left/right eye
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [263, 387, 385, 362, 380, 373]
    def eye_aspect_ratio(eye_points):
        # Calculate vertical/horizontal distances
        import math
        v1 = math.dist([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y], [landmarks[eye_points[5]].x, landmarks[eye_points[5]].y])
        v2 = math.dist([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y], [landmarks[eye_points[4]].x, landmarks[eye_points[4]].y])
        h = math.dist([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y], [landmarks[eye_points[3]].x, landmarks[eye_points[3]].y])
        return (v1 + v2) / (2.0 * h)
    left_ear = eye_aspect_ratio(LEFT_EYE)
    right_ear = eye_aspect_ratio(RIGHT_EYE)
    # Threshold for blink (empirical)
    blink = left_ear < 0.21 or right_ear < 0.21
    return blink
