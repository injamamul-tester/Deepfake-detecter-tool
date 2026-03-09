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
