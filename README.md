# Deepfake Detection Tool

A Python tool to detect deepfake content in images, videos, and audio using modern AI/ML libraries.

## Features
- Image deepfake detection (CNN, face detection)
- Video deepfake detection (frame extraction, aggregation)
- Audio deepfake detection (MFCC, neural network)
- CLI (Click) and Streamlit web interface
- GPU support (PyTorch/TensorFlow)

## Project Structure
```
main.py
image_detector.py
video_detector.py
audio_detector.py
models/
utils/
requirements.txt
```

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/injamamul-tester/Deepfake-detecter-tool.git
   cd Deepfake-detecter-tool
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Example Datasets
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [DFDC](https://www.kaggle.com/c/deepfake-detection-challenge)
- [ASVspoof](https://www.asvspoof.org/)

## Usage
### CLI
- Image:
  ```bash
  python main.py image path/to/image.jpg
  ```
- Video:
  ```bash
  python main.py video path/to/video.mp4
  ```
- Audio:
  ```bash
  python main.py audio path/to/audio.wav
  ```

### Web UI
- Start Streamlit app:
  ```bash
  streamlit run main.py web
  ```

## Pretrained Models
- Place your pretrained weights in `models/image_deepfake.pth` and `models/audio_deepfake.pth`.
- The tool will use default weights if custom weights are not found.

## Output
- Real or Deepfake
- Confidence score (0.0-1.0)

## Notes
- Code is optimized for GPU if available.
- All modules are well-commented and modular.

## License
MIT
