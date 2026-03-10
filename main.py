"""
Deepfake Detection Tool
Supports CLI (Click) and Web UI (Streamlit)
"""

import click
import streamlit as st
import torch
import os
from image_detector import detect_image_deepfake
from video_detector import detect_video_deepfake
from audio_detector import detect_audio_deepfake

# CLI Interface
@click.group()
def cli():
    """Deepfake Detection CLI"""
    pass

@cli.command()
@click.argument('image_path')
def image(image_path):
    """Detect deepfake in an image file"""
    result, confidence = detect_image_deepfake(image_path)
    click.echo(f"Result: {'Deepfake' if result else 'Real'} | Confidence: {confidence:.2f}")

@cli.command()
@click.argument('video_path')
def video(video_path):
    """Detect deepfake in a video file"""
    result, confidence = detect_video_deepfake(video_path)
    click.echo(f"Result: {'Deepfake' if result else 'Real'} | Confidence: {confidence:.2f}")

@cli.command()
@click.argument('audio_path')
def audio(audio_path):
    """Detect deepfake in an audio file"""
    result, confidence = detect_audio_deepfake(audio_path)
    click.echo(f"Result: {'Deepfake' if result else 'Real'} | Confidence: {confidence:.2f}")

# Streamlit Web UI

def run_streamlit():
    st.markdown("""
    <style>
    .result-box {
        padding: 1em;
        border-radius: 8px;
        margin-top: 1em;
        font-size: 1.2em;
        font-weight: bold;
    }
    .deepfake {background: #ffcccc; color: #b30000;}
    .real {background: #ccffcc; color: #006600;}
    </style>
    """, unsafe_allow_html=True)
    st.header("Detect Deepfake Content")
    st.write("Select the type of content and upload your file. The tool will analyze and display the result with a confidence score.")
    option = st.radio("Content Type", ["Image", "Video", "Audio", "Real-time Video"], horizontal=True)
    if option == "Real-time Video":
        st.write("Use your webcam for live deepfake detection.")
        try:
            import cv2
            import numpy as np
            run = st.button("Start Webcam Detection")
            if run:
                cap = cv2.VideoCapture(0)
                stframe = st.empty()
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Resize for speed
                    small_frame = cv2.resize(frame, (320, 240))
                    # Run deepfake detection on frame
                    is_fake, prob = detect_image_deepfake("temp_frame.jpg") if cv2.imwrite("temp_frame.jpg", small_frame) else (False, 0.0)
                    # Visual feedback
                    label = f"{'Deepfake' if is_fake else 'Real'} | Confidence: {prob:.2f}"
                    color = (0, 0, 255) if is_fake else (0, 255, 0)
                    cv2.putText(small_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    stframe.image(small_frame, channels="BGR")
                    if st.button("Stop Webcam Detection"):
                        break
                cap.release()
                cv2.destroyAllWindows()
        except Exception as e:
            st.error(f"Webcam not available or OpenCV error: {e}")
    else:
        uploaded_file = st.file_uploader("Upload file", type=["jpg", "png", "mp4", "avi", "wav", "mp3"])
        if uploaded_file:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            with st.spinner("Analyzing..."):
                if option == "Image":
                    result, confidence = detect_image_deepfake(temp_path)
                elif option == "Video":
                    result, confidence = detect_video_deepfake(temp_path)
                else:
                    result, confidence = detect_audio_deepfake(temp_path)
            result_text = f"{'Deepfake' if result else 'Real'}"
            score_text = f"Confidence: {confidence:.2f}"
            box_class = "deepfake" if result else "real"
            st.markdown(f'<div class="result-box {box_class}">{result_text}<br>{score_text}</div>', unsafe_allow_html=True)
            if option == "Image" and not result:
                st.image(temp_path, caption="Uploaded Image", use_column_width=True)
            os.remove(temp_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        run_streamlit()
    else:
        cli()
