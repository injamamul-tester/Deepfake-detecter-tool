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
    st.title("Deepfake Detection Tool")
    st.write("Upload an image, video, or audio file to detect deepfake content.")
    option = st.selectbox("Select content type", ["Image", "Video", "Audio"])
    uploaded_file = st.file_uploader("Upload file", type=["jpg", "png", "mp4", "avi", "wav", "mp3"])
    if uploaded_file:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        if option == "Image":
            result, confidence = detect_image_deepfake(temp_path)
        elif option == "Video":
            result, confidence = detect_video_deepfake(temp_path)
        else:
            result, confidence = detect_audio_deepfake(temp_path)
        st.success(f"Result: {'Deepfake' if result else 'Real'} | Confidence: {confidence:.2f}")
        os.remove(temp_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "web":
        run_streamlit()
    else:
        cli()
