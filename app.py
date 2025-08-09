import streamlit as st
import shutil
import os
import tempfile
import cv2
from utils import extract_frames, plot_trajectory, load_poses
from examples.pose_estimation.run_cycle_pose import process_single_sequence

st.title("3D Visual Odometry")
st.markdown("Upload a video file for 3D camera trajectorry...")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.success("✅ Video uploaded successfully.")
    
    # st.info("⏳ Extracting frames from video...")
    extract_progress = st.progress(0, text="Extracting frames from the video...")

    def extract_progress_callback(current, total):
        percent = int(current / total * 100) if total else 0
        extract_progress.progress(percent, text=f"Extracting frames... {percent}%")
    frames, frame_ids = extract_frames(video_path, output_dir="extracted_frames", progress_callback=extract_progress_callback)
    extract_progress.empty()
    st.success(f"Extracted {len(frames)} frames.")

    with tempfile.TemporaryDirectory() as frames_dir:
        frame_paths = []
        for idx, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
            # cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    # print("Frame paths:", frame_paths)

    # st.info("🔍 Estimating poses...")
    progress_bar = st.progress(0, text="Estimating poses...")

    def progress_indicator(progress, total):
        percent = int(progress / total * 100)
        progress_bar.progress(percent, text=f"Estimating poses... {percent}%")
    
    poses, processing_time = process_single_sequence("extracted_frames","est_poses.txt", progress_callback=progress_indicator)
    progress_bar.empty()
    st.info(f"Time taken to estimate poses: {processing_time:.2f} seconds")

    st.info("📈 Rendering 3D trajectory...")
    read_poses = load_poses("est_poses.txt")
    plot_trajectory(poses)

    
    shutil.rmtree("extracted_frames", ignore_errors=True)
    os.remove("est_poses.txt")