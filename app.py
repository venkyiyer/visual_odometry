import streamlit as st
import os
import tempfile
import cv2
from extract_frames import extract_frames, plot_trajectory, load_poses
from examples.pose_estimation.run_cycle_pose import process_single_sequence

st.title("3D Visual Odometry")
st.markdown("Upload a video file for 3D camera trajectorry...")

uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    st.success("✅ Video uploaded successfully.")
    
    st.info("⏳ Extracting frames from video...")
    frames, frame_ids = extract_frames(video_path, output_dir="extracted_frames")
    st.success(f"Extracted {len(frames)} frames.")

    with tempfile.TemporaryDirectory() as frames_dir:
        frame_paths = []
        for idx, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"frame_{idx:05d}.jpg")
            # cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    # print("Frame paths:", frame_paths)

    st.info("🔍 Estimating poses...")
    poses, processing_time = process_single_sequence("extracted_frames","est_poses.txt")
    st.info(f"Time taken: {processing_time:.2f} seconds")

    st.info("📈 Rendering 3D trajectory...")
    read_poses = load_poses("est_poses.txt")
    plot_trajectory(poses)

    
    # os.remove(video_path)