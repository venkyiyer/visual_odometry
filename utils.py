import streamlit as st
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Given a video path, extract frames and save them to a directory
def extract_frames(video_path, output_dir, progress_callback=None):
    try:
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0
        saved_paths = []
        saved_ids = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_paths.append(frame_path)
            saved_ids.append(frame_idx)
            frame_idx += 1

            if progress_callback:
                progress_callback(frame_idx, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        cap.release()
    except Exception as e:
        st.error(f"Error extracting frames from: {frame_path}")

    return saved_paths, saved_ids

# Load poses from a file
def load_poses(pose_file):
    poses = []
    try:
        with open(pose_file, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                pose = np.eye(4)
                pose[:3, :4] = np.array(vals).reshape(3, 4)
                poses.append(pose)
    except Exception as e:
        st.error(f"Error loading poses from file: {pose_file}\n{e}")
    
    return poses

# Plot the 3D trajectory from poses
def plot_trajectory(poses):
    try:
        if not poses:
            st.warning("No poses to plot.")
            return
        xyz = np.array([pose[:3, 3] for pose in poses])
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='.', markersize= 5, linewidth =3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Visual Odometry Trajectory')
        st.pyplot(fig)
        st.success("Trajectory plotted successfully.")
    except Exception as e:
        st.error(f"Error plotting trajectory: {e}")
        return None