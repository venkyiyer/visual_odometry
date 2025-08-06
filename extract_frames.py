import streamlit as st
import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

def extract_frames(video_path, output_dir, progress_callback=None):
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
    print(f"Extracted {frame_idx} frames to {output_dir}")
    
    return saved_paths, saved_ids

# extract_frames(video_path="/home/venkys/sanofi_videos/0c35ad30-14c3-4ab9-b60b-b421e568a921.avi", 
#                output_dir="/home/venkys/sanofi_videos/sanofi_frames")

def load_poses(pose_file):
    poses = []
    with open(pose_file, 'r') as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            pose = np.eye(4)
            pose[:3, :4] = np.array(vals).reshape(3, 4)
            poses.append(pose)
    return poses

def plot_trajectory(poses):
    xyz = np.array([pose[:3, 3] for pose in poses])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Visual Odometry Trajectory')
    # plt.show()
    # plt.savefig("/home/venkys/sanofi_videos/trajectory.png")
    # print("Trajectory plot saved as trajectory.png")
    st.pyplot(fig)  # Display in Streamlit UI
# Example usage:
# poses = load_poses('/home/venkys/sanofi_videos/seq_poses.txt')
# plot_trajectory(poses)