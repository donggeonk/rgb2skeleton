"""
python visualize_skeleton.py --data_input output/skeleton.npz --video_output visualization/skeleton_animation.mp4
python visualize_skeleton.py --data_input output/walking_skeleton.npz --video_output visualization/walking_visual.mp4
"""

import numpy as np
import argparse
import mediapipe as mp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import cv2
import os
import shutil
from tqdm import tqdm

# Use a non-interactive backend to prevent plot windows from popping up
matplotlib.use('Agg')

def visualize_skeleton_video(skeleton_data_path: str, output_path: str, fps: int = 30):
    """
    Loads 3D skeleton data and creates a video of the animated 3D skeleton.

    Args:
        skeleton_data_path (str): Path to the .npz file containing skeleton data.
        output_path (str): Path to save the output MP4 video.
        fps (int): Frames per second for the output video.
    """
    # --- 1. Load the 3D Skeleton Data ---
    try:
        data = np.load(skeleton_data_path)
        skeleton_sequence = data['skeleton']
    except FileNotFoundError:
        print(f"Error: Skeleton data file not found at {skeleton_data_path}")
        return

    # --- 2. Create a temporary directory to store frame images ---
    temp_dir = "temp_frames"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"Generating {len(skeleton_sequence)} frames for the video...")

    # Get MediaPipe's official connections to draw the skeleton
    skeleton_connections = mp.solutions.pose.POSE_CONNECTIONS

    # --- 3. Generate and Save a 3D Plot for Each Frame ---
    for frame_idx in tqdm(range(len(skeleton_sequence)), desc="Generating Frames"):
        world_landmarks = skeleton_sequence[frame_idx]

        if not np.isnan(world_landmarks).any():
            fig = plt.figure(figsize=(8, 8), dpi=100)
            ax = fig.add_subplot(111, projection='3d')

            # Scatter plot all the landmarks
            ax.scatter(world_landmarks[:, 0], world_landmarks[:, 2], -world_landmarks[:, 1], c='red', marker='o')

            # Draw the skeleton connections
            for connection in skeleton_connections:
                start_idx, end_idx = connection
                ax.plot([world_landmarks[start_idx, 0], world_landmarks[end_idx, 0]],
                        [world_landmarks[start_idx, 2], world_landmarks[end_idx, 2]],
                        [-world_landmarks[start_idx, 1], -world_landmarks[end_idx, 1]], 'cyan')

            # Set consistent plot properties for a stable video
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.view_init(elev=20., azim=-90) # A good viewing angle for human motion
            
            # Save the figure to the temporary directory
            plt.savefig(os.path.join(temp_dir, f"frame_{frame_idx:05d}.png"))
            plt.close(fig)

    # --- 4. Compile the Saved Frames into a Video ---
    print("\nCompiling frames into video...")
    images = [img for img in os.listdir(temp_dir) if img.endswith(".png")]
    images.sort()

    if not images:
        print("No frames were generated. Cannot create video.")
        shutil.rmtree(temp_dir)
        return

    frame = cv2.imread(os.path.join(temp_dir, images[0]))
    height, width, layers = frame.shape

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image_name in tqdm(images, desc="Writing Video"):
        img_path = os.path.join(temp_dir, image_name)
        out.write(cv2.imread(img_path))

    out.release()

    # --- 5. Clean Up the Temporary Directory ---
    shutil.rmtree(temp_dir)
    print(f"Visualization complete. 3D skeleton video saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create a video of an animated 3D skeleton.")
    parser.add_argument(
        '--data_input', 
        type=str, 
        required=True, 
        help="Path to the .npz file containing the 3D skeleton data."
    )
    parser.add_argument(
        '--video_output', 
        type=str, 
        required=True, 
        help="Path to save the output visualization video (e.g., 'skeleton_animation.mp4')."
    )
    
    args = parser.parse_args()
    
    visualize_skeleton_video(args.data_input, args.video_output)