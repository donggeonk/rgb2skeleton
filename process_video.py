"""
1. Read an input video file by frame
2. Use MediaPipe's Pose model to detect the 3D skeleton in each frame
3. Store the 3D world landmark data for each frame
4. Save the collected 3D skeleton data into a single, compressed NumPy file 
Output format - 3D - frames, 33 body landmarks, coordinates (nunmber_of_frames, 33, 3)
python process_video.py --video_input intput/input.mp4 --data_output output/output.npz
"""

import cv2
import mediapipe as mp
import numpy as np
import argparse

def process_video_to_3d_skeleton(video_path: str, output_path: str):
    """
    Processes a video to extract 3D pose landmarks and saves them to a file.

    Args:
        video_path (str): Path to the input RGB video file.
        output_path (str): Path to save the output .npz file.
    """
    # --- 1. Initialize MediaPipe Pose ---
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,        # for video processing
        model_complexity=1,             # 0, 1, or 2. Higher values are more accurate but slower.
        enable_segmentation=False,      # Set to False to focus on pose landmarks
        min_detection_confidence=0.5
    )
    
    # --- 2. Prepare to read video and store data ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    all_frame_landmarks = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames from {video_path}...")

    # --- 3. Process each frame in the video ---
    for frame_idx in range(frame_count):
        success, image = cap.read()
        if not success:
            break

        # Convert the BGR image to RGB 0 object dector models are trained on RGB format
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to get pose results
        results = pose.process(image_rgb)

        # --- 4. Extract and store 3D world landmarks ---
        if results.pose_world_landmarks:
            # The 3D skeleton data
            landmarks = results.pose_world_landmarks.landmark
            
            # Store the [x, y, z] coordinates for all 33 landmarks in a NumPy array
            frame_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            all_frame_landmarks.append(frame_data)
        else:
            # If no person is detected, append a placeholder (e.g., NaNs)
            # This maintains the correct number of frames.
            all_frame_landmarks.append(np.full((33, 3), np.nan))

    cap.release()
    pose.close()

    # --- 5. Save the collected data ---
    if all_frame_landmarks:
        # Convert the list of arrays into a single large NumPy array
        # Shape will be (num_frames, 33_landmarks, 3_coordinates)
        skeleton_data = np.stack(all_frame_landmarks, axis=0)

        # Save the data to a compressed .npz file
        np.savez_compressed(output_path, skeleton=skeleton_data)
        print(f"Successfully processed video and saved 3D skeleton data to {output_path}")
        print(f"Data shape: {skeleton_data.shape} (Frames, Landmarks, Coordinates)")
    else:
        print("Could not detect any poses in the video.")


if __name__ == '__main__':
    # --- Setup argument parser to make the script easy to use ---
    parser = argparse.ArgumentParser(description="Convert an RGB video into 3D skeleton data using MediaPipe.")
    parser.add_argument(
        '--video_input', 
        type=str, 
        required=True, 
        help="Path to the input video file (e.g., 'my_video.mp4')."
    )
    parser.add_argument(
        '--data_output', 
        type=str, 
        required=True, 
        help="Path to save the output .npz file (e.g., 'skeleton_data.npz')."
    )
    
    args = parser.parse_args()
    
    # --- Run the processing function ---
    process_video_to_3d_skeleton(args.video_input, args.data_output)