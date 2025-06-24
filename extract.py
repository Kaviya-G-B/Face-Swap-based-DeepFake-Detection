import cv2
import os
FPP_PATH = r"D:\Miniproject\FF++"
OUTPUT_FOLDER = r"D:\frames"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
def extract_frames(video_path, output_dir):
    """ Extract frames from a video and save them in the output directory. """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames

        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def process_all_videos():
    """ Process all videos in the F++ dataset (both subfolders). """
    for subfolder in os.listdir(FPP_PATH):
        subfolder_path = os.path.join(FPP_PATH, subfolder)

        if os.path.isdir(subfolder_path):  # Check if it's a folder
            for file in os.listdir(subfolder_path):
                if file.lower().endswith(".mp4"):  # Process only MP4 files
                    video_path = os.path.join(subfolder_path, file)

                    # Create a separate folder for each video inside OUTPUT_FOLDER
                    video_output_folder = os.path.join(OUTPUT_FOLDER, f"{subfolder}_{file[:-4]}")
                    os.makedirs(video_output_folder, exist_ok=True)

                    print(f"Processing video: {video_path}")
                    extract_frames(video_path, video_output_folder)

# Run the function
process_all_videos()

print("✅ Frame extraction for all videos completed.")








