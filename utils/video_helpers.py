import cv2
import os

# Helper function to extract frames from a video between start_time and end_time (in seconds)
def extract_frames(video_path, output_folder, start_time, end_time):
    os.makedirs(output_folder, exist_ok=True)

    # Open video
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)  # frames per second
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert times (seconds) → frame numbers
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Make sure we don’t go past the video
    end_frame = min(end_frame, total_frames - 1)

    # Jump to start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = 0
    current_frame = start_frame

    while current_frame <= end_frame:
        success, frame = video.read()
        if not success:
            break

        frame_filename = os.path.join(output_folder, f"frame_{count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        count += 1
        current_frame += 1

    video.release()
    print(f"Successfully Extracted {count} frames from {start_time}s to {end_time}s into '{output_folder}'!")