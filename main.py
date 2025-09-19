import os
from utils.video_helpers import extract_frames
from utils.frame_helpers import hist_eq_frame, high_pass_frame
from vidstab import VidStab

stable_output_folder = "stabilized_videos"
videos_input_folder = "videos"

#TODO: Do this for each video we want to process, keep in mind we have 5, we only need 5

extract_frames(
    f"{videos_input_folder}/Couch film of John F. Kennedy assassination.mp4", 
    "extracted_frames/couch_film/original", 
    10, #TODO: decide our actual start time
    11 #TODO: Decide our actual end time
    )

for frame in os.listdir("extracted_frames/couch_film/original"):

    # # Apply High-pass filter
    # high_pass_frame(24.85,
    #                 f"extracted_frames/couch_film/original/{frame}", 
    #                 f"extracted_frames/couch_film/high_pass/{frame}")

    # # Histogram equalize each frame
    # hist_eq_frame(f"extracted_frames/couch_film/original/{frame}", #TODO: change this to the output of high-pass filter
    #               f"extracted_frames/couch_film/equalized/{frame}")

#TODO: Recombine the frames into a video to pass to the stabilizer

# # Stabilize video
# stabilizer = VidStab()
# os.makedirs(stable_output_folder, exist_ok=True)
# stabilizer.stabilize(
#     input_path=f"{videos_input_folder}/Couch film of John F. Kennedy assassination.mp4", 
#     output_path=f"{stable_output_folder}/Couch film Stablized.avi"
#     )


#TODO: Gather Quantitative Metrics on the video quality before and after all the Image processing