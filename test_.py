'''
import os
from utils.video_helpers import extract_frames
from utils.frame_helpers import hist_eq_frame, high_pass_frame
from vidstab import VidStab
import sewar

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
'''

#-------------------------------------------------------------------------
# ---- Unified pipeline: Stabilization -> Denoising -> Sharpening for 5 videos ----
import cv2
import numpy as np
import os
from pathlib import Path
import subprocess
import shutil

# ----------------------
# CONFIG
# ----------------------
# Update these paths (absolute or relative). Order/name them how you like.
# Point to your folder containing the videos
video_folder = Path("videos")   # change "my_videos" to your folder name

input_videos = sorted(
    [f for f in video_folder.glob("*") if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]]
)

output_dir = Path("processed_videos")
output_dir.mkdir(parents=True, exist_ok=True)

# Stabilization smoothing (in frames). Higher = smoother but more cropping/edge wobble.
SMOOTHING_RADIUS = 30

# Denoising strength (OpenCV fastNlMeansDenoisingColored)
DENOISE_H = 10          # luminance filter strength (typical 3–15)
DENOISE_H_COLOR = 10    # chroma filter strength
DENOISE_TEMPLATE = 7
DENOISE_SEARCH = 21

# Sharpen kernel (mild, avoids halos)
SHARPEN_KERNEL = np.array([[0, -0.5, 0],
                           [-0.5, 3.0, -0.5],
                           [0, -0.5, 0]], dtype=np.float32)

# Codec & output fps fallback
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
FPS_FALLBACK = 30.0

# If True and ffmpeg is available, we’ll mux original audio back into the processed video.
MERGE_AUDIO_WITH_FFMPEG = True and (shutil.which("ffmpeg") is not None)

def moving_average(curve, radius):
    """1D moving average with reflect padding."""
    if radius <= 0:
        return curve.copy()
    window_size = 2 * radius + 1
    # Reflect pad both ends
    padded = np.pad(curve, (radius, radius), mode='reflect')
    kernel = np.ones(window_size) / window_size
    return np.convolve(padded, kernel, mode='valid')

def smooth_trajectory(transforms, radius):
    """Smooth dx, dy, da (rotation) trajectories using moving average."""
    transforms = np.asarray(transforms)  # (N-1, 3)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed = np.vstack([
        moving_average(trajectory[:, 0], radius),
        moving_average(trajectory[:, 1], radius),
        moving_average(trajectory[:, 2], radius),
    ]).T
    # Compute difference to get smoothed transforms
    difference = smoothed - trajectory
    transforms_smooth = transforms + difference
    return transforms_smooth

def stabilize_frame(frame, transform, w, h):
    """Apply 2D affine transform to frame; fill borders by replicating edges."""
    dx, dy, da = transform
    m = np.array([[np.cos(da), -np.sin(da), dx],
                  [np.sin(da),  np.cos(da), dy]], dtype=np.float32)
    stabilized = cv2.warpAffine(frame, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return stabilized

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Could not open: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or FPS_FALLBACK
    if fps <= 1e-3:  # sometimes 0 from certain containers
        fps = FPS_FALLBACK
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1) Pass A: estimate per-frame transforms (dx, dy, da)
    transforms = []
    ret, prev = cap.read()
    if not ret:
        print(f"❌ No frames in: {input_path}")
        cap.release()
        return False

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    frame_idx = 1
    while True:
        ret, curr = cap.read()
        if not ret:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=30)
        if prev_pts is None or len(prev_pts) < 10:
            # If not enough features, assume no motion
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            frame_idx += 1
            continue

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        # Keep valid
        idx = status.reshape(-1) == 1
        prev_pts_v = prev_pts[idx]
        curr_pts_v = curr_pts[idx] if curr_pts is not None else None

        if curr_pts_v is None or len(prev_pts_v) < 10:
            transforms.append([0.0, 0.0, 0.0])
            prev_gray = curr_gray
            frame_idx += 1
            continue

        # Estimate rigid transform (translation + rotation)
        m, inliers = cv2.estimateAffinePartial2D(prev_pts_v, curr_pts_v, method=cv2.RANSAC, ransacReprojThreshold=3)
        if m is None:
            dx = dy = da = 0.0
        else:
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])

        transforms.append([dx, dy, da])
        prev_gray = curr_gray
        frame_idx += 1

    cap.release()

    if len(transforms) == 0:
        print(f"⚠️ Couldn’t estimate motion for: {input_path}")
        return False

    # Smooth transforms
    transforms = np.array(transforms, dtype=np.float32)
    transforms_smooth = smooth_trajectory(transforms, SMOOTHING_RADIUS)

    # 2) Pass B: write stabilized + denoised + sharpened frames
    cap = cv2.VideoCapture(str(input_path))
    out = cv2.VideoWriter(str(output_path), FOURCC, fps, (w, h))
    if not out.isOpened():
        print(f"❌ Could not open writer for: {output_path}")
        cap.release()
        return False

    # First frame: write as-is (no transform)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        out.release()
        return False

    # Denoise
    den = cv2.fastNlMeansDenoisingColored(first_frame, None, DENOISE_H, DENOISE_H_COLOR, DENOISE_TEMPLATE, DENOISE_SEARCH)
    # Sharpen
    shp = cv2.filter2D(den, -1, SHARPEN_KERNEL)
    out.write(shp)

    frame_idx = 1
    total = int(len(transforms_smooth)) + 1  # include first frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Stabilize current frame using smoothed transform for (frame_idx-1)
        t = transforms_smooth[frame_idx - 1]
        stab = stabilize_frame(frame, t, w, h)

        # Denoise -> Sharpen
        den = cv2.fastNlMeansDenoisingColored(stab, None, DENOISE_H, DENOISE_H_COLOR, DENOISE_TEMPLATE, DENOISE_SEARCH)
        shp = cv2.filter2D(den, -1, SHARPEN_KERNEL)

        out.write(shp)

        if frame_idx % 50 == 0 or frame_idx == total - 1:
            print(f"{input_path} :: {frame_idx+1}/{total} frames processed")
        frame_idx += 1

    cap.release()
    out.release()
    return True

def merge_audio(original_path, processed_path, output_with_audio):
    """
    Uses ffmpeg to copy original audio into processed video (re-encode video stream as is, copy audio).
    Requires ffmpeg installed and in PATH.
    """
    try:
        # -y overwrite; -c:v copy keeps processed video stream unchanged; -c:a copy copies original audio
        cmd = [
            "ffmpeg", "-y",
            "-i", str(processed_path),
            "-i", str(original_path),
            "-map", "0:v:0", "-map", "1:a:0?",
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(output_with_audio)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"FFmpeg audio merge failed for {processed_path}: {e}")
        return False

# ----------------------
# RUN
# ----------------------
for in_path in input_videos:
    in_path = Path(in_path)
    if not in_path.exists():
        print(f"⚠️ Skipping (not found): {in_path}")
        continue

    out_path = output_dir / f"{in_path.stem}_processed.mp4"
    print(f"▶️ Processing: {in_path} -> {out_path}")
    ok = process_video(in_path, out_path)

    if ok and MERGE_AUDIO_WITH_FFMPEG:
        out_audio = output_dir / f"{in_path.stem}_processed_audio.mp4"
        if merge_audio(in_path, out_path, out_audio):
            # Replace silent file with audio version
            try:
                os.replace(out_audio, out_path)
            except Exception:
                pass

print("✅ Done. Check the 'processed_videos' folder.")
