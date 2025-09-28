from pathlib import Path
import numpy as np
import shutil
import cv2
from vidstab import VidStab

# Folders
VIDEO_FOLDER = Path("videos")
PROCESSED_DIR = Path("processed_videos")
STABILIZED_DIR = Path("stabilized_videos")

# Stabilization
SMOOTHING_RADIUS = 30  # higher = smoother but more crop/edge wobble
STABILIZER = VidStab()

# Denoising (OpenCV fastNlMeansDenoisingColored)
DENOISE_H = 10           # luminance
DENOISE_H_COLOR = 10     # chroma
DENOISE_TEMPLATE = 7
DENOISE_SEARCH = 21

# Sharpen kernel (mild, avoids halos)
SHARPEN_KERNEL = np.array(
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]], dtype=np.float32
)

# I/O & codec
FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
FPS_FALLBACK = 30.0

# If True and ffmpeg is available, we’ll mux original audio back in
MERGE_AUDIO_WITH_FFMPEG = (shutil.which("ffmpeg") is not None)

# ── Hardcoded job list ──────────────────────────────────
# Use mm:ss or hh:mm:ss for times. Leave "end" as None to go until the file's end.
INSTRUCTIONS = [
    {
        "film": "Hughes film of John F. Kennedy assassination",
        "start": "0:06",
        "end":   "0:27",
        "enhancements": ["denoise", "sharpen", "stabilize"]
    },
    {
        "film": "Martin film of John F. Kennedy assassination",
        "start": "0:07",
        "end":   "0:12",
        "enhancements": ["denoise", "sharpen", "stabilize"]
    },
    {
        "film": "JFK Assassination Bell Film",
        "start": "0:35",
        "end":   "0:49",
        "enhancements": ["sharpen", "stabilize"]
    },
    {
        "film": "Wiegman film of John F. Kennedy assassination",
        "start": "0:05",
        "end":   "0:15",
        "enhancements": ["sharpen", "stabilize"]
    },
    {
        "film": "Couch film of John F. Kennedy assassination",
        "start": "0:00",
        "end":   "0:10",
        "enhancements": ["denoise", "sharpen", "stabilize"]
    },
]
