from pathlib import Path
import re
import cv2
import config as C

def to_seconds(tc: str | None) -> float | None:
    """Parse 'mm:ss' or 'hh:mm:ss' → seconds."""
    if not tc or not tc.strip():
        return None
    tc = tc.strip()
    parts = [int(p) for p in tc.split(":")]
    if len(parts) == 2:   # mm:ss
        mm, ss = parts
        return mm * 60 + ss
    if len(parts) == 3:   # hh:mm:ss
        hh, mm, ss = parts
        return hh * 3600 + mm * 60 + ss
    raise ValueError(f"Bad timecode '{tc}'. Use mm:ss or hh:mm:ss")

def match_file(videos: list[Path], query: str) -> Path | None:
    """Case-insensitive substring match of 'film' against file stem."""
    q = re.sub(r"\s+", " ", query.strip().lower())
    for p in videos:
        stem = p.stem.lower()
        if q in stem or stem in q:
            return p
    return None

def discover_videos(folder: Path):
    return sorted([f for f in folder.glob("*") if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])


def get_frame_from_video(frame_number: int, video_path: Path):
    """Grab a single frame (0-based index) from video_path and save it to C.FRAMES_DIR.

    Returns the Path to the saved frame on success, or None on failure.
    """
    if frame_number < 0:
        raise ValueError("frame_number must be >= 0")

    # Ensure output directory exists
    C.FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ERROR: Unable to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames and frame_number >= total_frames:
        print(f"    ERROR: Requested frame {frame_number} >= total frames {total_frames}")
        cap.release()
        return None

    # Seek to the desired frame and read
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_number))
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"    ERROR: Failed to read frame {frame_number} from {video_path}")
        return None

    frame_path = C.FRAMES_DIR / f"{video_path.stem}_frame{frame_number}.jpg"
    cv2.imwrite(str(frame_path), frame)
    print(f"    📷 Saved frame to {frame_path}")
    return frame_path

def get_midpoint_frame(video_path: Path):
    """Grab the midpoint frame from video_path and save it to C.FRAMES_DIR.

    Returns the Path to the saved frame on success, or None on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ERROR: Unable to open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        print(f"    ERROR: Video has zero frames: {video_path}")
        cap.release()
        return None

    mid_frame_number = total_frames // 2

    # Seek to the midpoint frame and read
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(mid_frame_number))
    ret, frame = cap.read()
    cap.release()

    return frame

