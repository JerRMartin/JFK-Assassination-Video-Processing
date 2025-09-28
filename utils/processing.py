# utils/processing.py
from pathlib import Path
import cv2

import config as C

def _apply_effects(frame, do_denoise: bool, do_sharpen: bool):
    out = frame
    if do_denoise:
        out = cv2.fastNlMeansDenoisingColored(
            out, None, C.DENOISE_H, C.DENOISE_H_COLOR, C.DENOISE_TEMPLATE, C.DENOISE_SEARCH
        )
    if do_sharpen:
        out = cv2.filter2D(out, -1, C.SHARPEN_KERNEL)
    return out

def process_clip(input_path: Path, output_path: Path,
                 do_denoise: bool, do_sharpen: bool, 
                 start_s: float = 0.0, end_s: float = None):
    """
    Process only [start_s, end_s] and apply selected effects.
    Returns (ok, fps, start_used, end_used).
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"❌ Could not open: {input_path}")
        return False, 0.0, 0.0, 0.0

    fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS_FALLBACK
    if fps <= 1e-3:
        fps = C.FPS_FALLBACK
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(round((start_s or 0.0) * fps)))
    end_frame = total_frames - 1 if end_s is None else min(total_frames - 1, int(round(end_s * fps)))
    if end_frame <= start_frame:
        # ensure at least a tiny segment (≈1s) if inputs collide
        end_frame = min(total_frames - 1, start_frame + max(1, int(1 * fps)))
   
    # ---------- Pass A: write only the window with selected effects ----------
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    out = cv2.VideoWriter(str(output_path), C.FOURCC, fps, (w, h))
    if not out.isOpened():
        cap.release()
        print(f"❌ Could not open writer for: {output_path}")
        return False, fps, start_s, end_frame / fps

    # first frame in window
    ret, frame = cap.read()
    if not ret:
        cap.release(); out.release()
        return False, fps, start_s, end_frame / fps
    # (no transform for first frame in the segment)
    frame = _apply_effects(frame, do_denoise, do_sharpen)
    out.write(frame)

    # remaining frames
    idx_in_segment = 0  # index into transforms_smooth
    total_segment_frames = (end_frame - start_frame) + 1
    for n in range(1, total_segment_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = _apply_effects(frame, do_denoise, do_sharpen)
        out.write(frame)
        idx_in_segment += 1
        if n % 50 == 0 or n == total_segment_frames - 1:
            print(f"    {input_path} :: {n+1}/{total_segment_frames} frames processed")

    cap.release()
    out.release()
    return True, fps, start_s, end_frame / fps
