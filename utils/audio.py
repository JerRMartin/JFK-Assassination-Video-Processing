# utils/audio.py
from pathlib import Path
import shutil
import subprocess

def merge_audio_window(original_path: Path,
                       processed_path: Path,
                       output_with_audio: Path,
                       start_s: float,
                       end_s: float | None) -> bool:
    """
    Mux the processed video stream with an audio segment cut from the original file.
    - Audio window = [start_s, end_s] (if end_s is None, goes to end)
    - Requires ffmpeg on PATH.
    """
    if shutil.which("ffmpeg") is None:
        print("⚠️ FFmpeg not found on PATH; skipping audio merge.")
        return False

    try:
        # Build command:
        #  input 0: original (apply -ss/-t to trim audio)
        #  input 1: processed (video only, we copy it)
        #
        # Note: place -ss/-t *before* the -i to trim that input only.
        cmd = ["ffmpeg", "-y"]

        if start_s and start_s > 0:
            cmd += ["-ss", f"{start_s:.3f}"]

        if end_s is not None and end_s > start_s:
            duration = end_s - start_s
            cmd += ["-t", f"{duration:.3f}"]

        cmd += ["-i", str(original_path)]            # input 0 (trimmed audio)
        cmd += ["-i", str(processed_path)]           # input 1 (final video)

        cmd += [
            "-map", "1:v:0",     # take video from processed file
            "-map", "0:a:0?",    # take (optional) audio from original
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            str(output_with_audio),
        ]

        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print(f"FFmpeg audio merge failed: {e}")
        return False
