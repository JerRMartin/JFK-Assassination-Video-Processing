from pathlib import Path
import re
import config as C
from utils.processing import process_clip
from utils.audio import merge_audio_window

def _to_seconds(tc: str | None) -> float | None:
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

def _match_file(videos: list[Path], query: str) -> Path | None:
    """Case-insensitive substring match of 'film' against file stem."""
    q = re.sub(r"\s+", " ", query.strip().lower())
    for p in videos:
        stem = p.stem.lower()
        if q in stem or stem in q:
            return p
    return None

def discover_videos(folder: Path):
    return sorted([f for f in folder.glob("*") if f.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]])

def main():
    jobs = C.INSTRUCTIONS
    if not jobs:
        print("❌ No hardcoded jobs in config.INSTRUCTIONS")
        return

    videos = discover_videos(C.VIDEO_FOLDER)
    if not videos:
        print(f"❌ No videos found in '{C.VIDEO_FOLDER}'.")
        return

    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        film = job["film"]
        start_s = _to_seconds(job.get("start"))
        end_s   = _to_seconds(job.get("end"))
        enh     = [e.lower() for e in job.get("enhancements", [])]

        do_stab     = any("stabil" in e for e in enh)
        do_denoise  = any("denois"  in e for e in enh)
        do_sharpen  = any("sharpen" in e for e in enh)

        src = _match_file(videos, film)
        if not src:
            print(f"⚠️ Could not match film '{film}' to a file in {C.VIDEO_FOLDER}. Skipping.")
            continue

        tag_parts = []
        if do_stab: tag_parts.append("stab")
        if do_denoise: tag_parts.append("denoise")
        if do_sharpen: tag_parts.append("sharp")
        tag = "_".join(tag_parts) if tag_parts else "copy"

        s_tag = f"{int((start_s or 0)//60)}m{int((start_s or 0)%60):02d}s"
        e_tag = "end" if end_s is None else f"{int(end_s//60)}m{int(end_s%60):02d}s"

        out_path = C.OUTPUT_DIR / f"{src.stem}_{s_tag}-{e_tag}_{tag}.mp4"
        print(f"▶️ {src.name} | {s_tag} → {e_tag} | effects={tag}")

        ok, fps, used_start, used_end = process_clip(
            src, out_path, start_s or 0.0, end_s, do_stab, do_denoise, do_sharpen
        )

        if ok and C.MERGE_AUDIO_WITH_FFMPEG:
            out_audio = C.OUTPUT_DIR / f"{out_path.stem}_aud.mp4"
            if merge_audio_window(src, out_path, out_audio, used_start, end_s if end_s is not None else used_end):
                try:
                    out_audio.replace(out_path)  # overwrite silent with audio version
                except Exception:
                    pass

    print("✅ Done. Check the 'processed_videos' folder.")

if __name__ == "__main__":
    main()
