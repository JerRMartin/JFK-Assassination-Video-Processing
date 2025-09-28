from pathlib import Path
import re

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

def timestamp_to_seconds():
    pass