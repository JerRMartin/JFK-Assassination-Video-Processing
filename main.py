import config as C
import cv2
from utils.processing import process_clip
from utils.audio import merge_audio_window
from utils.helpers import to_seconds, match_file, discover_videos, get_midpoint_frame
from utils.metrics import compute_video_metrics


def main():
    jobs = C.INSTRUCTIONS
    if not jobs:
        print("❌ No hardcoded jobs in config.INSTRUCTIONS")
        return

    videos = discover_videos(C.VIDEO_FOLDER)
    if not videos:
        print(f"❌ No videos found in '{C.VIDEO_FOLDER}'.")
        return

    # Ensure output folders exist
    C.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    C.STABILIZED_DIR.mkdir(parents=True, exist_ok=True)
    C.FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    for job in jobs:
        film = job["film"]
        start_s = to_seconds(job.get("start"))
        end_s   = to_seconds(job.get("end"))
        enh     = [e.lower() for e in job.get("enhancements", [])]

        do_stab     = any("stabil" in e for e in enh)
        do_denoise  = any("denois"  in e for e in enh)
        do_sharpen  = any("sharpen" in e for e in enh)

        src = match_file(videos, film)
        if not src:
            print(f"⚠️ Could not match film '{film}' to a file in {C.VIDEO_FOLDER}. Skipping.")
            continue

        tag_parts = []
        if do_denoise: tag_parts.append("denoise")
        if do_sharpen: tag_parts.append("sharp")
        tag = "_".join(tag_parts) if tag_parts else "copy"

        s_tag = f"{int((start_s or 0)//60)}m{int((start_s or 0)%60):02d}s"
        e_tag = "end" if end_s is None else f"{int(end_s//60)}m{int(end_s%60):02d}s"

        out_path = C.PROCESSED_DIR / f"{src.stem}_{s_tag}-{e_tag}_{tag}.mp4"
        stab_out_path = C.STABILIZED_DIR / f"{src.stem}_{s_tag}-{e_tag}_{tag}_stab.avi"
        print(f"▶️  {src.name} | {s_tag} → {e_tag} | effects={tag}")

        ok, fps, used_start, used_end = process_clip(
            src, out_path, do_denoise, do_sharpen, start_s, end_s
        )

        if ok and C.MERGE_AUDIO_WITH_FFMPEG:
            out_audio = C.PROCESSED_DIR / f"{out_path.stem}_aud.mp4"
            if merge_audio_window(src, out_path, out_audio, used_start, end_s if end_s is not None else used_end):
                try:
                    out_audio.replace(out_path)  # overwrite silent with audio version
                except Exception:
                    pass
        
        if do_stab:
            C.STABILIZER.stabilize(input_path=str(out_path), 
                                   output_path=str(stab_out_path), 
                                   border_type='black',
                                   border_size=100)
            print(f"✅ Stabilization done. Check the '{C.STABILIZED_DIR}' folder for {stab_out_path.name}.")

        # Get POST frame for visual assessment
        frame_path = C.FRAMES_DIR / f"POST_{out_path.stem}_frame.jpg"
        cv2.imwrite(str(frame_path), get_midpoint_frame(out_path))
        print(f"    📷 Saved frame to {frame_path}")

        # Only compute metrics if requested in the job's metrics list
        metrics_req = [m.lower() for m in job.get("metrics", [])]
        if any(m in metrics_req for m in ("ssim", "mse")):
            # Compare stabilized vs the pre-stabilization trimmed clip
            res = compute_video_metrics(
                ref_path=str(out_path),
                test_path=str(stab_out_path),
                crop_border=100,      # ignore 100px border to avoid black edges
                max_frames=None      # or set e.g. 300 for speed
            )
            print(f"📊 Metrics on stabilized window "
                  f"(frames={res['frames']}, crop=100px): "
                  f"SSIM={res['ssim']:.4f}, MSE={res['mse']:.2f}")

    print(f"✅ Done. Check the '{C.PROCESSED_DIR}' folder.")



if __name__ == "__main__":
    main()
