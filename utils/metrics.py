# utils/metrics.py
from typing import Optional, Dict
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def _to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def _center_crop(img, crop_px: int) -> np.ndarray:
    if crop_px <= 0:
        return img
    h, w = img.shape[:2]
    y0, y1 = crop_px, max(crop_px, h - crop_px)
    x0, x1 = crop_px, max(crop_px, w - crop_px)
    return img[y0:y1, x0:x1]

def compute_video_metrics(
    ref_path: str,
    test_path: str,
    crop_border: int = 20,
    max_frames: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compare two same-length videos frame-by-frame.
    Returns average SSIM and MSE over compared frames.
    - ref_path : original trimmed (pre-stabilization) clip
    - test_path: stabilized clip
    - crop_border: pixels to crop from each edge to avoid black borders
    - max_frames: cap frames for speed (None = all)
    """
    cap_ref = cv2.VideoCapture(ref_path)
    cap_t  = cv2.VideoCapture(test_path)

    if not cap_ref.isOpened() or not cap_t.isOpened():
        return {"frames": 0, "ssim": float("nan"), "mse": float("nan")}

    # Unify size (some codecs can change dimensions slightly)
    w = int(min(cap_ref.get(cv2.CAP_PROP_FRAME_WIDTH),
                cap_t.get(cv2.CAP_PROP_FRAME_WIDTH)))
    h = int(min(cap_ref.get(cv2.CAP_PROP_FRAME_HEIGHT),
                cap_t.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    total_frames = int(min(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT),
                           cap_t.get(cv2.CAP_PROP_FRAME_COUNT)))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    ssim_vals = []
    mse_vals  = []

    for i in range(total_frames):
        ok1, f1 = cap_ref.read()
        ok2, f2 = cap_t.read()
        if not (ok1 and ok2):
            break

        # Resize to common size
        f1 = cv2.resize(f1, (w, h), interpolation=cv2.INTER_AREA)
        f2 = cv2.resize(f2, (w, h), interpolation=cv2.INTER_AREA)

        # Optional center crop to avoid artificial black borders from stabilization
        if crop_border > 0:
            f1 = _center_crop(f1, crop_border)
            f2 = _center_crop(f2, crop_border)

        g1 = _to_gray(f1)
        g2 = _to_gray(f2)

        # skimage SSIM expects data_range
        dr = float(g1.max() - g1.min()) if g1.max() > g1.min() else 255.0
        ssim_score = ssim(g1, g2, data_range=dr)
        mse_score  = mean_squared_error(g1, g2)

        ssim_vals.append(float(ssim_score))
        mse_vals.append(float(mse_score))

    cap_ref.release()
    cap_t.release()

    n = len(ssim_vals)
    if n == 0:
        return {"frames": 0, "ssim": float("nan"), "mse": float("nan")}
    return {
        "frames": n,
        "ssim":  float(np.mean(ssim_vals)),
        "mse":   float(np.mean(mse_vals)),
    }
