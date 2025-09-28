import numpy as np
import cv2
from vidstab import VidStab

def moving_average(curve, radius: int):
    """1D moving average with reflect padding."""
    if radius <= 0:
        return curve.copy()
    window = 2 * radius + 1
    padded = np.pad(curve, (radius, radius), mode='reflect')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')

def smooth_trajectory(transforms: np.ndarray, radius: int):
    """
    Smooth dx, dy, da (rotation) trajectories using moving average.
    transforms: shape (N-1, 3)
    """
    transforms = np.asarray(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed = np.vstack([
        moving_average(trajectory[:, 0], radius),
        moving_average(trajectory[:, 1], radius),
        moving_average(trajectory[:, 2], radius),
    ]).T
    difference = smoothed - trajectory
    transforms_smooth = transforms + difference
    return transforms_smooth

def stabilize_frame(frame, transform, w: int, h: int):
    """Apply 2D affine transform; fill borders by replicating edges."""
    dx, dy, da = transform
    m = np.array([[np.cos(da), -np.sin(da), dx],
                  [np.sin(da),  np.cos(da), dy]], dtype=np.float32)
    return cv2.warpAffine(
        frame, m, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )
