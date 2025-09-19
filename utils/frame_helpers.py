import cv2
import os
import numpy as np


def hist_eq_frame(img_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame = cv2.imread(img_path)

    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided to histogram_equalization")

    # Convert from BGR to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Split into channels
    y, cr, cb = cv2.split(ycrcb)

    # Equalize only the Y channel
    y_eq = cv2.equalizeHist(y)

    # Merge back the channels
    ycrcb_eq = cv2.merge((y_eq, cr, cb))

    # Convert back to BGR
    frame_eq = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(output_path, frame_eq)
    print(f"Histogram equalization complete. Saved to '{output_path}'")


def high_pass_frame(intensity, img_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    frame = cv2.imread(img_path)

    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided to high_pass_filter")

    # Define a simple 5x5 high-pass kernel
    kernel = np.array([
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1,  intensity, -1, -1],
        [-1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=np.float32)

    # Apply filter2D (depth = same as source)
    high_pass = cv2.filter2D(frame, ddepth=-1, kernel=kernel)

    cv2.imwrite(output_path, high_pass)
    print(f"High-pass filtering complete. Saved to '{output_path}'")