import os
import cv2
import numpy as np


def load_images(data_dir):
    """Load all images from a folder."""
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    image_paths = []
    for name in sorted(os.listdir(data_dir)):
        if name.lower().endswith(valid_exts):
            image_paths.append(os.path.join(data_dir, name))
    return image_paths


def read_grayscale(image_path):
    """Read an image in grayscale."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def compute_canny_stages(image, low_threshold, high_threshold):
    """
    Run the standard Canny steps up to hysteresis.

    Returns a dictionary so the main script can inspect or save each stage.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx) * 180.0 / np.pi
    angle[angle < 0] += 180

    nms = non_maximum_suppression(magnitude, angle)

    strong_map = nms >= high_threshold
    weak_map = (nms >= low_threshold) & (nms < high_threshold)

    return {
        "blurred": blurred,
        "gx": gx,
        "gy": gy,
        "magnitude": magnitude,
        "angle": angle,
        "nms": nms,
        "strong_map": strong_map,
        "weak_map": weak_map,
    }


def non_maximum_suppression(magnitude, angle):
    """Thin edges by keeping only local maxima along the gradient direction."""
    h, w = magnitude.shape
    out = np.zeros((h, w), dtype=np.float64)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            a = angle[r, c]
            q = 0.0
            s = 0.0

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                q = magnitude[r, c + 1]
                s = magnitude[r, c - 1]
            elif 22.5 <= a < 67.5:
                q = magnitude[r + 1, c - 1]
                s = magnitude[r - 1, c + 1]
            elif 67.5 <= a < 112.5:
                q = magnitude[r + 1, c]
                s = magnitude[r - 1, c]
            elif 112.5 <= a < 157.5:
                q = magnitude[r - 1, c - 1]
                s = magnitude[r + 1, c + 1]

            if magnitude[r, c] >= q and magnitude[r, c] >= s:
                out[r, c] = magnitude[r, c]

    return out


def to_uint8(image):
    """Convert arrays to 0-255 uint8 for saving."""
    image = np.asarray(image)
    if image.dtype == np.bool_:
        return image.astype(np.uint8) * 255

    image = image.astype(np.float64)
    min_val = image.min()
    max_val = image.max()
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - min_val) / (max_val - min_val)
    return (255 * image).astype(np.uint8)
