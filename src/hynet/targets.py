"""Harmonization targets: the classical pre-processing pipeline that HarmNet learns to reproduce.

CLAHE (with reflected borders), subtraction of a Gaussian-blurred background, normalization to the full range, histogram equalization and a
bilateral filter. The parameters are in pixels, tuned on the images of our dataset (1280x960): to apply the pipeline to images of another size
the image is first brought to `PIPELINE_SIZE`, so that the filters act on the same scale of details (applied after the resize to the 480x640 network
input, the pipeline gives a visibly different result: correlation of about 0.8 with the target of the original resolution).
"""
import cv2
import numpy as np

PIPELINE_SIZE = (1280, 960)  # (width, height) of the images of our dataset
CLAHE_PAD = 20

def to_uint8_gray(image):
    """Single-channel uint8 image: colour images are converted to grey, other bit depths are stretched to 0..255."""
    if image.ndim == 3:
        image = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY) if image.shape[2] >= 3 else image[..., 0]
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float32)
    lo, hi = float(image.min()), float(image.max())
    return np.round(255 * (image - lo) / (hi - lo)).astype(np.uint8) if hi > lo else np.zeros(image.shape, np.uint8)

def resize_to(image, size=PIPELINE_SIZE):
    """Resize to `size` = (width, height) (cubic when enlarging, area averaging when shrinking); size None keeps the image as it is."""
    if size is None or (image.shape[1], image.shape[0]) == tuple(size):
        return image
    enlarging = size[0] * size[1] > image.shape[0] * image.shape[1]
    return cv2.resize(image, tuple(size), interpolation=cv2.INTER_CUBIC if enlarging else cv2.INTER_AREA)

def classical_pipeline(image, clip_limit=5.0, tile_grid=(8, 8), background_kernel=31, bilateral=(9, 75, 75)):
    """uint8 grayscale image -> uint8 harmonization target of the same size."""
    padded = cv2.copyMakeBorder(image, CLAHE_PAD, CLAHE_PAD, CLAHE_PAD, CLAHE_PAD, cv2.BORDER_REFLECT)
    enhanced = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid).apply(padded)[CLAHE_PAD:-CLAHE_PAD, CLAHE_PAD:-CLAHE_PAD]
    background = cv2.GaussianBlur(enhanced, (background_kernel, background_kernel), 0)
    corrected = cv2.subtract(enhanced, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    equalized = cv2.equalizeHist(corrected)
    diameter, sigma_color, sigma_space = bilateral
    return cv2.bilateralFilter(equalized, d=diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space)

def make_target(image, size=PIPELINE_SIZE):
    """Target of an image of any size/bit depth: grey uint8, brought to `size`, then the classical pipeline."""
    return classical_pipeline(resize_to(to_uint8_gray(image), size))
