import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from hynet import data, targets
from hynet.cli.make_targets import build_processed_dataset

def gradient_image(height=96, width=128, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(40, 200, width)[None, :].repeat(height, axis=0)
    return np.clip(base + rng.normal(0, 8, base.shape), 0, 255).astype(np.uint8)

def test_pipeline_keeps_size_and_type_and_is_deterministic():
    image = gradient_image()
    out = targets.classical_pipeline(image)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(out, targets.classical_pipeline(image))

def test_make_target_resizes_before_the_pipeline():
    image = gradient_image(100, 100)
    out = targets.make_target(image, size=(160, 120))
    assert out.shape == (120, 160)
    assert np.array_equal(out, targets.classical_pipeline(targets.resize_to(image, (160, 120))))
    assert targets.make_target(image, size=None).shape == image.shape

def test_to_uint8_gray_handles_colour_and_bit_depth():
    colour = np.stack([gradient_image()] * 3, axis=-1)
    assert targets.to_uint8_gray(colour).shape == colour.shape[:2]
    sixteen = (gradient_image().astype(np.uint16) * 200)
    out = targets.to_uint8_gray(sixteen)
    assert out.dtype == np.uint8 and out.min() == 0 and out.max() == 255
    assert targets.to_uint8_gray(np.full((8, 8), 7, np.uint16)).max() == 0

def test_layout_matches_harmonization_dataset(tmp_path):
    src, out = tmp_path / "src", tmp_path / "processed"
    for split, n in (("train", 3), ("val", 1)):
        (src / split / "images").mkdir(parents=True)
        (src / split / "masks").mkdir()
        for i in range(n):
            cv2.imwrite(str(src / split / "images" / f"IMG_W{i}_Interval1.png"), gradient_image(seed=i))
            cv2.imwrite(str(src / split / "masks" / f"MASK_W{i}_Interval1.png"), (gradient_image(seed=i) > 120).astype(np.uint8) * 255)
    assert build_processed_dataset(str(src), str(out), size=(128, 96)) == {"train": 3, "val": 1}
    ds = data.HarmonizationDataset(str(out / "train"))  # checks that the three folders are paired by name
    assert len(ds) == 3
    raw = cv2.imread(str(out / "train" / "true_images" / "IMG_W0_Interval1.png"), cv2.IMREAD_UNCHANGED)
    target = cv2.imread(str(out / "train" / "images" / "IMG_W0_Interval1.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(raw, gradient_image(seed=0))
    assert np.array_equal(target, targets.classical_pipeline(raw))

def test_missing_mask_is_an_error(tmp_path):
    (tmp_path / "s" / "train" / "images").mkdir(parents=True)
    (tmp_path / "s" / "train" / "masks").mkdir()
    cv2.imwrite(str(tmp_path / "s" / "train" / "images" / "IMG_A_Interval1.png"), gradient_image())
    with pytest.raises(FileNotFoundError):
        build_processed_dataset(str(tmp_path / "s"), str(tmp_path / "o"))
