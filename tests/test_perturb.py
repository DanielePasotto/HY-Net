import numpy as np
import pytest

torch = pytest.importorskip("torch")
from PIL import Image
from hynet import data, perturb
from hynet.cli.robustness import well_scores

def _image(seed=0, size=(64, 80)):
    return np.random.default_rng(seed).random(size).astype(np.float32)

@pytest.mark.parametrize("spec", perturb.SWEEP)
def test_perturbations_keep_the_shape_and_the_range(spec):
    out = perturb.apply(_image(), spec, np.random.default_rng(0))
    assert out.shape == (64, 80) and out.dtype == np.float32
    assert out.min() == pytest.approx(0.0, abs=1e-6) and out.max() == pytest.approx(1.0, abs=1e-6)

def test_gamma_one_is_the_identity():
    img = _image()
    img = (img - img.min()) / (img.max() - img.min())  # images are min-max normalized when loaded
    assert np.allclose(perturb.apply(img, "gamma:1", np.random.default_rng(0)), img, atol=1e-6)

def test_blur_smooths_and_noise_roughens():
    img = _image()
    rng = np.random.default_rng(0)
    assert perturb.apply(img, "blur:2", rng).std() < img.std()
    smooth = perturb.apply(_image(), "blur:4", rng)
    assert np.abs(np.diff(perturb.apply(smooth, "noise:0.1", rng), axis=1)).mean() > np.abs(np.diff(smooth, axis=1)).mean()

def test_fixed_perturbation_depends_only_on_the_file_name():
    fn = perturb.fixed("noise:0.05")
    img = _image()
    assert np.array_equal(fn(img, "/a/IMG_B1_Interval1.tif"), fn(img, "/b/IMG_B1_Interval1.tif"))
    assert not np.array_equal(fn(img, "/a/IMG_B1_Interval1.tif"), fn(img, "/a/IMG_B2_Interval1.tif"))

def test_random_augment_is_reproducible_and_valid():
    img = _image()
    np.random.seed(3)
    first = perturb.random_augment(img)
    np.random.seed(3)
    assert np.array_equal(first, perturb.random_augment(img))
    assert first.shape == img.shape and first.min() >= 0.0 and first.max() <= 1.0 + 1e-6

def test_wide_ranges_contain_the_whole_sweep():
    draws = {name: [draw(np.random) for _ in range(2000)] for name, draw in perturb.AUGMENT_WIDE}
    for spec in perturb.SWEEP:
        name, level = spec.split(":")
        assert min(draws[name]) <= float(level) <= max(draws[name]), spec
    np.random.seed(3)
    img = _image()
    out = perturb.random_augment(img, ranges="wide")
    assert out.shape == img.shape and out.min() >= 0.0 and out.max() <= 1.0 + 1e-6

def _write_dataset(root, names):
    for sub in ("images", "masks"):
        (root / sub).mkdir(parents=True)
    rng = np.random.default_rng(1)
    for name in names:
        Image.fromarray((rng.random((48, 64)) * 255).astype(np.uint8)).save(root / "images" / f"IMG_{name}.png")
        mask = np.zeros((48, 64), np.uint8)
        mask[10:30, 10:40] = 255
        Image.fromarray(mask).save(root / "masks" / f"MASK_{name}.png")

def test_dataset_applies_the_image_function(tmp_path):
    _write_dataset(tmp_path, ["B1_Interval1"])
    calls = []
    def image_fn(img, path):
        calls.append(path)
        return img * 0.0
    ds = data.OrganoidDataset(str(tmp_path), transform=data.IMG_TRANSFORM, target_transform=data.MASK_TRANSFORM, image_fn=image_fn)
    assert float(ds[0][0].abs().sum()) == 0.0 and len(calls) == 1

def test_well_scores_groups_images_by_well(tmp_path):
    _write_dataset(tmp_path, ["B1_Interval1", "B1_Interval2", "B2_Interval1"])
    ds = data.OrganoidDataset(str(tmp_path), transform=data.IMG_TRANSFORM, target_transform=data.MASK_TRANSFORM)
    masks = [data.load_mask_480x640(p) for p in ds.mask_paths]
    always_foreground = lambda x: torch.full_like(x, 10.0)  # logits > 0 everywhere
    scores = well_scores(always_foreground, ds, masks, torch.device("cpu"), workers=0)
    assert set(scores) == {"B1", "B2"} and scores["B1"][0] == 2
    fraction = masks[0].mean()
    assert scores["B1"][1] == pytest.approx(2 * fraction / (1 + fraction), rel=1e-5)
