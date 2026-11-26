import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from hynet.scale import random_scale

IMG_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640))
])

MASK_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((480, 640), interpolation=transforms.InterpolationMode.NEAREST)
])

IMG_EXTENSIONS = ('.png', '.jpg', '.tif', '.tiff')
SPLITS = ("train", "val", "test")

def _list_files(directory):
    return sorted([f for f in os.listdir(directory) if f.endswith(IMG_EXTENSIONS)])

def _load_normalized(path):
    """Load an image as float32 grayscale normalized to the [0, 1] range."""
    img_array = np.array(Image.open(path))
    if img_array.ndim == 3:
        img_array = np.mean(img_array[..., :3], axis=2)
    img_array = img_array.astype(np.float32)
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max > img_min:
        return (img_array - img_min) / (img_max - img_min)
    return img_array - img_min

def _load_binary_mask(path):
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.float32)

def load_mask_480x640(path):
    """Ground-truth mask on the 480x640 grid used for training and evaluation (nearest-neighbour resize), as a boolean array."""
    return MASK_TRANSFORM(_load_binary_mask(path))[0].numpy() > 0.5

def _pair_key(filename):
    """'IMG_B11_Interval1.tif' and 'MASK_B11_Interval1.tif' -> 'B11_Interval1.tif'."""
    return filename.split("_", 1)[-1]

def sample_key(filename):
    """Extension-less pairing key: 'IMG_B11_Interval1.tif' / 'MASK_B11_Interval1.png' / 'IMG_B11_Interval1' -> 'B11_Interval1'."""
    return os.path.splitext(_pair_key(os.path.basename(filename)))[0]

class OrganoidDataset(Dataset):
    """Segmentation dataset: images and masks paired by sorted file name.

    Either `root_dir` (containing images/ and masks/) or an explicit list of `samples` [(image_path, mask_path), ...] is given.
    `with_masks=False` allows inference on a folder that only contains images/ (a zero mask is returned).
    `image_fn(image, path)`, if given, modifies the normalized float image before the resize (see hynet.perturb).
    """
    def __init__(self, root_dir=None, transform=None, target_transform=None, train=False, with_masks=True, samples=None, image_fn=None, scale_aug=0.0):
        self.image_fn = image_fn
        self.scale_aug = scale_aug  # > 1: random zoom of the training pairs by a factor in [1/scale_aug, scale_aug] (hynet.scale)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.with_masks = with_masks

        if samples is not None:
            self.image_paths = [s[0] for s in samples]
            self.mask_paths = [s[1] for s in samples]
        else:
            self.root_dir = root_dir
            img_dir = os.path.join(root_dir, "images")
            mask_dir = os.path.join(root_dir, "masks")
            self.image_paths = [os.path.join(img_dir, f) for f in _list_files(img_dir)]
            if len(self.image_paths) == 0:
                print(f"ATTENTION: No images found in {img_dir}")
            self.mask_paths = []
            if with_masks:
                self.mask_paths = [os.path.join(mask_dir, f) for f in _list_files(mask_dir)]
                if len(self.mask_paths) == 0:
                    print(f"ATTENTION: No masks found in {mask_dir}")
                if len(self.image_paths) != len(self.mask_paths):
                    raise ValueError("Number of images and masks do not match!")
        self.images = [os.path.basename(p) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = _load_normalized(self.image_paths[idx])
        if self.image_fn:
            image = self.image_fn(image, self.image_paths[idx])
        if self.transform:
            image = self.transform(image)

        if self.with_masks:
            mask = _load_binary_mask(self.mask_paths[idx])
            if self.target_transform:
                mask = self.target_transform(mask)
        else:
            mask = torch.zeros_like(image)

        if self.train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            if self.scale_aug > 1 and random.random() < 0.5:
                image, mask = random_scale(image, mask, self.scale_aug)
        return image, mask

# ----------------------------------------------------------------------------------------------------------------
# Pooling of the train/val/test folders (needed to select the wells of a fold)
# ----------------------------------------------------------------------------------------------------------------
def pool_segmentation_samples(dataset_path):
    """{key: (image_path, mask_path)} over all the splits of the segmentation dataset."""
    pool = {}
    for split in SPLITS:
        if not os.path.isdir(os.path.join(dataset_path, split)):
            continue  # e.g. the external datasets only have a test split
        ds = OrganoidDataset(os.path.join(dataset_path, split))
        for image, mask in zip(ds.image_paths, ds.mask_paths):
            key = sample_key(image)
            if key in pool:
                raise ValueError(f"Duplicate sample {key} in {dataset_path}")
            pool[key] = (image, mask)
    return pool


# ----------------------------------------------------------------------------------------------------------------
# Datasets pooled from several sources (hynet-prepare-pool): files are named IMG_<domain>-<id>_Interval<n>.<ext>
# ----------------------------------------------------------------------------------------------------------------
def domain_of(path):
    """'.../IMG_orgaseg-0002_x_Interval0.png' -> 'orgaseg'; names without a domain prefix all belong to the domain ''."""
    key = _pair_key(os.path.basename(path))
    return key.split("-", 1)[0] if "-" in key.split("_", 1)[0] else ""

def domain_weights(paths):
    """Sampling weight of every sample such that each domain is drawn with the same total probability."""
    domains = [domain_of(p) for p in paths]
    counts = {d: domains.count(d) for d in set(domains)}
    return [1.0 / counts[d] for d in domains]
