import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

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
    """
    def __init__(self, root_dir=None, transform=None, target_transform=None, train=False, with_masks=True, samples=None):
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
        return image, mask

class HarmonizationDataset(Dataset):
    """HarmNet dataset: (raw image, binary organoid mask, harmonization target).

    Read from `root_dir`, where the three sets live in sub-folders paired by file name
    ('IMG_B11_Interval1.tif' <-> 'MASK_B11_Interval1.tif'). In the released processed dataset the raw images are in
    `true_images/`, the targets (output of the classical pre-processing pipeline) in `images/` and the masks in `masks/`.
    Alternatively an explicit list of `samples` [(image_path, mask_path, target_path), ...] can be given.
    """
    def __init__(self, root_dir=None, input_folder="true_images", target_folder="images", mask_folder="masks",
                 transform=None, target_transform=None, samples=None):
        self.transform = transform
        self.target_transform = target_transform

        if samples is not None:
            self.image_paths = [s[0] for s in samples]
            self.mask_paths = [s[1] for s in samples]
            self.target_paths = [s[2] for s in samples]
        else:
            dirs = [os.path.join(root_dir, f) for f in (input_folder, mask_folder, target_folder)]
            files = [_list_files(d) for d in dirs]
            if len(files[0]) == 0:
                print(f"ATTENTION: No images found in {dirs[0]}")
            if not (len(files[0]) == len(files[1]) == len(files[2])):
                raise ValueError("Number of images, targets and masks do not match!")
            for names in zip(*files):
                if not (_pair_key(names[0]) == _pair_key(names[1]) == _pair_key(names[2])):
                    raise ValueError(f"File names do not match: {names}")
            self.image_paths, self.mask_paths, self.target_paths = ([os.path.join(d, f) for f in fs] for d, fs in zip(dirs, files))
        self.images = [os.path.basename(p) for p in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = _load_normalized(self.image_paths[idx])
        target = _load_normalized(self.target_paths[idx])
        mask = _load_binary_mask(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            target = self.transform(target)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask, target

# ----------------------------------------------------------------------------------------------------------------
# Pooling of the train/val/test folders (needed to build cross-validation folds)
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

def pool_harmonization_samples(processed_path, input_folder="true_images", target_folder="images", mask_folder="masks"):
    """{key: (image_path, mask_path, target_path)} over all the splits of the processed dataset."""
    pool = {}
    for split in SPLITS:
        ds = HarmonizationDataset(os.path.join(processed_path, split), input_folder, target_folder, mask_folder)
        for image, mask, target in zip(ds.image_paths, ds.mask_paths, ds.target_paths):
            pool[sample_key(image)] = (image, mask, target)
    return pool
