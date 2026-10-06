"""The 902 RGB JPEGs published with FCS-VMamba (paper Subset A)."""

import io
import zipfile
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from papers.vmamba.data_utils.wafer_dataset import WM811K_CLASSES

FOLDER_LABELS = {
    "none": 0,
    "center": 1,
    "donut": 2,
    "edge local": 3,
    "edge ring": 4,
    "local": 5,
    "near full": 6,
    "random": 7,
    "scratch": 8,
}


class AuthorWM811KDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_size=224,
        train=False,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.data_root = Path(data_root)
        if not self.data_root.is_file():
            raise FileNotFoundError(
                f"Author dataset not found: {data_root}; run python papers/vmamba/prepare_data.py"
            )
        with zipfile.ZipFile(self.data_root) as archive:
            self._samples = sorted(
                (name, FOLDER_LABELS[Path(name).parent.name.lower()])
                for name in archive.namelist()
                if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
                and Path(name).parent.name.lower() in FOLDER_LABELS
            )
        if not self._samples:
            raise ValueError("Archive contains no supported class folders")
        ops = [T.Resize((image_size, image_size))]
        if train:
            # Paper gives augmentation types, not their magnitudes.
            ops += [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        ops += [T.ToTensor(), T.Normalize(mean, std)]
        self.transform = T.Compose(ops)

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        name, label = self._samples[index]
        with zipfile.ZipFile(self.data_root) as archive:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                x = self.transform(image.convert("RGB"))
        return {"image": x, "label": label}

    @property
    def class_names(self):
        return list(WM811K_CLASSES)


def build_dataset(config, train=False):
    source = config.get("data.source", "raw_maps")
    if source == "author_archive":
        return AuthorWM811KDataset(
            config.get("data.data_root"),
            image_size=config.get("data.image_size", 224),
            train=train,
            mean=config.get("data.mean", [0.485, 0.456, 0.406]),
            std=config.get("data.std", [0.229, 0.224, 0.225]),
        )
    if source == "raw_maps":
        from papers.vmamba.data_utils.wafer_dataset import WaferWM811KDataset, build_train_augment

        return WaferWM811KDataset(
            config.get("data.data_root"),
            image_size=config.get("data.image_size", 224),
            transform=build_train_augment() if train else None,
        )
    raise ValueError(f"Unknown VMamba data.source: {source}")
