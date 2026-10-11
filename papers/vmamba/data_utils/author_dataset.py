"""The 902 RGB JPEGs published with FCS-VMamba (paper Subset A)."""

import io
import math
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from papers.vmamba.data_utils.wafer_dataset import WM811K_CLASSES
from papers.vmamba.prepare_data import SHA256, archive_sha256

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


def validate_author_archive(path):
    """Reject an incomplete or different archive before creating GPU tensors."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Author dataset not found: {path}; run python papers/vmamba/prepare_data.py"
        )
    if archive_sha256(path) != SHA256:
        raise ValueError(
            f"Author dataset checksum mismatch: {path}. Use the pinned 902-image "
            "archive prepared by python papers/vmamba/prepare_data.py. "
            "For an explicitly different diagnostic dataset, set data.verify_author_archive: false."
        )


def validate_dataset_images(dataset, indices=None):
    """Decode selected images once before model allocation or a CUDA update."""
    indices = list(dict.fromkeys(range(len(dataset)) if indices is None else indices))
    if not indices:
        raise ValueError("VMamba dataset selection is empty")
    if isinstance(dataset, AuthorWM811KDataset):
        with zipfile.ZipFile(dataset.data_root) as archive:
            for index in indices:
                name, _ = dataset._samples[index]
                try:
                    with Image.open(io.BytesIO(archive.read(name))) as image:
                        image.load()
                        image.convert("RGB")
                except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                    raise ValueError(
                        f"Cannot decode author image {name!r} in {dataset.data_root}"
                    ) from exc
    else:
        from papers.vmamba.data_utils.wafer_dataset import WaferWM811KDataset
        from papers.wafer_encoding import decode_die_map

        if not isinstance(dataset, WaferWM811KDataset):
            raise TypeError("Expected the author archive or raw WM-811K dataset")
        for index in indices:
            name, _ = dataset._samples[index]
            path = dataset.image_dir / name
            try:
                with Image.open(path) as image:
                    decode_die_map(image.convert("L"))
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot decode raw WM-811K image {path}") from exc
    return len(indices)


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
        if not isinstance(image_size, int) or isinstance(image_size, bool) or image_size <= 0:
            raise ValueError("VMamba image_size must be a positive integer")
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("Author RGB normalization needs three mean and std values")
        mean, std = tuple(float(x) for x in mean), tuple(float(x) for x in std)
        if not all(math.isfinite(x) for x in (*mean, *std)) or not all(x > 0 for x in std):
            raise ValueError("Author normalization must be finite, with positive std values")
        try:
            with zipfile.ZipFile(self.data_root) as archive:
                samples = []
                for name in archive.namelist():
                    # ZIP member paths have archive semantics, independent of the host OS.
                    member = PurePosixPath(name.replace("\\", "/"))
                    folder = member.parent.name.lower()
                    if member.suffix.lower() in {".jpg", ".jpeg", ".png"} and folder in FOLDER_LABELS:
                        samples.append((name, FOLDER_LABELS[folder]))
                self._samples = sorted(samples)
        except zipfile.BadZipFile as exc:
            raise ValueError(f"Invalid author dataset ZIP archive: {self.data_root}") from exc
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
        if config.get("data.verify_author_archive", True):
            validate_author_archive(config.get("data.data_root"))
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
