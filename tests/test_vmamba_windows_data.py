"""Windows-compatible VMamba data preparation and spawn loading contracts."""

from __future__ import annotations

import hashlib
import io
import urllib.error
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from common.engine.config import EngineConfig
from papers.vmamba import prepare_data
from papers.vmamba.data_utils.author_dataset import (
    FOLDER_LABELS,
    AuthorWM811KDataset,
    build_dataset,
    validate_dataset_images,
)
from papers.vmamba.data_utils.protocol import stratified_split
from papers.vmamba.data_utils.wafer_dataset import (
    WaferWM811KDataset,
    parse_wm811k_labeled_samples,
)


def _archive(path: Path, names=None, corrupt=False):
    names = names or [f"WM811k_Dataset/{folder}/sample.jpg" for folder in FOLDER_LABELS]
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (255, 64, 0)).save(buffer, format="JPEG")
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, b"broken image" if corrupt else buffer.getvalue())
    return path


def test_author_archive_accepts_windows_member_paths_without_reordering(tmp_path):
    names = ["dataset\\Center\\z.jpg", "dataset\\none\\a.JPG"]
    path = _archive(tmp_path / "source archive.zip", names)
    dataset = AuthorWM811KDataset(path, image_size=16)
    assert dataset._samples == sorted([(names[0], 1), (names[1], 0)])
    assert dataset[0]["image"].shape == (3, 16, 16)
    assert validate_dataset_images(dataset) == 2


def test_author_rgb_tensor_matches_declared_normalization(tmp_path):
    path = _archive(tmp_path / "source.zip", ["dataset/Center/a.jpg"])
    dataset = AuthorWM811KDataset(path, image_size=32, mean=(0, 0, 0), std=(1, 1, 1))
    image = dataset[0]["image"]
    assert image.shape == (3, 32, 32)
    assert image[0].mean() > 0.9
    assert 0.2 < image[1].mean() < 0.3
    assert image[2].mean() < 0.1
    assert dataset[0]["label"] == 1


@pytest.mark.parametrize("train", [False, True])
def test_author_dataset_is_picklable_for_spawn_workers(tmp_path, train):
    path = _archive(tmp_path / "source.zip")
    dataset = AuthorWM811KDataset(path, image_size=16, train=train)
    loader = DataLoader(dataset, batch_size=3, num_workers=1, multiprocessing_context="spawn")
    try:
        samples = list(loader)
    except RuntimeError as exc:
        if "torch_shm_manager" in str(exc) and "Operation not permitted" in str(exc):
            pytest.skip("Host sandbox blocks PyTorch shared-memory manager; rerun without sandbox")
        raise
    images = torch.cat([sample["image"] for sample in samples])
    labels = torch.cat([sample["label"] for sample in samples])
    assert images.shape == (9, 3, 16, 16)
    assert torch.isfinite(images).all()
    assert sorted(labels.tolist()) == list(range(9))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"image_size": 0}, "positive integer"),
        ({"image_size": True}, "positive integer"),
        ({"mean": (0, 0)}, "three mean"),
        ({"std": (1, 0, 1)}, "positive std"),
        ({"std": (1, float("nan"), 1)}, "finite"),
    ],
)
def test_invalid_author_transform_fails_before_loading(tmp_path, kwargs, message):
    path = _archive(tmp_path / "source.zip")
    with pytest.raises(ValueError, match=message):
        AuthorWM811KDataset(path, **kwargs)


def test_build_dataset_rejects_different_author_archive_by_default(tmp_path):
    path = _archive(tmp_path / "source.zip")
    config = EngineConfig.from_dict({"data": {"source": "author_archive", "data_root": str(path)}})
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_dataset(config)
    config._data["data"]["verify_author_archive"] = False
    assert len(build_dataset(config)) == 9


def test_dataset_preflight_identifies_corrupt_member(tmp_path):
    path = _archive(tmp_path / "source.zip", ["dataset/Center/a.jpg"], corrupt=True)
    dataset = AuthorWM811KDataset(path)
    with pytest.raises(ValueError, match="Center/a.jpg"):
        validate_dataset_images(dataset)


def test_selected_image_preflight_does_not_decode_unselected_members(tmp_path):
    path = _archive(tmp_path / "source.zip", ["dataset/Center/good.jpg"])
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("dataset/Scratch/bad.jpg", b"bad image")
    dataset = AuthorWM811KDataset(path)
    assert validate_dataset_images(dataset, [0, 0]) == 1
    with pytest.raises(ValueError, match="Scratch/bad.jpg"):
        validate_dataset_images(dataset)


def test_invalid_zip_has_actionable_error(tmp_path):
    path = tmp_path / "source.zip"
    path.write_bytes(b"partial download")
    with pytest.raises(ValueError, match="Invalid author dataset ZIP"):
        AuthorWM811KDataset(path)


def test_prepared_archive_can_be_reverified_offline(tmp_path, monkeypatch):
    output = tmp_path / "source.zip"
    payload = b"verified archive payload"
    output.write_bytes(payload)
    monkeypatch.setattr(prepare_data, "SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(
        prepare_data.urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("Already prepared data must not use the network"),
    )
    old_mtime = output.stat().st_mtime_ns
    assert prepare_data.prepare(output) == output
    assert output.stat().st_mtime_ns == old_mtime


def test_prepare_copies_verified_archive_atomically(tmp_path, monkeypatch):
    source = tmp_path / "source.zip"
    source.write_bytes(b"verified data")
    monkeypatch.setattr(prepare_data, "SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    output = tmp_path / "output directory" / "data.zip"
    assert prepare_data.prepare(output, source) == output
    assert output.read_bytes() == source.read_bytes()
    assert list(output.parent.iterdir()) == [output]


def test_prepare_refuses_different_output_before_network(tmp_path, monkeypatch):
    output = tmp_path / "source.zip"
    output.write_bytes(b"existing wrong data")
    monkeypatch.setattr(
        prepare_data.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("No network")
    )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_data.prepare(output)
    assert output.read_bytes() == b"existing wrong data"


def test_prepare_checksum_failure_leaves_no_partial_file(tmp_path):
    source = tmp_path / "wrong.zip"
    source.write_bytes(b"wrong data")
    output = tmp_path / "prepared.zip"
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_data.prepare(output, source)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [source]


def test_prepare_network_error_has_offline_recovery(tmp_path, monkeypatch):
    def fail_download(*args, **kwargs):
        raise urllib.error.URLError("Connection blocked")

    monkeypatch.setattr(prepare_data.urllib.request, "urlopen", fail_download)
    with pytest.raises(RuntimeError, match=r"Copy the verified.*--archive"):
        prepare_data.prepare(tmp_path / "source.zip")
    assert not (tmp_path / "source.zip").exists()


def test_prepare_missing_local_archive_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Local author archive not found"):
        prepare_data.prepare(tmp_path / "output.zip", tmp_path / "missing.zip")


def test_raw_csv_handles_bom_crlf_quotes_and_header_order(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        'failureType,trianTestLabel,filename\r\nCenter,Training,"a,b.png"\r\n'
        'none,Test,none.png\r\n,Training,unlabeled.png\r\n',
        encoding="utf-8-sig",
    )
    assert parse_wm811k_labeled_samples(path) == [("a,b.png", 1), ("none.png", 0)]
    assert parse_wm811k_labeled_samples(path, split="training") == [("a,b.png", 1)]


@pytest.mark.parametrize("label", ["9", "-1", "Unrecognized"])
def test_invalid_raw_labels_fail_before_cuda(tmp_path, label):
    path = tmp_path / "labels.csv"
    path.write_text(f"filename,failureType\na.png,{label}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 2"):
        parse_wm811k_labeled_samples(path)


@pytest.mark.parametrize(
    "text,message",
    [
        ("filename,failureType\na.png\n", "Malformed CSV row 2"),
        ("filename,failureType\n,Center\n", "Empty filename at row 2"),
    ],
)
def test_invalid_raw_csv_rows_fail_clearly(tmp_path, text, message):
    path = tmp_path / "labels.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parse_wm811k_labeled_samples(path)


def test_raw_split_request_requires_explicit_split_column(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("filename,failureType\na.png,Center\n", encoding="utf-8")
    with pytest.raises(ValueError, match="requires trianTestLabel"):
        parse_wm811k_labeled_samples(path, split="training")


def test_raw_preflight_rejects_missing_image(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels.csv").write_text("filename,failureType\nmissing.png,Center\n")
    dataset = WaferWM811KDataset(tmp_path, image_size=16)
    with pytest.raises(ValueError, match="missing.png"):
        validate_dataset_images(dataset)


def test_raw_encoding_is_preserved(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels.csv").write_text("filename,failureType\na.png,Center\n")
    image = np.resize(np.array([0, 1, 2], dtype=np.uint8), (16, 16))
    Image.fromarray(image).save(tmp_path / "images" / "a.png")
    dataset = WaferWM811KDataset(tmp_path, image_size=16)
    assert validate_dataset_images(dataset) == 1
    tensor = dataset[0]["image"]
    assert torch.equal(tensor.argmax(dim=0), torch.from_numpy(image))
    assert torch.allclose(tensor.sum(dim=0), torch.ones(16, 16))


def test_prepared_real_author_order_and_split_remain_unchanged():
    config = EngineConfig.from_yaml("papers/vmamba/configs/config.yaml")
    path = Path(config.get("data.data_root"))
    if not path.is_file():
        pytest.skip("Author archive is not installed; test does not download it")
    dataset = build_dataset(config)
    with zipfile.ZipFile(path) as archive:
        expected = sorted(
            (name, FOLDER_LABELS[Path(name).parent.name.lower()])
            for name in archive.namelist()
            if Path(name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            and Path(name).parent.name.lower() in FOLDER_LABELS
        )
    assert dataset._samples == expected
    labels = [label for _, label in expected]
    assert np.bincount(labels, minlength=9).tolist() == [100, 100, 102, 103, 102, 100, 95, 100, 100]
    train, holdout = stratified_split(list(range(len(dataset))), labels, seed=42)
    assert (len(train), len(holdout)) == (721, 181)
    assert not set(train).intersection(holdout)
    assert validate_dataset_images(dataset) == 902
