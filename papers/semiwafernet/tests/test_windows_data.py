"""CSV, input validation and SMOTE checks without a CUDA device."""

import csv

import numpy as np
import pytest
from PIL import Image

from papers.semiwafernet.data_utils.wafer_dataset import (
    SMOTEDataset,
    UnlabeledWM811KDirectoryDataset,
    _resolve_image_path,
    die_map_to_onehot,
    inspect_wm811k_labels,
    parse_wm811k_labeled_rows,
    parse_wm811k_unlabeled_filenames,
    validate_image_inventory,
)


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(header)
        writer.writerows(rows)


def test_windows_csv_bom_crlf_quoted_fields_and_split(tmp_path):
    path = tmp_path / "labels.csv"
    _write_csv(
        path,
        ["filename", "lotName", "trianTestLabel", "failureType"],
        [
            ["wafer, one.png", "lot\nwith line", "Training", "none"],
            ["test.png", "test", "Test", "Scratch"],
            ["unlabeled, one.png", "u", "", ""],
            ["unlabeled_nan.png", "u", "", "nan"],
        ],
    )
    assert inspect_wm811k_labels(path) == {
        "format": "official", "has_failure_type": True,
        "has_official_split": True, "unlabeled_count": 2,
    }
    assert parse_wm811k_labeled_rows(path, "training") == [("wafer, one.png", 0)]
    assert parse_wm811k_labeled_rows(path, "test") == [("test.png", 8)]
    assert parse_wm811k_unlabeled_filenames(path) == [
        "unlabeled, one.png", "unlabeled_nan.png"
    ]


def test_simple_csv_uses_column_names_instead_of_column_position(tmp_path):
    path = tmp_path / "labels.csv"
    _write_csv(path, ["label", "filename"], [["Center", "one, wafer.png"], ["8", "two.png"]])
    assert parse_wm811k_labeled_rows(path) == [("one, wafer.png", 1), ("two.png", 8)]
    with pytest.raises(ValueError, match="no trianTestLabel"):
        parse_wm811k_labeled_rows(path, "training")


@pytest.mark.parametrize("label", ["-1", "9", "unknown"])
@pytest.mark.parametrize("official", [False, True])
def test_invalid_labels_fail_before_cross_entropy(tmp_path, label, official):
    path = tmp_path / "labels.csv"
    if official:
        _write_csv(path, ["filename", "failureType", "trianTestLabel"], [["one.png", label, "Training"]])
    else:
        _write_csv(path, ["image", "label"], [["one.png", label]])
    with pytest.raises(ValueError, match="row 2"):
        parse_wm811k_labeled_rows(path)


@pytest.mark.parametrize("row", [["one.png", "none"], ["", "none", "Training"]])
def test_malformed_official_rows_fail_with_location(tmp_path, row):
    path = tmp_path / "labels.csv"
    _write_csv(path, ["filename", "failureType", "trianTestLabel"], [row])
    with pytest.raises(ValueError, match="row 2"):
        parse_wm811k_labeled_rows(path, "training")


def test_unlabeled_sampling_preserves_seeded_manifest_order(tmp_path):
    path = tmp_path / "labels.csv"
    filenames = [f"unlabeled_{index}.png" for index in range(20)]
    _write_csv(path, ["filename", "failureType"], [[name, ""] for name in filenames])
    indices = np.random.RandomState(137).choice(len(filenames), size=7, replace=False)
    expected = [filenames[index] for index in sorted(indices.tolist())]
    assert parse_wm811k_unlabeled_filenames(path, max_samples=7, seed=137) == expected
    with pytest.raises(ValueError, match="nonnegative"):
        parse_wm811k_unlabeled_filenames(path, max_samples=-1)


def test_smote_preload_preserves_encoded_features_and_seed(tmp_path):
    from imblearn.over_sampling import SMOTE

    (tmp_path / "images").mkdir()
    samples, encoded = [], []
    for index in range(7):
        die = np.zeros((4, 4), dtype=np.uint8)
        die[index % 4, :] = 2
        filename = f"{index}.png"
        Image.fromarray(die).save(tmp_path / "images" / filename)
        samples.append((filename, int(index >= 5)))
        encoded.append(die_map_to_onehot(die).numpy().reshape(-1))
    labels = np.asarray([label for _, label in samples])
    expected_x, expected_y = SMOTE(k_neighbors=1, random_state=137).fit_resample(
        np.stack(encoded), labels
    )
    dataset = SMOTEDataset(
        tmp_path, samples, image_size=4, num_classes=2, seed=137, augment=False
    )
    assert dataset._X.dtype == np.float32
    assert np.array_equal(dataset._X.reshape(len(dataset), -1), expected_x)
    assert np.array_equal(dataset._y, expected_y)
    assert np.allclose(dataset._X.sum(axis=1), 1)


@pytest.mark.parametrize(
    ("samples", "kwargs", "message"),
    [
        ([], {}, "empty training set"),
        ([("not-read.png", 9)], {}, "within 0..8"),
        ([("not-read.png", 0)], {"method": "bad"}, "Unknown oversample"),
        ([("not-read.png", 0)], {"smote_k_neighbors": 0}, "must be positive"),
        ([("not-read.png", 0)], {}, "at least two samples"),
    ],
)
def test_invalid_oversample_inputs_fail_before_image_io(tmp_path, samples, kwargs, message):
    with pytest.raises(ValueError, match=message):
        SMOTEDataset(tmp_path, samples, **kwargs)


def test_image_resolution_supports_nested_names_and_png_fallback(tmp_path):
    (tmp_path / "subdirectory").mkdir()
    nested = tmp_path / "subdirectory" / "one.png"
    fallback = tmp_path / "two.png"
    nested.touch()
    fallback.touch()
    assert _resolve_image_path(tmp_path, "subdirectory/one.png") == nested
    assert _resolve_image_path(tmp_path, "two.jpg") == fallback


def test_flat_inventory_scans_once_without_per_image_resolution(tmp_path, monkeypatch):
    from papers.semiwafernet.data_utils import wafer_dataset

    (tmp_path / "one.png").touch()
    (tmp_path / "two.png").touch()
    (tmp_path / "directory.png").mkdir()

    def unexpected_resolution(*args):
        raise AssertionError("Flat manifests should use the directory inventory")

    monkeypatch.setattr(wafer_dataset, "_resolve_image_path", unexpected_resolution)
    assert validate_image_inventory(
        tmp_path, [("one.png", 0), ("two.jpg", 1), "one.png"]
    ) == 2
    with pytest.raises(FileNotFoundError, match="Missing 1 of 1"):
        validate_image_inventory(tmp_path, ["directory.png"])


def test_inventory_checks_nested_absolute_and_quoted_names_reports_missing(tmp_path):
    (tmp_path / "quoted, subdirectory").mkdir()
    nested = tmp_path / "quoted, subdirectory" / "one, wafer.png"
    nested.touch()
    absolute = tmp_path / "two.png"
    absolute.touch()
    assert validate_image_inventory(
        tmp_path, [("quoted, subdirectory/one, wafer.png", 0), str(absolute)]
    ) == 2
    with pytest.raises(FileNotFoundError) as failure:
        validate_image_inventory(
            tmp_path,
            ["missing_one.png", "quoted, subdirectory/missing.png", "missing_one.png", "missing_three.png", "missing_four.png"],
        )
    message = str(failure.value)
    assert "Missing 4 of 4" in message
    assert "missing_one.png" in message and "quoted, subdirectory/missing.png" in message
    assert "missing_three.png" in message and "missing_four.png" not in message


def test_external_unlabeled_maps_match_classifier_encoding_and_are_deterministic(tmp_path):
    die = np.zeros((6, 10), dtype=np.uint8)
    die[1:5, 2:8] = 1
    die[2:4, 4:6] = 2
    Image.fromarray(die).save(tmp_path / "one.png")
    dataset = UnlabeledWM811KDirectoryDataset(tmp_path, image_size=32)
    image = dataset[0]["image"]
    assert image.shape == (3, 32, 32)
    assert image.numpy().dtype == np.float32
    assert np.array_equal(image.numpy(), dataset[0]["image"].numpy())
    assert set(np.unique(image.numpy())) == {0, 1}
    assert np.array_equal(image.sum(dim=0).numpy(), np.ones((32, 32)))
    assert all(image[channel].sum() > 0 for channel in range(3))


def test_external_unlabeled_sampling_is_capped_seeded_sorted_and_unique(tmp_path):
    filenames = [f"map_{index:02d}.png" for index in range(20)]
    for filename in reversed(filenames):
        (tmp_path / filename).touch()
    (tmp_path / "ignored.txt").touch()
    (tmp_path / "ignored_directory.png").mkdir()
    dataset = UnlabeledWM811KDirectoryDataset(tmp_path, max_samples=7, seed=137)
    indices = np.random.RandomState(137).choice(len(filenames), size=7, replace=False)
    assert dataset._filenames == [filenames[index] for index in sorted(indices.tolist())]
    assert len(dataset) == len(set(dataset._filenames)) == 7
    assert dataset._filenames == UnlabeledWM811KDirectoryDataset(
        tmp_path, max_samples=7, seed=137
    )._filenames
    assert dataset._filenames != UnlabeledWM811KDirectoryDataset(
        tmp_path, max_samples=7, seed=42
    )._filenames


def test_external_unlabeled_directory_rejects_missing_or_empty_pool(tmp_path):
    with pytest.raises(FileNotFoundError, match="directory not found"):
        UnlabeledWM811KDirectoryDataset(tmp_path / "absent")
    with pytest.raises(ValueError, match="No unlabeled wafer images"):
        UnlabeledWM811KDirectoryDataset(tmp_path)
    with pytest.raises(ValueError, match="max_samples must be positive"):
        UnlabeledWM811KDirectoryDataset(tmp_path, max_samples=0)
