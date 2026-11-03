"""Locked prospective BCE architecture confirmation data preparation; no inference.

Only explicit --stage download/prepare/check commands are available. Each first
verifies the complete model/code/threshold lock before any new image access.
Function bodies below reuse the historical C4 confirmation preparation with
controlled changes to 24 frames, 6144 selected patches and seven prior manifests.
Old target predictions and scores are never read. No model is instantiated.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import struct
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from common import (ROOT, G, SHARED, ORIGINAL, OLD_CONFIRM, PREVIOUS_GROUP, PRIOR,
    EVAL_PREFIX, sha, save, module, verify_lock)

NATIVE_HEIGHT, NATIVE_WIDTH, BORDER = 3536, 4096, 512
NOMINAL_OVERLAP_FRACTION = 0.10

REUSED_FUNCTION_NAMES = ("unchanged_helper", "download", "evaluation_subset", "prepare", "check")


def prior_hashes():
    assert len(PRIOR) == 7
    return {str((folder / "DATA_MANIFEST.json").relative_to(G)):
            sha(folder / "DATA_MANIFEST.json") for folder in PRIOR}


def frozen_plan():
    # verify_lock hashes locked weights/code; it does not instantiate any model.
    plan, selection = verify_lock()
    assert len(plan["test_frames"]) == len(set(plan["test_frames"])) == 24
    assert plan["excluded_frame_count"] == len(plan["excluded_previous_frames"]) == 110
    assert not set(plan["test_frames"]) & set(plan["excluded_previous_frames"])
    assert plan["prior_manifests_sha256"] == prior_hashes()
    assert len(plan["prior_manifests_sha256"]) == 7
    assert plan["evaluation"]["target_examples"] == 6144
    assert plan["evaluation"]["patches_per_frame"] == 256
    assert plan["evaluation"]["selection_prefix"] == EVAL_PREFIX
    assert plan["evaluation"]["minimum_examples_each_class"] == 8
    assert plan["evaluation"]["minimum_frames_each_class"] == 3
    assert plan["evaluation"]["replacement_on_gate_failure"] is False
    assert plan["prepare_script_sha256"] == sha(Path(__file__))
    assert selection["source_plan_sha256"] == sha(ROOT / "SOURCE_PLAN.json")
    return plan


def unchanged_helper():
    helper = module("bce_confirmation_original_preprocessor", ORIGINAL / "prepare.py")
    helper.ROOT = ROOT
    return helper


def download():
    plan = frozen_plan()
    assert not (ROOT / "TEST.npz").exists(), "Do not download over a prepared cohort"
    remote = module("bce_confirmation_remote_zip", SHARED / "sem_remote_zip.py")
    directory = ROOT / "sources"
    directory.mkdir(exist_ok=True)
    attempts = []
    infos = None
    for attempt in range(1, 4):
        try:
            with zipfile.ZipFile(remote.RemoteZip()) as archive:
                infos = {frame: archive.getinfo(f"dataset/sems/m2/sem{frame:04}.png")
                         for frame in plan["test_frames"]}
            attempts.append(dict(stage="CENTRAL_DIRECTORY", attempt=attempt, status="COMPLETE"))
            break
        except Exception as error:
            attempts.append(dict(stage="CENTRAL_DIRECTORY", attempt=attempt,
                status="FAILED", error=str(error)))
            save(ROOT / "DOWNLOAD_ATTEMPTS.json", attempts)
    assert infos is not None, "Unable to read immutable remote archive directory"

    def fetch(frame):
        info = infos[frame]
        assert info.file_size <= plan["maximum_png_bytes_each"] and info.compress_type == 0
        path = directory / f"sem{frame:04}.png"
        if path.exists():
            raw = path.read_bytes()
        else:
            error = None
            for attempt in range(1, 4):
                try:
                    stream = remote.RemoteZip()
                    stream.seek(info.header_offset)
                    block = stream.read(min(30 + len(info.filename.encode()) + 2048 + info.file_size,
                                            remote.SIZE - info.header_offset))
                    header = struct.unpack("<4s5H3L2H", block[:30])
                    assert header[0] == b"PK\x03\x04"
                    name_size, extra_size = header[-2:]
                    assert block[30:30 + name_size].decode() == info.filename
                    start = 30 + name_size + extra_size
                    raw = block[start:start + info.file_size]
                    assert len(raw) == info.file_size and zlib.crc32(raw) == info.CRC
                    attempts.append(dict(stage="PNG", frame=frame, attempt=attempt, status="COMPLETE"))
                    break
                except Exception as current:
                    error = current
                    attempts.append(dict(stage="PNG", frame=frame, attempt=attempt,
                        status="FAILED", error=str(current)))
            else:
                raise RuntimeError(f"Unable to fetch unchanged frame{frame}") from error
            temporary = path.with_suffix(".png.partial")
            temporary.write_bytes(raw)
            temporary.replace(path)
        assert len(raw) == info.file_size and zlib.crc32(raw) == info.CRC
        return dict(frame=frame, file=str(path.relative_to(ROOT)), bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(), crc32=info.CRC)

    rows = []
    started = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = {pool.submit(fetch, frame): frame for frame in plan["test_frames"]}
            for future in as_completed(pending):
                rows.append(future.result())
                save(ROOT / "DOWNLOAD_PROGRESS.json", dict(completed=len(rows), total=24,
                    seconds=time.monotonic() - started, files=sorted(rows, key=lambda row: row["frame"])))
                print(json.dumps(dict(stage="DOWNLOAD", completed=len(rows), total=24)), flush=True)
    finally:
        save(ROOT / "DOWNLOAD_ATTEMPTS.json", attempts)
    with zipfile.ZipFile(SHARED / "labels.zip") as labels:
        for frame in plan["test_frames"]:
            raw = labels.read(f"dataset/labels/m2/label{frame:04}.svg")
            path = directory / f"label{frame:04}.svg"
            if path.exists():
                assert path.read_bytes() == raw
            else:
                path.write_bytes(raw)
    assert frozen_plan() == plan
    save(ROOT / "DOWNLOAD.json", dict(status="COMPLETE", files=sorted(rows, key=lambda row: row["frame"]),
        source_plan_sha256=sha(ROOT / "SOURCE_PLAN.json"), model_lock_sha256=sha(ROOT / "MODEL_LOCK.json"),
        selection_sha256=sha(ROOT / "SELECTION.json"), lock_verified_before_new_image_access=True,
        seconds=time.monotonic() - started))


def evaluation_subset(test, records, manifest, plan):
    selected = []
    frame_counts = {}
    for frame in plan["test_frames"]:
        candidates = [i for i, row in enumerate(records) if row["frame"] == frame]
        candidates.sort(key=lambda i: hashlib.sha256(
            (EVAL_PREFIX + records[i]["raw_D4_sha256"]).encode()).hexdigest())
        selected.extend(candidates[:256])
        frame_counts[str(frame)] = min(len(candidates), 256)
    ids = np.array(sorted(selected), dtype=np.int64)
    assert len(ids) == len(set(ids.tolist()))
    subset = {key: value[ids] for key, value in test.items()}
    counts = np.bincount(subset["label"], minlength=9)
    class_frames = {str(c): sorted(set(subset["frame"][subset["label"] == c].tolist())) for c in range(9)}
    gates = dict(exactly_6144=bool(len(ids) == 6144), exactly_256_each_frame=all(v == 256 for v in frame_counts.values()),
        min_8_each_class=bool(counts.min() >= 8), min_3_frames_each_class=all(len(v) >= 3 for v in class_frames.values()))
    np.savez_compressed(ROOT / "EVAL.npz", **subset)
    index = dict(status="PASS_EVAL_DATA_GATE" if all(gates.values()) else "INSUFFICIENT_FROZEN_COHORT",
        selection_prefix=EVAL_PREFIX, selection_rule="SHA256(prefix+raw_D4_sha256),first256/frame,then originalTESTorder",
        label_blind_selection=True, mask_not_used_in_ranking=True, replacement_on_gate_failure=False,
        local_TEST_indices=ids.tolist(), records=[records[int(i)] for i in ids], examples=len(ids),
        frame_counts=frame_counts, class_counts=counts.tolist(), class_frames=class_frames, gates=gates,
        EVAL_sha256=sha(ROOT / "EVAL.npz"), TEST_sha256=manifest["TEST_sha256"],
        source_plan_sha256=sha(ROOT / "SOURCE_PLAN.json"), protocol_sha256=plan["protocol_sha256"],
        prepare_script_sha256=plan["prepare_script_sha256"], model_lock_sha256=sha(ROOT / "MODEL_LOCK.json"),
        selection_sha256=sha(ROOT / "SELECTION.json"),
        scope="Fixed subset of 24 new frames; unselected patches share frames and are not another independent test")
    save(ROOT / "EVAL_INDEX.json", index)
    if not all(gates.values()):
        save(ROOT / "FAIL_EVAL_DATA_GATE.json", {key: value for key, value in index.items()
             if key not in ["records", "local_TEST_indices"]})
        raise RuntimeError("Fixed fresh EVAL cohort failed its gate; no replacement or resampling")
    return index


def prepare():
    plan = frozen_plan()
    assert not (ROOT / "TEST.npz").exists() and not (ROOT / "EVAL.npz").exists(), "Never overwrite a prepared cohort"
    downloaded = json.loads((ROOT / "DOWNLOAD.json").read_text())
    assert downloaded["status"] == "COMPLETE" and len(downloaded["files"]) == 24
    assert downloaded["source_plan_sha256"] == sha(ROOT / "SOURCE_PLAN.json")
    assert downloaded["model_lock_sha256"] == sha(ROOT / "MODEL_LOCK.json")
    assert downloaded["selection_sha256"] == sha(ROOT / "SELECTION.json")
    assert downloaded["lock_verified_before_new_image_access"] is True
    assert all(sha(ROOT / row["file"]) == row["sha256"] for row in downloaded["files"])
    prior_records = [row for folder in PRIOR for row in json.loads((folder / "DATA_MANIFEST.json").read_text())["records"]]
    prior_mask = {row["canonical_D4"] for row in prior_records}
    prior_raw = {row["raw_D4_sha256"] for row in prior_records}
    assert {row["frame"] for row in prior_records} <= set(plan["excluded_previous_frames"])
    used_mask, used_raw = prior_mask.copy(), prior_raw.copy()
    helper = unchanged_helper()
    records, xs, ys, labels, dedup = [], [], [], [], {}
    started = time.monotonic()
    for frame in plan["test_frames"]:
        x, y, coordinates = helper.patches(frame)
        keys, rawkeys = helper.canonical_keys(y), helper.canonical_keys(x, raw=True)
        kept, rows = [], []
        for i, (key, rawkey) in enumerate(zip(keys, rawkeys)):
            canonical, raw_sha = key.hex(), hashlib.sha256(rawkey).hexdigest()
            if canonical in used_mask or raw_sha in used_raw:
                continue
            used_mask.add(canonical)
            used_raw.add(raw_sha)
            kept.append(i)
            rows.append(dict(frame=frame, y=int(4 * coordinates[i, 0] + 512),
                x=int(4 * coordinates[i, 1] + 512), canonical_D4=canonical, raw_D4_sha256=raw_sha, role="test"))
        ids = np.array(kept, dtype=np.int64)
        classes = helper.class_labels(y[ids])
        for row, label in zip(rows, classes):
            row["label"] = int(label)
        records.extend(rows)
        xs.extend(x[ids].astype(np.float32) / (16 * 255))
        ys.extend(y[ids])
        labels.extend(classes)
        dedup[str(frame)] = dict(candidate_patches=len(x), excluded_duplicates=len(x) - len(ids), kept=len(ids))
        progress = dict(stage="PREPARE_FRESH_TEST", frame=frame, completed_frames=len(dedup), total_frames=24,
            unique_patches=len(records), seconds=time.monotonic() - started)
        save(ROOT / "PREPARE_PROGRESS.json", progress)
        print(json.dumps(progress), flush=True)
    test = dict(X=np.array(xs, dtype=np.float32)[:, None], Y=np.array(ys, dtype=np.uint8)[:, None],
        label=np.array(labels, dtype=np.int64), role=np.array(["test"] * len(records)),
        frame=np.array([row["frame"] for row in records], dtype=np.int64))
    assert set(test["frame"]) == set(plan["test_frames"])
    assert len({row["canonical_D4"] for row in records}) == len(records)
    assert len({row["raw_D4_sha256"] for row in records}) == len(records)
    assert not {row["canonical_D4"] for row in records} & prior_mask
    assert not {row["raw_D4_sha256"] for row in records} & prior_raw
    assert frozen_plan() == plan and prior_hashes() == plan["prior_manifests_sha256"]
    np.savez_compressed(ROOT / "TEST.npz", **test)
    manifest = dict(status="PREPARED_NEW_ONLY_TEST", records=records, test_frames=plan["test_frames"],
        TEST_sha256=sha(ROOT / "TEST.npz"), examples=len(records), deduplication=dedup,
        prior_manifests_sha256=plan["prior_manifests_sha256"],
        prior_union_unique_counts=dict(mask=len(prior_mask), raw=len(prior_raw)),
        sources_sha256={str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "sources").iterdir()) if path.is_file()},
        protocol_sha256=plan["protocol_sha256"], config_sha256=plan["config_sha256"],
        source_plan_sha256=sha(ROOT / "SOURCE_PLAN.json"), prepare_script_sha256=sha(Path(__file__)),
        model_lock_sha256=sha(ROOT / "MODEL_LOCK.json"), selection_sha256=sha(ROOT / "SELECTION.json"),
        lock_verified_before_new_image_access=True,
        reused_preparation_source_file=str((PREVIOUS_GROUP / "prepare.py").relative_to(G)),
        reused_preparation_source_sha256=plan["reused_prepare_source_sha256"],
        reused_function_bodies=list(REUSED_FUNCTION_NAMES),
        no_TRAIN_VAL_or_combined_archive=True, no_models_loaded=True,
        native_preprocessing=dict(image_height=NATIVE_HEIGHT, image_width=NATIVE_WIDTH,
            border_exclusion_pixels=BORDER, reduction_factor=4, native_patch_size=48,
            reduced_patch_size=12, reduced_stride=6, native_stride=24,
            nominal_author_field_overlap_fraction=NOMINAL_OVERLAP_FRACTION,
            border_exceeds_nominal_overlap_per_axis=bool(BORDER > NOMINAL_OVERLAP_FRACTION * max(NATIVE_HEIGHT, NATIVE_WIDTH)),
            physical_stage_coordinate_registration_verified=False,
            overlap_interpretation="512px border exceeds nominal10% of image width/height; physical overlap is not independently verified by stage coordinates"),
        scope="Fresh held-out frames of the same physical IC; raw/mask D4 motifs from all seven prior cohorts excluded")
    save(ROOT / "DATA_MANIFEST.json", manifest)
    index = evaluation_subset(test, records, manifest, plan)
    print(json.dumps(dict(status=index["status"], frames=plan["test_frames"], TEST_examples=len(records),
        EVAL_examples=index["examples"], gates=index["gates"], seconds=time.monotonic() - started)), flush=True)


def check():
    plan = frozen_plan()
    manifest = json.loads((ROOT / "DATA_MANIFEST.json").read_text())
    index = json.loads((ROOT / "EVAL_INDEX.json").read_text())
    checks = dict(source_plan_hash=manifest["source_plan_sha256"] == sha(ROOT / "SOURCE_PLAN.json"),
        TEST_hash=manifest["TEST_sha256"] == sha(ROOT / "TEST.npz"),
        EVAL_hash=index["EVAL_sha256"] == sha(ROOT / "EVAL.npz"),
        prior_metadata_unchanged=prior_hashes() == plan["prior_manifests_sha256"],
        frozen_model_and_code_lock=manifest["model_lock_sha256"] == index["model_lock_sha256"] == sha(ROOT / "MODEL_LOCK.json")
            and manifest["selection_sha256"] == index["selection_sha256"] == sha(ROOT / "SELECTION.json"),
        all_source_hashes=all(sha(ROOT / name) == value for name, value in manifest["sources_sha256"].items()),
        EVAL_gate=all(index["gates"].values()), no_combined_archives=not any((ROOT / name).exists() for name in ["TRAIN_VAL.npz", "DATA.npz"]))
    with np.load(ROOT / "TEST.npz", allow_pickle=False) as z:
        test = {key: z[key] for key in z.files}
    with np.load(ROOT / "EVAL.npz", allow_pickle=False) as z:
        evaluation = {key: z[key] for key in z.files}
    ids = np.array(index["local_TEST_indices"], dtype=np.int64)
    checks["EVAL_exact_subset"] = all(np.array_equal(evaluation[key], test[key][ids]) for key in test)
    checks["EVAL_exact6144_and256perframe"] = len(ids) == 6144 and all(int((evaluation["frame"] == frame).sum()) == 256 for frame in plan["test_frames"])
    checks["newframes_disjoint"] = not set(test["frame"]) & set(plan["excluded_previous_frames"])
    records = manifest["records"]
    selected = []
    for frame in plan["test_frames"]:
        eligible = [i for i, row in enumerate(records) if row["frame"] == frame]
        eligible.sort(key=lambda i: hashlib.sha256((EVAL_PREFIX + records[i]["raw_D4_sha256"]).encode()).hexdigest())
        selected.extend(eligible[:256])
    checks["selection_reproduces_label_blind_rank"] = ids.tolist() == sorted(selected)
    maximum = 0.
    native_geometry_ok = True
    for frame in plan["test_frames"]:
        image = np.asarray(Image.open(ROOT / "sources" / f"sem{frame:04}.png"))
        native_geometry_ok &= image.shape == (NATIVE_HEIGHT, NATIVE_WIDTH) and image.dtype == np.uint8
        for i in np.flatnonzero(evaluation["frame"] == frame):
            row = index["records"][int(i)]
            native_geometry_ok &= (BORDER <= row["y"] and row["y"] + 48 <= NATIVE_HEIGHT - BORDER and
                                   BORDER <= row["x"] and row["x"] + 48 <= NATIVE_WIDTH - BORDER and
                                   (row["y"] - BORDER) % 24 == 0 and (row["x"] - BORDER) % 24 == 0)
            crop = image[row["y"]:row["y"] + 48, row["x"]:row["x"] + 48]
            reduced = crop.reshape(12, 4, 12, 4).sum((1, 3)).astype(np.float32) / (16 * 255)
            maximum = max(maximum, float(np.abs(reduced - evaluation["X"][i, 0]).max()))
    checks["native_geometry_and_512px_border_exclusion"] = bool(native_geometry_ok)
    checks["native_crop_reconstruction_bit_exact"] = maximum == 0.
    result = dict(status="PASS_PREPARE_CHECK" if all(checks.values()) else "FAIL_PREPARE_CHECK",
                  checks=checks, maximum_native_reconstruction_error=maximum, no_model_evaluation=True)
    save(ROOT / "PREPARE_CHECK.json", result)
    assert all(checks.values()), result
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["download", "prepare", "check"], required=True)
    args = parser.parse_args()
    {"download": download, "prepare": prepare, "check": check}[args.stage]()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
