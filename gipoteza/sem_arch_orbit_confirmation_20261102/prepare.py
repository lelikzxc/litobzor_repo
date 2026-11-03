"""Guarded future C4 confirmation data preparation; no model code or fitting.

The default command writes a metadata-only draft. Downloading and processing
fresh images are separate explicit commands and require frozen PROTOCOL.md
and CONFIG.json. This module reads no old TEST arrays, predictions or scores.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import importlib.util
import json
import os
import re
import struct
import time
import zipfile
import zlib

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G = ROOT.parent
SHARED = G / "shared_data/ic_sem_m2"
PILOT = G / "sem_architecture_pilot_20261102"
ORIGINAL = G / "sem_confirmation_20261028"
PRIOR = [ORIGINAL, G / "sem_cldice_confirmation_20261030",
    G / "sem_cldice_expansion_20261102", G / "sem_scarcity_20261102", PILOT]
PREFIX = "SEM-ORBIT-C4-CONFIRM-20261102/"
EVAL_PREFIX = "SEM-ORBIT-C4-EVAL-20261102/"
EXPECTED_FRAMES = [137, 227, 111, 135, 213, 265, 205, 82]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prior_hashes():
    return {str((folder / "DATA_MANIFEST.json").relative_to(G)):
            sha(folder / "DATA_MANIFEST.json") for folder in PRIOR}


def draft_metadata():
    exclusion_path = PILOT / "SOURCE_PLAN.json"
    old = json.loads(exclusion_path.read_text())
    excluded = set(old["excluded_previous_frames"]) | set(old["test"])
    assert len(old["excluded_previous_frames"]) == 78 and len(old["test"]) == 8 and len(excluded) == 86
    listing_path = SHARED / "sem_zip_listing.json"
    listing = json.loads(listing_path.read_text())
    available = [int(match.group(1)) for row in listing
        if (match := re.fullmatch(r"dataset/sems/m2/sem(\d+)\.png", row["name"]))]
    assert len(available) == len(set(available)) == 321
    candidates = [frame for frame in available if frame not in excluded]
    candidates.sort(key=lambda frame: hashlib.sha256((PREFIX + str(frame)).encode()).hexdigest())
    frames = candidates[:8]
    assert frames == EXPECTED_FRAMES
    draft = dict(status="DRAFT_METADATA_ONLY_NO_FRESH_DATA_ACCESS", test_frames=frames,
        selection_prefix=PREFIX, frame_key="SHA256(prefix+unpadded_decimal_frame_id)",
        available_M2_frames=321, eligible_M2_frames=len(candidates),
        excluded_frame_count=len(excluded), excluded_previous_frames=sorted(excluded),
        exclusion_rule="archpilot SOURCE_PLAN excluded_previous_frames78 union its test8",
        exclusion_source_file=str(exclusion_path.relative_to(G)), exclusion_source_sha256=sha(exclusion_path),
        archive_listing_file=str(listing_path.relative_to(G)), archive_listing_sha256=sha(listing_path),
        prior_manifests_sha256=prior_hashes(),
        helper_prepare_sha256=sha(ORIGINAL / "prepare.py"),
        shared_sha256={name: sha(SHARED / name) for name in ["labels.zip", "sem_remote_zip.py"]},
        evaluation=dict(selection_prefix=EVAL_PREFIX, patches_per_frame=256, target_examples=2048,
            ranking="SHA256(prefix+raw_D4_sha256); image-derived key only, no mask or class label",
            minimum_examples_each_class=8, minimum_frames_each_class=3,
            replacement_on_gate_failure=False),
        preprocessing="unchanged historical:crop512 border,4x4sums,12x12window,stride6,mask>=.5",
        deduplication="canonical mask D4 plus raw-image D4 SHA against all5 prior manifests and within new cohort",
        outputs="8 native PNG/SVG pairs; new-only TEST.npz; fixed EVAL.npz; no combined TRAIN/VAL archive",
        scope="Draft of future same-IC confirmation; no downloads, fresh pixels, model access or evaluation")
    path = ROOT / "SOURCE_PLAN_DRAFT.json"
    if path.exists():
        assert json.loads(path.read_text()) == draft, "Metadata draft changed; never silently replace selection"
    else:
        save(path, draft)
    return draft


def frozen_plan():
    assert (ROOT / "PROTOCOL.md").is_file() and (ROOT / "CONFIG.json").is_file(), "Freeze protocol/config before fresh data access"
    draft = draft_metadata()
    c = json.loads((ROOT / "CONFIG.json").read_text())
    assert c["test_frames"] == draft["test_frames"]
    assert c.get("selection_prefix", PREFIX) == PREFIX
    assert c.get("evaluation_prefix", EVAL_PREFIX) == EVAL_PREFIX
    assert c.get("evaluation_per_frame", 256) == 256
    assert c.get("minimum_examples_each_class", 8) == 8
    assert c.get("minimum_frames_each_class", 3) == 3
    if "protocol_sha256" in c:
        assert c["protocol_sha256"] == sha(ROOT / "PROTOCOL.md")
    plan = dict(draft, status="FROZEN_FRESH_DATA_PLAN", draft_sha256=sha(ROOT / "SOURCE_PLAN_DRAFT.json"),
        protocol_sha256=sha(ROOT / "PROTOCOL.md"), config_sha256=sha(ROOT / "CONFIG.json"),
        prepare_script_sha256=sha(Path(__file__)),
        remote_zip_url="https://edmond.mpg.de/api/access/datafile/223856",
        maximum_png_bytes_each=10_000_000)
    path = ROOT / "SOURCE_PLAN.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan, "Frozen plan/source/protocol changed"
    else:
        save(path, plan)
    return plan


def unchanged_helper():
    helper = module("orbit_confirmation_original_preprocessor", ORIGINAL / "prepare.py")
    helper.ROOT = ROOT
    return helper


def download():
    plan = frozen_plan()
    assert not (ROOT / "TEST.npz").exists(), "Do not download over a prepared cohort"
    remote = module("orbit_confirmation_remote_zip", SHARED / "sem_remote_zip.py")
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
                save(ROOT / "DOWNLOAD_PROGRESS.json", dict(completed=len(rows), total=8,
                    seconds=time.monotonic() - started, files=sorted(rows, key=lambda row: row["frame"])))
                print(json.dumps(dict(stage="DOWNLOAD", completed=len(rows), total=8)), flush=True)
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
        source_plan_sha256=sha(ROOT / "SOURCE_PLAN.json"), seconds=time.monotonic() - started))


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
    gates = dict(exactly_2048=bool(len(ids) == 2048), exactly_256_each_frame=all(v == 256 for v in frame_counts.values()),
        min_8_each_class=bool(counts.min() >= 8), min_3_frames_each_class=all(len(v) >= 3 for v in class_frames.values()))
    np.savez_compressed(ROOT / "EVAL.npz", **subset)
    index = dict(status="PASS_EVAL_DATA_GATE" if all(gates.values()) else "FAIL_EVAL_DATA_GATE",
        selection_prefix=EVAL_PREFIX, selection_rule="SHA256(prefix+raw_D4_sha256),first256/frame,then originalTESTorder",
        label_blind_selection=True, mask_not_used_in_ranking=True, replacement_on_gate_failure=False,
        local_TEST_indices=ids.tolist(), records=[records[int(i)] for i in ids], examples=len(ids),
        frame_counts=frame_counts, class_counts=counts.tolist(), class_frames=class_frames, gates=gates,
        EVAL_sha256=sha(ROOT / "EVAL.npz"), TEST_sha256=manifest["TEST_sha256"],
        source_plan_sha256=sha(ROOT / "SOURCE_PLAN.json"), protocol_sha256=plan["protocol_sha256"],
        prepare_script_sha256=plan["prepare_script_sha256"],
        scope="Fixed subset of8newframes; unselected patches shareframes and are not another independent test")
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
    assert downloaded["status"] == "COMPLETE" and len(downloaded["files"]) == 8
    assert downloaded["source_plan_sha256"] == sha(ROOT / "SOURCE_PLAN.json")
    assert all(sha(ROOT / row["file"]) == row["sha256"] for row in downloaded["files"])
    prior_records = [row for folder in PRIOR for row in json.loads((folder / "DATA_MANIFEST.json").read_text())["records"]]
    prior_mask = {row["canonical_D4"] for row in prior_records}
    prior_raw = {row["raw_D4_sha256"] for row in prior_records}
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
        progress = dict(stage="PREPARE_FRESH_TEST", frame=frame, completed_frames=len(dedup), total_frames=8,
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
        no_TRAIN_VAL_or_combined_archive=True, no_models_loaded=True,
        scope="Fresh held-out frames of samephysicalIC; all5priorpilots'raw/maskD4motifs excluded")
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
        all_source_hashes=all(sha(ROOT / name) == value for name, value in manifest["sources_sha256"].items()),
        EVAL_gate=all(index["gates"].values()), no_combined_archives=not any((ROOT / name).exists() for name in ["TRAIN_VAL.npz", "DATA.npz"]))
    with np.load(ROOT / "TEST.npz", allow_pickle=False) as z:
        test = {key: z[key] for key in z.files}
    with np.load(ROOT / "EVAL.npz", allow_pickle=False) as z:
        evaluation = {key: z[key] for key in z.files}
    ids = np.array(index["local_TEST_indices"], dtype=np.int64)
    checks["EVAL_exact_subset"] = all(np.array_equal(evaluation[key], test[key][ids]) for key in test)
    checks["EVAL_exact2048_and256perframe"] = len(ids) == 2048 and all(int((evaluation["frame"] == frame).sum()) == 256 for frame in plan["test_frames"])
    checks["newframes_disjoint"] = not set(test["frame"]) & set(plan["excluded_previous_frames"])
    records = manifest["records"]
    selected = []
    for frame in plan["test_frames"]:
        eligible = [i for i, row in enumerate(records) if row["frame"] == frame]
        eligible.sort(key=lambda i: hashlib.sha256((EVAL_PREFIX + records[i]["raw_D4_sha256"]).encode()).hexdigest())
        selected.extend(eligible[:256])
    checks["selection_reproduces_label_blind_rank"] = ids.tolist() == sorted(selected)
    maximum = 0.
    for frame in plan["test_frames"]:
        image = np.asarray(Image.open(ROOT / "sources" / f"sem{frame:04}.png"))
        for i in np.flatnonzero(evaluation["frame"] == frame):
            row = index["records"][int(i)]
            crop = image[row["y"]:row["y"] + 48, row["x"]:row["x"] + 48]
            reduced = crop.reshape(12, 4, 12, 4).sum((1, 3)).astype(np.float32) / (16 * 255)
            maximum = max(maximum, float(np.abs(reduced - evaluation["X"][i, 0]).max()))
    checks["native_crop_reconstruction_bit_exact"] = maximum == 0.
    result = dict(status="PASS_PREPARE_CHECK" if all(checks.values()) else "FAIL_PREPARE_CHECK",
                  checks=checks, maximum_native_reconstruction_error=maximum, no_model_evaluation=True)
    save(ROOT / "PREPARE_CHECK.json", result)
    assert all(checks.values()), result
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["draft", "freeze-plan", "download", "prepare", "check"], default="draft")
    args = parser.parse_args()
    if args.stage == "draft":
        print(json.dumps(draft_metadata()), flush=True)
    elif args.stage == "freeze-plan":
        print(json.dumps(frozen_plan()), flush=True)
    else:
        {"download": download, "prepare": prepare, "check": check}[args.stage]()


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
