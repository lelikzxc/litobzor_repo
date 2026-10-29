#!/usr/bin/env python3
"""Small, TRAIN-only LithoSim availability/ordering audit; never extracts tar members.

The anchor labels below are a feasibility diagnostic, NOT validated electrical labels.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import urllib.request

BASE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(BASE / '.mplconfig'))
import numpy as np
from PIL import Image
from scipy import ndimage

SHARD_URL = 'https://huggingface.co/datasets/grandiflorum/LithoSim/resolve/main/lithosim/opc_metal/train_val/9987_to_9998.tar.gz'
PAPER_URL = 'https://www.cse.cuhk.edu.hk/~byu/papers/C300-NeurIPS2025-LithoSim.pdf'
SHARD = BASE / '9987_to_9998.tar.gz'
DOSES = [-.1, -.08, -.06, -.04, -.02, 0, .02, .04, .06, .08, .1]
FOCUSES = [-40, 0, 40]

def sha(data):
    if isinstance(data, Path): data = data.read_bytes()
    return hashlib.sha256(data).hexdigest()

def fetch():
    BASE.mkdir(parents=True, exist_ok=True)
    for url, path in [(SHARD_URL, SHARD), (PAPER_URL, BASE / 'LithoSim_NeurIPS2025.pdf')]:
        if not path.exists():
            raw = urllib.request.urlopen(url, timeout=60).read()
            path.write_bytes(raw)
        print(path.name, path.stat().st_size, sha(path), flush=True)
    pdf = (BASE / 'LithoSim_NeurIPS2025.pdf').read_bytes()
    raw = subprocess.run(['pdftotext', '-', '-'], input=pdf, stdout=subprocess.PIPE, check=True).stdout.decode()
    clean = re.sub(r'<latexit\b[^>]*>.*?</latexit>', '', raw, flags=re.S)
    (BASE / 'LithoSim_NeurIPS2025.txt').write_text(clean)

def image(tar, name):
    content = tar.extractfile(name).read()
    im = Image.open(io.BytesIO(content))
    assert im.mode == '1' and im.size == (4096, 4096), (name, im.mode, im.size)
    return np.asarray(im, dtype=bool)

def anchors(target):
    """Four extrema within deep interior of each target component; fixed before RI.

    Full-image boundary anchors are discarded. These are geometric correspondence
    probes, not netlist pins; component identity must NOT be called electrical net ID.
    """
    labels, count = ndimage.label(target, structure=np.ones((3, 3), int))
    depth = ndimage.distance_transform_edt(target)
    points, ids, component_info = [], [], []
    for ci, section in enumerate(ndimage.find_objects(labels), 1):
        region = labels[section] == ci
        area = int(region.sum())
        maximum = float(depth[section][region].max())
        if area < 64 or maximum < 3:
            continue
        core = region & (depth[section] >= max(2., .75 * maximum))
        yy, xx = np.where(core)
        if not len(yy): continue
        yy += section[0].start
        xx += section[1].start
        chosen = set()
        for index in [xx.argmin(), xx.argmax(), yy.argmin(), yy.argmax()]:
            y, x = int(yy[index]), int(xx[index])
            if min(y, x, target.shape[0]-1-y, target.shape[1]-1-x) >= 64:
                chosen.add((y, x))
        if len(chosen) < 2: continue
        points.extend(sorted(chosen)); ids.extend([ci] * len(chosen))
        component_info.append(dict(target_component=ci, area=area, maximum_interior_depth_px=maximum, anchors=sorted(chosen)))
    return np.asarray(points, int), np.asarray(ids, int), component_info, count

def correspondence(produced, points, ids):
    labels, count = ndimage.label(produced, structure=np.ones((3, 3), int))
    at = labels[points[:, 0], points[:, 1]]
    split_or_missing = {}
    for ci in np.unique(ids):
        z = at[ids == ci]
        split_or_missing[int(ci)] = bool((z == 0).any() or len(set(z[z > 0])) > 1)
    merges = set()
    for pi in np.unique(at[at > 0]):
        components = sorted(set(int(x) for x in ids[at == pi]))
        for i, a in enumerate(components):
            for b in components[i+1:]: merges.add((a, b))
    return split_or_missing, merges, count

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--fetch', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        fetch(); return
    assert SHARD.exists(), 'First run --fetch with public network access.'
    tar = tarfile.open(SHARD, 'r:gz')
    members = tar.getmembers()
    for member in members:
        name = PurePosixPath(member.name)
        assert not name.is_absolute() and '..' not in name.parts
        assert member.isfile() or member.isdir(), member.name
    groups = [m.name for m in members if m.isdir()]
    result = dict(shard_url=SHARD_URL, shard_sha256=sha(SHARD), shard_bytes=SHARD.stat().st_size,
                  split='Published OPC-metal train_val only; no test assets opened',
                  groups=groups, doses=DOSES, focuses=FOCUSES, image_mode='PIL mode1; numpy bool; white=True',
                  source_names=['layout.png', 'mask.png', 'source_simple.src'], geometry={}, curves=[], anchor_specs={},
                  anchor_label_status='Feasibility proxy only: geometric anchor connectivity, not validated electrical open/short truth',
                  anchor_rule='Target CC8 area>=64,maxEDT>=3;four coordinate extrema in EDT>=max(2,.75maxEDT);anchors>=64px from frame;at least2/component',
                  limitations=['One small lexicographically last TRAIN shard, not a representative prevalence sample',
                               'Target components are geometric features, not known electrical nets',
                               'No physical semantic validation of anchor classes',
                               'Mode1 dose direction is empirical filename ordering; dose sign is not yet physically resolved',
                               'CC8 chosen for prototype; sensitivity to digital connectivity remains unvalidated'])
    classes = {'neither': 0, 'split_or_missing': 0, 'unintended_merge': 0, 'both': 0}
    total_added = total_removed = total_comparisons = 0
    image_violations = component_violations = pair_violations = 0
    informative_component_curves = informative_pair_curves = 0
    for group in groups:
        raw = {name: tar.extractfile(f'{group}/{name}').read() for name in result['source_names']}
        target, mask = image(tar, f'{group}/layout.png'), image(tar, f'{group}/mask.png')
        points, ids, info, target_cc = anchors(target)
        assert len(points)
        result['anchor_specs'][group] = info
        result['geometry'][group] = dict(target_file_sha256=sha(raw['layout.png']), target_pixel_sha256=sha(target.tobytes()),
                                        mask_file_sha256=sha(raw['mask.png']), source_sha256=sha(raw['source_simple.src']),
                                        target_cc8=target_cc, components_with_anchors=len(info), anchor_points=len(points),
                                        target_white_fraction=float(target.mean()), mask_white_fraction=float(mask.mean()))
        for focus in FOCUSES:
            previous = None; added = removed = 0; states = []; comp = []; pairs = []; fractions = []
            for dose in DOSES:
                produced = image(tar, f'{group}/RI_dose_{dose if dose else 0}_defocus_{focus}.png')
                fractions.append(float(produced.mean()))
                if previous is not None:
                    added += int(np.count_nonzero(~previous & produced))
                    removed += int(np.count_nonzero(previous & ~produced))
                    total_comparisons += 1
                previous = produced
                failures, merges, cc = correspondence(produced, points, ids)
                opening, bridging = any(failures.values()), bool(merges)
                state = 'both' if opening and bridging else 'split_or_missing' if opening else 'unintended_merge' if bridging else 'neither'
                classes[state] += 1; states.append(state); comp.append(failures); pairs.append(merges)
            # Empirically white features mostly shrink as filename dose grows:
            # splitting/missing should be nondecreasing; merging nonincreasing.
            image_open = [any(x.values()) for x in comp]; image_bridge = [bool(x) for x in pairs]
            image_violation = sum(a and not b for a, b in zip(image_open, image_open[1:])) + sum(not a and b for a, b in zip(image_bridge, image_bridge[1:]))
            local_component_violations = local_pair_violations = 0
            for ci in comp[0]:
                y = [x[ci] for x in comp]
                informative_component_curves += len(set(y)) > 1
                local_component_violations += sum(a and not b for a, b in zip(y, y[1:]))
            all_pairs = set().union(*pairs)
            for pair in all_pairs:
                y = [pair in x for x in pairs]
                informative_pair_curves += len(set(y)) > 1
                local_pair_violations += sum(not a and b for a, b in zip(y, y[1:]))
            curve = dict(group=group, focus=focus, white_fractions_by_dose=fractions,
                         added_white_pixels_as_filename_dose_increases=added, removed_white_pixels=removed,
                         exact_nesting_white_shrinking=added == 0, exact_nesting_white_expanding=removed == 0,
                         image_anchor_states=states, image_anchor_monotonicity_violations=image_violation,
                         component_anchor_monotonicity_violations=local_component_violations,
                         pair_anchor_monotonicity_violations=local_pair_violations,
                         per_component_split_or_missing_by_dose={str(ci): [x[ci] for x in comp] for ci in comp[0]},
                         merged_target_pairs_by_dose=[[list(p) for p in sorted(x)] for x in pairs])
            result['curves'].append(curve)
            total_added += added; total_removed += removed
            image_violations += image_violation; component_violations += local_component_violations; pair_violations += local_pair_violations
        print(f'Analyzed {group}: {len(points)} anchors, {len(info)} components', flush=True)
    result['summary'] = dict(n_folders=len(groups), unique_target_geometries=len(set(x['target_pixel_sha256'] for x in result['geometry'].values())),
                           n_focus_curves=len(result['curves']), n_RI_images=sum(classes.values()), n_adjacent_dose_comparisons=total_comparisons,
                           n_exact_white_shrinking_curves=sum(x['exact_nesting_white_shrinking'] for x in result['curves']),
                           white_added_pixels=total_added, white_removed_pixels=total_removed,
                           added_fraction_of_all_transition_pixels=total_added/(total_comparisons*4096*4096),
                           added_fraction_of_changed_pixels=total_added/(total_added+total_removed),
                           image_anchor_classes=classes, image_anchor_monotonicity_violations=image_violations,
                           component_anchor_monotonicity_violations=component_violations,
                           pair_anchor_monotonicity_violations=pair_violations,
                           informative_component_curves=informative_component_curves, informative_pair_curves=informative_pair_curves)
    # Six-panel TRAIN-only illustration. Fixed group9988; ROI chosen by maximal
    # changed-pixel count in a fixed nonoverlapping512px grid. No model involved.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    group='9988'; target=image(tar,f'{group}/layout.png'); mask=image(tar,f'{group}/mask.png')
    low=image(tar,f'{group}/RI_dose_-0.1_defocus_0.png'); nominal=image(tar,f'{group}/RI_dose_0_defocus_0.png'); high=image(tar,f'{group}/RI_dose_0.1_defocus_0.png')
    change=low!=high
    counts=change.reshape(8,512,8,512).sum(axis=(1,3)); yi,xi=np.unravel_index(counts.argmax(),counts.shape)
    yy,xx=int(yi*512),int(xi*512); sl=np.s_[yy:yy+512,xx:xx+512]
    fig,axes=plt.subplots(2,3,figsize=(12,8),constrained_layout=True)
    for ax,a,title in zip(axes.ravel(),[target,mask,low,nominal,high,change],['Target layout','OPC mask','RI filename dose -0.10','RI dose 0.00','RI filename dose +0.10','White-set change: low vs high']):
        ax.imshow(a[sl],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title);ax.set_xlabel('native pixel x');ax.set_ylabel('native pixel y')
    fig.suptitle(f'LithoSim TRAIN-only audit: folder{group}, focus0, native ROI x={xx},y={yy},512x512\nWhite geometry shrinks with filename dose; no electrical label claimed')
    fig.savefig(BASE/'six_panel_train.png',dpi=150);plt.close(fig)
    result['gallery']=dict(file='six_panel_train.png',folder=group,focus=0,roi_xywh=[xx,yy,512,512],rule='max changed pixel count among fixed512px grid tiles; TRAIN only')
    result['script_sha256']=sha(Path(__file__))
    (BASE/'AUDIT_RESULTS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(archive_url=SHARD_URL, paper_url=PAPER_URL, files_sha256={p.name:sha(p) for p in BASE.iterdir() if p.is_file() and p.name!='CHECKSUMS.json'})
    (BASE/'CHECKSUMS.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2),flush=True)

if __name__=='__main__': main()
