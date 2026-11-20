"""Actual SEM frame and SVG mask split into aligned image/target patches."""
from pathlib import Path
import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/sem_supervisor_figure_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'sem_arch_calibrated_confirmation_20261103'
ASSETS = ROOT / 'assets'


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ASSETS.mkdir(exist_ok=True)
    metadata = json.loads((SOURCE / 'EVAL_INDEX.json').read_text())
    candidates = [(i, r) for i, r in enumerate(metadata['records']) if r['frame'] == 11]
    chosen = [candidates[j] for j in [0, len(candidates) // 2, len(candidates) - 1]]
    png, svg = SOURCE / 'sources/sem0011.png', SOURCE / 'sources/label0011.svg'
    image = np.asarray(Image.open(png)); assert image.shape == (3536, 4096)
    canvas = Image.new('L', (4096, 3536)); draw = ImageDraw.Draw(canvas)
    for element in ET.parse(svg).iter():
        assert 'transform' not in element.attrib
        if element.tag.endswith('}path'):
            data = element.attrib['d']; assert set(re.findall('[A-Za-z]', data)) <= set('MLZz')
            points = [(int(x), int(y)) for x, y in re.findall(r'[ML](\d+),(\d+) ', data)]
            assert len(points) >= 3; draw.polygon(points, fill=1)
    mask = np.array(canvas, dtype=bool)
    patches, checks = [], []
    with np.load(SOURCE / 'EVAL.npz', allow_pickle=False) as z:
        for index, row in chosen:
            x, y = row['x'], row['y']
            raw = image[y:y+48, x:x+48]; target48 = mask[y:y+48, x:x+48]
            x12 = raw.reshape(12, 4, 12, 4).sum((1, 3)).astype(np.float32) / np.float32(16 * 255)
            y12 = target48.reshape(12, 4, 12, 4).mean((1, 3)) >= .5
            error = float(np.max(np.abs(x12 - z['X'][index, 0])))
            assert error < 1e-7 and np.array_equal(y12, z['Y'][index, 0].astype(bool))
            patches.append((raw, target48, x12, y12))
            checks.append(dict(EVAL_index=index, frame=11, x=x, y=y, input_max_error=error,
                               majority_target_exact=True, class_label=int(z['label'][index])))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig = plt.figure(figsize=(11.2, 9.2), layout='constrained')
    grid = fig.add_gridspec(4, 4, height_ratios=[2.1, 1, 1, 1])
    colors = ['#e56534', '#1e9b92', '#6d5eba']
    ax0, ax1 = fig.add_subplot(grid[0, :2]), fig.add_subplot(grid[0, 2:])
    for ax, array, title in [(ax0, image, 'Исходный SEM-кадр 4096 × 3536'),
                             (ax1, mask, 'Его SVG-разметка → бинарная маска')]:
        ax.imshow(array, cmap='gray', interpolation='nearest', vmin=0, vmax=255 if array is image else 1)
        ax.set_title(title, fontsize=12, pad=8)
        for n, (_, row) in enumerate(chosen):
            x, y = row['x'], row['y']
            ax.add_patch(Rectangle((x, y), 48, 48, fill=False, linewidth=1.7, edgecolor=colors[n]))
            ax.annotate(str(n+1), (x+24, y+24), (x+320, y+20), fontsize=11, color='white',
                        ha='center', va='center', bbox=dict(boxstyle='circle,pad=.25', fc=colors[n], ec='white'),
                        arrowprops=dict(arrowstyle='-', color=colors[n], lw=1.5))
        ax.set_xticks([]); ax.set_yticks([])
    columns = ['Фрагмент SEM 48 × 48', 'Та же область маски 48 × 48',
               'Вход CNN 12 × 12', 'Целевая маска 12 × 12']
    for n, values in enumerate(patches):
        for col, array in enumerate(values):
            ax = fig.add_subplot(grid[n+1, col])
            ax.imshow(array, cmap='gray', interpolation='nearest', vmin=0,
                      vmax=255 if col == 0 else 1)
            ax.set_xticks([]); ax.set_yticks([])
            if n == 0: ax.set_title(columns[col], fontsize=10, pad=8)
            if col == 0: ax.set_ylabel(f'Пара {n+1}', color=colors[n], fontsize=11, labelpad=10)
            for spine in ax.spines.values(): spine.set_edgecolor(colors[n]); spine.set_linewidth(1.2)
    fig.suptitle('Изображения вырезаются из SEM-кадра; маски — по тем же координатам', fontsize=13)
    fig.savefig(ASSETS / 'mask_pairs.png', dpi=200, bbox_inches='tight'); plt.close(fig)
    evidence = dict(status='PASS_MASK_PAIR_RECONSTRUCTION', frame=11,
        selection='First, middle and last EVAL rows of frame11; no model/error-based selection',
        checks=checks, source_sha256={str(p.relative_to(ROOT.parent)): sha(p)
            for p in [png, svg, SOURCE / 'EVAL.npz', SOURCE / 'EVAL_INDEX.json']},
        figure_sha256=sha(ASSETS / 'mask_pairs.png'))
    (ASSETS / 'FIGURE_CHECKS.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=evidence['status'], checks=checks), ensure_ascii=False))


if __name__ == '__main__': main()
