"""Plot the recorded training minibatches of the five final model pairs.

No model is trained or evaluated here. Missing warmup geometry losses stay missing.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
G = REPO / "gipoteza"
ASSETS = HERE / "assets"
SEEDS = (5701, 5721, 5757, 5801, 5821)
ARCHITECTURES = {"group6": "GroupCNN", "untied24_single": "Untied24"}
LOGS = [
    G / "sem_arch_groupcnn_20261102/trials/groupcnn_screen_s5701_metadata_recovery_27b9055b42/progress.jsonl",
    G / "sem_arch_groupcnn_20261102/trials/groupcnn_replicate_s5721-5757_metadata_recovery_27b9055b42/progress.jsonl",
    G / "sem_arch_untied_control_20261103/trial/progress.jsonl",
    G / "sem_arch_seed_extension_20261103/progress.jsonl",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ASSETS.mkdir(exist_ok=True)
    records = {}
    for path in LOGS:
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            row = json.loads(line)
            arch, seed = row.get("architecture"), row.get("seed")
            if arch not in ARCHITECTURES or seed not in SEEDS or "BCE" not in row:
                continue
            if row.get("budget", 27) != 27:
                continue
            step = row["step"]
            key = (arch, seed, step)
            assert key not in records, key
            bce, geom = row["BCE"], row["geometry_loss"]
            assert np.isfinite(bce) and bce >= 0
            assert (geom is None) == (step <= 400)
            if geom is not None:
                assert np.isfinite(geom) and geom >= 0
            total = bce + (0 if geom is None else geom)
            if "total_loss" in row:
                assert abs(total - row["total_loss"]) < 2e-7
            records[key] = dict(
                architecture=ARCHITECTURES[arch], seed=seed, step=step,
                BCE=bce, geometry_loss=geom,
                source=str(path.relative_to(REPO)), line=line_number,
            )
    steps = np.arange(200, 2001, 200)
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            assert sorted(k[2] for k in records if k[:2] == (arch, seed)) == steps.tolist()
    assert len(records) == 100
    rows = [records[key] for key in sorted(records)]
    csv_path = ASSETS / "training_curves.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#9ba8b2", "axes.labelcolor": "#202a32",
        "xtick.color": "#4b5862", "ytick.color": "#4b5862",
        "axes.titleweight": "normal", "axes.titlesize": 11,
    })
    colors = {"group6": "#21618c", "untied24_single": "#bd5b29"}
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.8), sharex=True)
    means = {}
    for ax, metric, title in zip(
        axes, ("BCE", "geometry_loss"),
        ("Пиксельная ошибка Lseg (BCE)", "Геометрическая ошибка Lgeom"),
    ):
        ax.axvspan(0, 400, color="#eaf0f4", zorder=0)
        ax.axvline(400, color="#6e7d88", lw=1, linestyle=":")
        for arch in ARCHITECTURES:
            values = np.array([
                [np.nan if records[(arch, seed, int(step))][metric] is None
                 else records[(arch, seed, int(step))][metric] for step in steps]
                for seed in SEEDS
            ])
            valid = ~np.isnan(values).all(axis=0)
            for curve in values:
                ax.plot(steps, curve, color=colors[arch], alpha=.25, lw=.85)
            mean = values[:, valid].mean(axis=0)
            ax.plot(steps[valid], mean, color=colors[arch], lw=2.2, marker="o", ms=3.5)
            means[f"{arch}/{metric}"] = dict(steps=steps[valid].tolist(), values=mean.tolist())
        ax.set_title(title, loc="left", pad=9)
        ax.set_ylabel("Значение ошибки")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#dce2e6", lw=.6)
        ax.set_axisbelow(True)
    axes[0].text(200, .97, "Прогрев", ha="center", va="top", fontsize=8,
                 color="#64717b", transform=axes[0].get_xaxis_transform())
    axes[-1].set_xlim(0, 2050)
    axes[-1].set_xticks(np.arange(0, 2001, 400))
    axes[-1].set_xlabel("Шаг обучения (всего 2000)")
    legend = [Line2D([0], [0], color=colors[a], lw=2.2, marker="o", ms=3.5,
                     label=ARCHITECTURES[a]) for a in ARCHITECTURES]
    fig.legend(handles=legend, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.52, 1.005))
    fig.subplots_adjust(left=.12, right=.985, bottom=.09, top=.92, hspace=.27)
    figure = ASSETS / "training_curves.png"
    fig.savefig(figure, dpi=240, facecolor="white")
    plt.close(fig)
    checks = dict(
        records=100, seeds=list(SEEDS), training_runs=10,
        steps=steps.tolist(), dense_masks=27, weak_labels=1152,
        warmup_steps=400, first_geometry_update=401,
        first_logged_geometry_step=600,
        values="current training minibatch loss before optimizer update",
        validation_time_series=False, smoothing=False,
        thick_lines="arithmetic mean of five runs at matching logged steps",
        thin_lines="individual runs; no confidence interval",
        source_sha256={str(p.relative_to(REPO)): sha(p) for p in LOGS},
        means=means,
        output_sha256={p.name: sha(p) for p in (figure, csv_path)},
    )
    (ASSETS / "TRAINING_CURVES_CHECKS.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(records=len(rows), figure=str(figure), means=means)))


if __name__ == "__main__":
    main()
