"""Generate a development-only report from saved JSON, without model execution."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path, root=ROOT):
    return Path(os.path.relpath(path, root)).as_posix()


def percent(value):
    return f"{100 * value:.2f}"


def pp(value):
    return f"{100 * value:+.2f}"


def passed(value):
    return "PASS" if value else "FAIL"


def mean(rows, arm=None, key="mask_F1"):
    return float(np.mean([row["selected"][key] for row in rows
                          if arm is None or row["arm"] == arm]))


def discover(root):
    ledger = read(root / "LEDGER.json", {"events": []})
    order = {}
    for event in ledger["events"]:
        if event.get("trial"):
            order.setdefault(Path(event["trial"]).name, len(order))
    trials = []
    for path in (root / "trials").glob("*"):
        if path.is_dir():
            trials.append({"path": path, **{key: read(path / name, default) for key, name, default in [
                ("frozen", "FROZEN.json", {}), ("result", "RESULT.json", None),
                ("aggregate", "AGGREGATE_RESULT.json", None), ("failure", "FAILURE.json", None),
                ("partial", "PARTIAL_RESULTS.json", []), ("audit", "AUDIT.json", None)]}})
    return sorted(trials, key=lambda trial: (order.get(trial["path"].name, float("inf")), trial["path"].name))


def actual_rows(trials):
    return [row for trial in trials for row in (trial["result"]["rows"] if trial["result"] else trial["partial"])]


def views(trials):
    complete = [trial for trial in trials if trial["result"]]
    aggregate = [trial for trial in complete if trial["aggregate"]]
    if aggregate:
        trial = aggregate[-1]
        return [("all 3 seeds", trial["aggregate"]), ("new 2 seeds", trial["result"])]
    return [(f"{trial['result']['phase']}: {trial['result']['seeds']}", trial["result"]) for trial in complete]


def verify_control(result, control):
    """Validate one control row per exact unit and equal starts/batches/steps."""
    units = {(row["budget"], row["seed"]) for row in result["rows"]}
    controls = [row for row in control["rows"] if (row["budget"], row["seed"]) in units]
    assert len(controls) == len(units)
    by_unit = {(row["budget"], row["seed"]): row for row in controls}
    assert set(by_unit) == units
    assert result["high_lr_controls"] == [row for row in result["high_lr_controls"]
                                          if (row["budget"], row["seed"]) in units]
    copied = {(row["budget"], row["seed"]): row for row in result["high_lr_controls"]}
    assert len(result["high_lr_controls"]) == len(units) and copied == by_unit
    for row in result["rows"]:
        reference = by_unit[row["budget"], row["seed"]]
        for field in ["backbone_before_sha256", "batch_sequence_sha256", "steps"]:
            assert row[field] == reference[field], field
    for arm in ["continued_base_low_lr", "plain", "joint"]:
        rows = [row for row in result["rows"] if row["arm"] == arm]
        assert len(rows) == len(units) and {(row["budget"], row["seed"]) for row in rows} == units
    return controls


def figure(root, result_views, control):
    if not result_views:
        return None
    output = root / "figures" / "joint_adaptation_validation.png"
    output.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(len(result_views));width = 0.18
    for offset, arm, label, color in [
        (-1.5, "continued_base_low_lr", "CNN lr=0.0001", "#777777"),
        (-0.5, "plain", "plain", "#3c78a8"),
        (0.5, "joint", "joint", "#b25a43"),
        (1.5, None, "shared CNN lr=0.0005", "#90734e")]:
        values = [100 * mean(result["rows"], arm) if arm else
                  100 * mean(verify_control(result, control)) for _, result in result_views]
        bars = axes[0].bar(x + offset * width, values, width, label=label, color=color)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=7, padding=3)
    axes[0].set_ylabel("Validation Macro-F1, %")
    axes[0].set_ylim(0, min(100, max(bar.get_height() for bar in axes[0].patches) + 9))
    axes[0].set_title("Selected thresholds; equal budget weights")
    axes[0].legend(fontsize=8, loc="lower left")
    for offset, field, label, color, marker in [
        (-0.16, "vs_low_lr", "A joint − CNN lr=0.0001", "#b25a43", "o"),
        (0, "vs_high_lr", "A joint − shared CNN lr=0.0005", "#90734e", "x"),
        (0.16, None, "B joint − plain", "#4d8d63", "D")]:
        values = [100 * (result["gates"]["A_architecture"]["joint"][field]["delta"] if field
                         else result["gates"]["B_physics"]["delta"]) for _, result in result_views]
        axes[1].scatter(x + offset, values, s=50, marker=marker, color=color, label=label)
    axes[1].axhline(0, lw=.8, color="black")
    axes[1].axhline(.5, lw=.8, color="gray", ls="--", label="Mean-gain gate: +0.5 pp")
    axes[1].set_ylabel("Macro-F1 difference, percentage points")
    axes[1].set_title("Development point estimates; no independent TEST")
    axes[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2)
    for axis in axes:
        axis.set_xticks(x, [label for label, _ in result_views])
        axis.set_xlim(-.6, len(x) - .4)
    fig.suptitle("Joint CNN adaptation on repeatedly used validation data", fontsize=11)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def build(root=ROOT):
    config = read(root / "CONFIG.json")
    trials = discover(root)
    previous = []
    for link in config["prior_stage_links"]:
        path = (root / link).resolve()
        assert path.parent == root.parent and path != root
        assert path not in [item["path"] for item in previous]
        previous.append({"path": path, "trials": discover(path)})
    control_path = (root.parents[1] / config["high_lr_control_result"]).resolve()
    assert control_path.parent.parent in [stage["path"] for stage in previous]
    control = read(control_path)
    frozen_control = control_path.with_name("FROZEN.json")
    assert control["frozen_sha256"] == sha(frozen_control)
    local_rows = actual_rows(trials)
    prior_trials = [trial for stage in previous for trial in stage["trials"]]
    prior_rows = actual_rows(prior_trials)
    complete = [trial for trial in trials if trial["result"]]
    failed = [trial for trial in trials if trial["failure"]]
    pending = [trial for trial in trials if not trial["result"] and not trial["failure"]]
    result_views = views(trials)
    for trial in complete:
        assert trial["result"]["frozen_sha256"] == sha(trial["path"] / "FROZEN.json")
        verify_control(trial["result"], control)
        assert trial["frozen"]["high_lr_control"]["result_sha256"] == sha(control_path)
        if trial["aggregate"]:
            verify_control(trial["aggregate"], control)
    output_figure = figure(root, result_views, control)
    text = [
        "# Совместная адаптация CNN и FiLM: третий этап разработки", "",
        "Это последовательная разработка **модели** на прежних TRAIN/VAL. "
        "Во всех вариантах остаётся прежний лосс: плотная BCE и geometry_loss с весом 1 по слабым меткам. "
        "Независимый TEST не проводится; численный сигнал на повторно используемой VAL не доказывает "
        "статистическое превосходство или новизну.", "",
        "FiLM использует исходную CNN с 10 681 параметром и уточнитель с 5 537 параметрами: всего 16 218. "
        "Логит `z_coarse + 4*tanh(residual)`. Нулевой последний слой сохраняет исходный выход при старте. "
        "Первые 100 шагов обучается уточнитель; следующие 400 — CNN и уточнитель совместно. "
        "AdamW: lr CNN = 0,0001, lr уточнителя = 0,0005, weight_decay = 0,0001. "
        "По 8 плотных и 8 слабых примеров в шаге; у plain/joint одинаковы начальные веса, параметры и батчи. "
        "Признаки и карты пересчитываются при текущих весах; карта отделена от градиента.", "",
        "## Объём поиска", "",
        f"В этом этапе начато попыток: {len(trials)}; завершено: {len(complete)}; "
        f"технических ошибок: {len(failed)}; незавершённых: {len(pending)}. "
        f"Сохранено локальных обучений: {len(local_rows)}, из них уточнителей "
        f"{sum(row['arm'] in ['plain', 'joint'] for row in local_rows)} и новых CNN-контролей "
        f"{sum(row['arm'] == 'continued_base_low_lr' for row in local_rows)}. "
        f"Обновлений в сохранённых обучениях: {sum(row['steps'] for row in local_rows):,}.", "",
        f"Предыдущие два этапа: {len(prior_trials)} сохранённых попыток, "
        f"{len(prior_rows)} обучений уточнителей. Общий контроль lr=0,0005 содержит "
        f"{len(control['rows'])} обучений и учитывается один раз. Уникальных сохранённых обучений "
        f"по трём этапам с этим контролем: {len(prior_rows) + len(local_rows) + len(control['rows'])}. "
        "Строки AGGREGATE_RESULT и копии high_lr_controls не добавляют обучений.", "",
        "## Критерии этого этапа", "",
        "A проверяется отдельно для plain/joint **против каждого** из двух CNN-контролей: "
        "новая исходная CNN с lr=0,0001 и общий контроль с lr=0,0005; обе обучаются 500 шагов. "
        "A требует прохождения обоих сравнений. B проверяет joint против plain и требует A joint. "
        "Пороги: средний прирост F1 ≥0,5 п.п.; каждый бюджет ≥−0,25 п.п.; средний ΔIoU ≥−0,2 п.п. "
        "Бюджеты 18/27 имеют равный вес. При расширении новые seed 5521/5557 оцениваются отдельно "
        "от общего среднего с 5501; нужны оба результата. Прежние A/B относительно замороженной CNN "
        "в предыдущих этапах сохраняются и не переименовываются задним числом.", "",
        f"Общий контроль: [RESULT.json]({rel(control_path, root)}), "
        f"SHA-256 `{sha(control_path)}`; [FROZEN.json]({rel(frozen_control, root)}), "
        f"SHA-256 `{sha(frozen_control)}`. Сравнения используют точные пары (бюджет, seed). "
        "Проверены одинаковые SHA исходных CNN, батчей и число шагов; совпадение копий контроля проверено.", "",
    ]
    if result_views:
        text += ["| Набор seed | CNN lr 0,0001 F1, % | CNN lr 0,0005 F1, % | Plain F1, % | Joint F1, % | A plain | A joint | B |",
                 "|---|---:|---:|---:|---:|---|---|---|"]
        for label, result in result_views:
            gates = result["gates"]
            text.append(f"| {label}: {result['seeds']} | {percent(mean(result['rows'], 'continued_base_low_lr'))} | "
                        f"{percent(mean(verify_control(result, control)))} | {percent(mean(result['rows'], 'plain'))} | "
                        f"{percent(mean(result['rows'], 'joint'))} | {passed(gates['A_architecture']['plain']['passed'])} | "
                        f"{passed(gates['A_architecture']['joint']['passed'])} | {passed(gates['B_physics']['passed'])} |")
        text.append("")
    else:
        text += ["Завершённых результатов этого этапа пока нет. Вывод о прохождении критериев не сделан.", ""]
    if output_figure:
        text += ["![Средние VAL и раздельные сравнения](figures/joint_adaptation_validation.png)", "",
                 "Точечные средние, без интервалов независимого подтверждения. "
                 "Средние всех трёх и новых двух seed перекрываются по данным.", ""]
    text += ["## Все сохранённые попытки этого этапа", ""]
    for trial in trials:
        path = trial["path"]
        text += [f"### {path.name}", "", f"[FROZEN.json]({rel(path / 'FROZEN.json', root)}).", ""]
        if not trial["result"]:
            if trial["failure"]:
                item = trial["failure"]
                text += [f"Техническая ошибка `{item.get('error_type', 'см. файл')}`: "
                         f"{str(item.get('error', '')).replace(chr(10), ' ')}. "
                         f"[FAILURE.json]({rel(path / 'FAILURE.json', root)}). "
                         "Она не считается отрицательным научным результатом.", ""]
            else:
                text += ["Попытка начата; итоговый RESULT.json пока отсутствует.", ""]
            text += [f"Завершённых сохранённых обучений до текущего состояния: {len(trial['partial'])}.", ""]
            continue
        result = trial["result"]
        text += [f"`{result['status']}`; {result['phase']}; seed {result['seeds']}; "
                 f"{result['updates']:,} обновлений; {result['seconds']:.2f} с. "
                 f"[RESULT.json]({rel(path / 'RESULT.json', root)}); SHA-256 `{sha(path / 'RESULT.json')}`.", "",
                 "| Бюджет | Seed | Вариант | Порог VAL | Macro-F1, % | IoU, % | Шагов CNN / уточнителя | Zero ΔF1, п.п. | Roll ΔF1, п.п. |",
                 "|---:|---:|---|---:|---:|---:|---|---:|---:|"]
        for row in result["rows"]:
            selected = row["selected"]
            ablations = row.get("ablations", {})
            zero = pp(ablations["zero"]["mask_F1"] - selected["mask_F1"]) if "zero" in ablations else "—"
            roll = pp(ablations["roll"]["mask_F1"] - selected["mask_F1"]) if "roll" in ablations else "—"
            text.append(f"| {row['budget']} | {row['seed']} | {row['arm']} | {selected['threshold']:.2f} | "
                        f"{percent(selected['mask_F1'])} | {percent(selected['IoU'])} | "
                        f"{row['CNN_updates']} / {row['refiner_updates']} | {zero} | {roll} |")
        text.append("")
        gate_results = [("только эта попытка", result)]
        if trial["aggregate"]:
            gate_results.append(("все три seed, включая прежний скрининг", trial["aggregate"]))
            text += [f"[AGGREGATE_RESULT.json]({rel(path / 'AGGREGATE_RESULT.json', root)}) "
                     "объединяет исходный и новые seed; он не создаёт дополнительной попытки или обучений.", ""]
        for label, item in gate_results:
            text += [f"**{label}: {item['seeds']}**", "",
                     "| Сравнение | ΔF1, п.п. | ΔIoU, п.п. | ΔF1 бюджеты 18 / 27, п.п. | Критерий |",
                     "|---|---:|---:|---|---|"]
            for arm in ["plain", "joint"]:
                for key, label in [("vs_low_lr", "CNN lr=0,0001"), ("vs_high_lr", "CNN lr=0,0005")]:
                    gate = item["gates"]["A_architecture"][arm][key]
                    text.append(f"| A {arm} − {label} | {pp(gate['delta'])} | {pp(gate['IoU_delta'])} | "
                                f"{' / '.join(pp(v) for v in gate['budget_deltas'])} | {passed(gate['passed'])} |")
            gate = item["gates"]["B_physics"]
            text.append(f"| B joint − plain, с требованием A joint | {pp(gate['delta'])} | {pp(gate['IoU_delta'])} | "
                        f"{' / '.join(pp(v) for v in gate['budget_deltas'])} | {passed(gate['passed'])} |")
            text.append("")
        if trial["aggregate"]:
            aggregate = trial["aggregate"]
            robust = {arm: aggregate["gates"]["A_architecture"][arm]["passed"] and
                      result["gates"]["A_architecture"][arm]["passed"] for arm in ["plain", "joint"]}
            robust_b = aggregate["gates"]["B_physics"]["passed"] and result["gates"]["B_physics"]["passed"]
            text += [f"Одновременно все три и новые два seed: A plain {passed(robust['plain'])}; "
                     f"A joint {passed(robust['joint'])}; B {passed(robust_b)}.", ""]
        audit = trial["audit"]
        text += [(f"Независимый CPU-аудит: `{audit.get('status', 'см. файл')}`; "
                  f"[AUDIT.json]({rel(path / 'AUDIT.json', root)})." if audit else
                  "Независимый аудит этой попытки на момент генерации ещё не сохранён."), ""]
    text += ["## Предыдущие попытки того же адаптивного поиска", "",
             "В первых двух этапах A использовал замороженную CNN. Это другой контроль, "
             "чем два дообученных контроля третьего этапа. Каждая прежняя отрицательная попытка сохранена.", "",
             "| Этап / попытка | Seed | Сохранённых обучений | Plain F1, % | Joint F1, % | Прежний A plain / joint / B |",
             "|---|---|---:|---:|---:|---|"]
    for stage in previous:
        for trial in stage["trials"]:
            result = trial["result"]
            path = trial["path"]
            label = f"{stage['path'].name} / {path.name}"
            if result:
                gates = result["gates"]
                text.append(f"| [{label}]({rel(path / 'RESULT.json', root)}) | {result['seeds']} | "
                            f"{len(result['rows'])} | {percent(mean(result['rows'], 'plain'))} | "
                            f"{percent(mean(result['rows'], 'joint'))} | "
                            f"{passed(gates['A_architecture']['plain']['passed'])} / "
                            f"{passed(gates['A_architecture']['joint']['passed'])} / {passed(gates['B_physics']['passed'])} |")
            else:
                text.append(f"| {label}: {'техническая ошибка' if trial['failure'] else 'незавершена'} | "
                            f"{trial['frozen'].get('seeds', 'см. файл')} | {len(trial['partial'])} | — | — | — |")
    text += ["", "## Данные и пределы вывода", "",
             "Прежние TRAIN/VAL: пул 54 плотных масок, 1152 слабых примера, 695 VAL; "
             "вложенные бюджеты 18/27, розыгрыш 5303. Слабые маски оптимизатору не передаются. "
             "Девять классов рассчитаны условной решёткой, а не являются авторской типологией дефектов. "
             "Порог каждой модели выбирается на VAL из [0,30; 0,40; 0,45; 0,50; 0,55; 0,60; 0,70] "
             "по F1, затем IoU, близости к 0,5 и меньшему порогу. Zero/roll используют выбранный actual-порог. "
             "Чувствительность к карте сама по себе не доказывает её полезность.", "",
             "Этот этап использует те же данные после предыдущих решений. Нужны сохранение всех попыток, "
             "фиксация выбранного варианта и критериев до независимой проверки и заранее закрытые новые данные, "
             "ещё не использованные в EDA, разработке или прежних тестах. Такой набор здесь не использован. "
             "Перенос на другие изделия и новизна требуют отдельных оснований.", "",
             "[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [LEDGER.json](LEDGER.json) · "
             "[model.py](model.py) · [runner.py](runner.py) · [audit.py](audit.py).", ""]
    (root / "REPORT.md").write_text("\n".join(text))
    (root / "README.md").write_text(
        "# Совместная адаптация CNN и FiLM\n\n"
        "Третий последовательный этап разработки модели с прежним лоссом. "
        "100 шагов обучается уточнитель, затем 400 — CNN и уточнитель совместно. "
        "Plain/joint сравниваются с двумя CNN, дообученными 500 шагов с lr 0,0001 и 0,0005.\n\n"
        f"Сохранено попыток: {len(trials)}; завершено {len(complete)}; технических ошибок {len(failed)}; "
        f"локальных обучений {len(local_rows)}. Все предыдущие попытки перечислены в отчёте. "
        "Общий контроль из первого этапа повторно не считается.\n\n"
        "[Отчёт](REPORT.md) · [Протокол](PROTOCOL.md) · [Конфигурация](CONFIG.json) · [Журнал](LEDGER.json).\n\n"
        "Все результаты — повторно используемые TRAIN/VAL. Нового TEST нет; для независимого подтверждения "
        "нужны заранее закрытые свежие данные.\n\n"
        "Пересоздать только отчёт и рисунок, без моделей или обучения:\n\n"
        "```bash\n.venv/bin/python -B gipoteza/sem_arch_joint_adaptation_20261102/report.py\n```\n")
    inputs = [root / name for name in ["PROTOCOL.md", "CONFIG.json", "LEDGER.json"] if (root / name).exists()]
    inputs += [control_path, frozen_control, control_path.with_name("CONFIG_SNAPSHOT.json")]
    for stage in previous + [{"path": root, "trials": trials}]:
        inputs += [stage["path"] / name for name in ["PROTOCOL.md", "CONFIG.json", "QUEUE.json", "LEDGER.json", "STAGE_LINK.json"]
                   if (stage["path"] / name).exists()]
        for trial in stage["trials"]:
            inputs += [trial["path"] / name for name in ["FROZEN.json", "RESULT.json", "AGGREGATE_RESULT.json", "FAILURE.json",
                       "PARTIAL_RESULTS.json", "AUDIT.json", "HIGH_LR_CONTROL_SNAPSHOT.json"] if (trial["path"] / name).exists()]
    outputs = [root / "REPORT.md", root / "README.md"] + ([output_figure] if output_figure else [])
    manifest = dict(status="REPORT_GENERATED", scope="ADAPTIVE_REUSED_TRAIN_VAL_ONLY",
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), report_script_sha256=sha(Path(__file__)),
        trial_count=len(trials), completed_trials=len(complete), failed_trials=len(failed), pending_trials=len(pending),
        actual_completed_fits=len(local_rows), local_refiner_fits=sum(row["arm"] in ["plain", "joint"] for row in local_rows),
        local_low_lr_control_fits=sum(row["arm"] == "continued_base_low_lr" for row in local_rows),
        referenced_high_lr_control_fits=len(control["rows"]), high_lr_control_counted_as_local=False,
        prior_stage_attempts=len(prior_trials), linked_stage_total_attempts=len(prior_trials) + len(trials),
        linked_stage_unique_completed_fits=len(prior_rows) + len(local_rows) + len(control["rows"]),
        shared_control=dict(result_path=rel(control_path, root), result_sha256=sha(control_path),
            frozen_sha256=sha(frozen_control), matching_unit=["budget", "seed"],
            matched_fields=["backbone_before_sha256", "batch_sequence_sha256", "steps"]),
        scientific_inputs_sha256={rel(path, root): sha(path) for path in inputs},
        outputs_sha256={rel(path, root): sha(path) for path in outputs})
    (root / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
