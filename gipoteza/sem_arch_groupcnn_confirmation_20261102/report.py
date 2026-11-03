"""Report locked confirmation outputs; no model import, fitting or prediction."""
from pathlib import Path
import datetime
import importlib.util
import json

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DEV = ROOT.parent / "sem_arch_groupcnn_20261102"
HELPER = ROOT.parent / "sem_arch_joint_adaptation_20261102" / "report.py"
spec = importlib.util.spec_from_file_location("group_confirmation_report_helpers", HELPER)
h = importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
read, sha, rel, pct, passed = h.read, h.sha, h.rel, h.percent, h.passed
plt, np = h.plt, h.np
ROUTES = ["group6_single", "plain12_C4", "plain24_C4"]
GOALS = ["A_matched_quality", "B_full_efficiency"]
MODES = ["primary_fine", "secondary_seven"]
COLORS = ["#b25a43", "#3c78a8", "#4d8d63"]
LABELS = ["Group6 single", "Plain12 C4", "Plain24 C4"]
PARAMETERS = [2671, 2749, 10681]
MAC = [1527552, 1562112, 6110208]


def pp(value):
    return f"{100 * value:+.6f}"


def audit_pass(item):
    return bool(item and item.get("status", "").startswith("PASS") and all(item.get("checks", {}).values()))


def mean(result, route, metric="mask_F1", budget=None, mode="primary_fine"):
    values = [row["metrics"][mode][metric] for row in result["rows"]
              if row["route"] == route and (budget is None or row["budget"] == budget)]
    return float(np.mean(values))


def directional_errors():
    """Descriptive primary-label errors; never change thresholds or gates."""
    with np.load(ROOT / "PREDICTIONS.npz", allow_pickle=False) as saved:
        modes = saved["modes"].tolist();routes = saved["routes"].tolist()
        truth = saved["truth"].astype(np.int64)
        predictions = saved["mask_predictions"][modes.index("primary_fine")].astype(np.int64)
    assert routes == ROUTES and predictions.shape == (3, 2, 3, len(truth))
    target = np.stack((truth // 3, truth % 3), axis=-1)
    predicted = np.stack((predictions // 3, predictions % 3), axis=-1)
    events = dict(false_connection=(target == 0) & (predicted > 0),
                  missed_connection=(target > 0) & (predicted == 0),
                  grade_confusion=(target > 0) & (predicted > 0) & (target != predicted))
    assert np.array_equal(sum(events.values()), (predicted != target).astype(int))
    summary = {}
    for name, event in events.items():
        counts = event.sum(axis=(-2, -1));by_direction = event.sum(axis=-2)
        group, control = [routes.index(route) for route in ["group6_single", "plain12_C4"]]
        summary[name] = dict(group_mean_count=float(counts[group].mean()), narrow_mean_count=float(counts[control].mean()),
            delta_group_minus_narrow=float((counts[group] - counts[control]).mean()),
            LR_delta=float((by_direction[group, :, :, 0] - by_direction[control, :, :, 0]).mean()),
            TB_delta=float((by_direction[group, :, :, 1] - by_direction[control, :, :, 1]).mean()),
            per_state_counts={route: counts[routes.index(route)].tolist() for route in ["group6_single", "plain12_C4"]})
    return dict(scope="POSTHOC_DESCRIPTIVE_PRIMARY_GEOMETRIC_DIRECTION_ERRORS_NOT_A_GATE_OR_PRODUCTION_DEFECT_RATE",
                examples=len(truth), directions_per_state=2 * len(truth), states_averaged=6, categories=summary)


def interval_figure(result, config):
    path = ROOT / "figures" / "confirmation_intervals.png";path.parent.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for axis, goal, title in zip(axes, GOALS, ["A: group − narrow C4", "B: group − full C4"]):
        for x, mode, color, marker in [(0, "primary_fine", "#3c78a8", "o"), (1, "secondary_seven", "#b25a43", "D")]:
            row = result["analyses"][mode]["goals"][goal];low, high = np.array(row["CI97_5"]) * 100
            axis.vlines(x, low, high, color=color, lw=1.8)
            axis.scatter([x, x], [low, high], marker="_", color=color, s=90)
            axis.scatter(x, 100 * row["delta"], marker=marker, color=color, s=55)
            axis.annotate(f"{100 * row['delta']:+.2f} pp", (x, 100 * row["delta"]),
                          xytext=(8, 5), textcoords="offset points", fontsize=9)
        limits = config["goals"][goal]
        axis.axhline(0, color="black", lw=.8)
        axis.axhline(100 * limits["minimum_mean_F1_delta"], color="#777777", ls="--", lw=.9,
                     label=f"Mean floor {100 * limits['minimum_mean_F1_delta']:+.2f} pp")
        axis.axhline(100 * limits["CI_lower_strictly_above"], color="#999999", ls=":", lw=1,
                     label=f"CI lower must exceed {100 * limits['CI_lower_strictly_above']:+.2f} pp")
        axis.set_xticks([0, 1], ["Primary\nfrozen fine-VAL threshold", "Secondary only\nfrozen 7-grid threshold"])
        axis.set_xlim(-.45, 1.55);axis.set_ylabel("Macro-F1 difference, percentage points")
        axis.set_title(title + f"; primary {'confirmed' if result['confirmed_by_goal'][goal] else 'not confirmed'}", fontsize=10)
        axis.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.18))
    fig.suptitle("16 new fields of the same IC: paired seed/frame 97.5% intervals", fontsize=11)
    fig.tight_layout();fig.savefig(path, dpi=180, bbox_inches="tight");plt.close(fig)
    return path


def quality_figure(result):
    path = ROOT / "figures" / "confirmation_quality_compute.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8));x = np.arange(2)
    for offset, route, label, color in zip([-.24, 0, .24], ROUTES, LABELS, COLORS):
        for axis, metric in zip(axes[0], ["mask_F1", "IoU"]):
            values = [100 * mean(result, route, metric, budget) for budget in [18, 27]]
            bars = axis.bar(x + offset, values, .22, label=label, color=color)
            axis.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    for axis, ylabel, title in zip(axes[0], ["Nine-class Macro-F1, %", "Mask IoU, %"],
                                  ["Mask-derived classification", "Segmentation; fixed primary thresholds"]):
        axis.set_xticks(x, ["budget 18", "budget 27"]);axis.set_ylim(0, 100)
        axis.set_ylabel(ylabel);axis.set_title(title, fontsize=10);axis.legend(fontsize=8, loc="lower left")
    for axis, values, ylabel, title, fmt in [
        (axes[1, 0], PARAMETERS, "Independent learned parameters", "Materialized group storage: 10,681 tied coefficients", "%.0f"),
        (axes[1, 1], np.array(MAC) / 1e6, "MAC per image, millions", "Analytical CNN compute; 1 versus 4 views", "%.3f")]:
        bars = axis.bar(np.arange(3), values, color=COLORS, width=.6)
        axis.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
        axis.set_xticks(np.arange(3), LABELS);axis.set_ylim(0, max(values) * 1.18)
        axis.set_ylabel(ylabel);axis.set_title(title, fontsize=10)
    fig.suptitle("Frozen models on 4,096 new patches; means over three seeds", fontsize=11)
    fig.tight_layout();fig.savefig(path, dpi=180, bbox_inches="tight");plt.close(fig)
    return path


def verify(result, selection, data_audit, audit):
    for key, name in [("selection_sha256", "SELECTION.json"), ("evaluation_frozen_sha256", "EVALUATION_FROZEN.json"),
                      ("predictions_sha256", "PREDICTIONS.npz"), ("bootstrap_sha256", "BOOTSTRAP.npz")]:
        assert result[key] == sha(ROOT / name)
    assert result["fit_count"] == result["optimizer_updates"] == 0
    assert result["source_model_count"] == result["evaluated_routes"] == len(result["rows"]) == 18
    assert result["test_examples"] == 4096 and len(result["test_frames"]) == 16
    assert result["bootstrap_repeats"] == 10000 and result["bootstrap_seed"] == 202611027
    assert result["CI_quantiles"] == [.0125, .9875]
    assert selection["fit_count"] == selection["optimizer_updates"] == 0
    keys = {(r["route"], r["budget"], r["seed"]) for r in result["rows"]}
    assert keys == {(r, b, s) for r in ROUTES for b in [18, 27] for s in [5701, 5721, 5757]}
    for goal in GOALS:
        primary = result["analyses"]["primary_fine"]["goals"][goal]
        assert primary["passed"] == all(primary["checks"].values())
        assert primary["development_eligible"] == selection["eligible_by_goal"][goal]
        assert primary["confirmed"] == (primary["passed"] and primary["development_eligible"])
        assert result["confirmed_by_goal"][goal] == primary["confirmed"]
        assert not result["analyses"]["secondary_seven"]["goals"][goal]["confirmed"]
    assert (result["status"] == "PASS_CONFIRMATION") == any(result["confirmed_by_goal"].values())
    frozen = read(ROOT / "EVALUATION_FROZEN.json")
    for name, expected in frozen["input_sha256"].items():
        assert sha(REPO / name) == expected
    for name, expected in selection["code_sha256"].items():
        assert sha(REPO / name) == expected
    for row in selection["rows"]:
        for prefix in ["free_model", "prediction_model"]:
            assert sha(REPO / row[prefix + "_file"]) == row[prefix + "_sha256"]
    for name, expected in (data_audit or {}).get("input_sha256", {}).items():
        assert sha(REPO / name) == expected
    if audit:
        assert audit["result_sha256"] == sha(ROOT / "RESULT.json")


def build():
    config = read(ROOT / "CONFIG.json");selection = read(ROOT / "SELECTION.json")
    plan = read(ROOT / "SOURCE_PLAN.json");index = read(ROOT / "EVAL_INDEX.json");result = read(ROOT / "RESULT.json")
    data_manifest = read(ROOT / "DATA_MANIFEST.json")
    audit = read(ROOT / "AUDIT.json");data_audit = read(ROOT / "DATA_AUDIT.json")
    dev_manifest = read(DEV / "REPORT_MANIFEST.json");historical_fits = dev_manifest["linked_stage_unique_completed_fits"]
    images = [];diagnostic = None
    if result:
        verify(result, selection, data_audit, audit)
        images = [interval_figure(result, config), quality_figure(result)]
        diagnostic = directional_errors()
    final = bool(result and audit_pass(audit) and audit_pass(data_audit))
    status = result["status"] if result else "РЕЗУЛЬТАТ ЕЩЁ НЕ СОХРАНЁН"
    text = ["# Закрытое подтверждение C4-группового encoder", "",
        "Зафиксированная Group6 CNN одним проходом предсказывает маску металлизации M2. "
        "Обычные Plain12 и Plain24 используют четырёхвидовое C4-усреднение вероятностей. "
        "Класс 0…8 вычисляется из бинарной маски прежним геометрическим правилом по направлениям LR/TB. "
        "Это девять модельных геометрических состояний, а не экспертные типы дефектов или измеренное электрическое состояние.", "",
        "**`mask_F1` в JSON — Macro-F1 девяти классов из маски, не пиксельный F1.** "
        "Сегментация измеряется отдельно средним IoU. Для всех моделей сохранены прежние BCE и geometry_loss; "
        "на новых данных нет обучения, поиска порога или выбора модели.", "",
        f"**Первичный исход: {status}.** " + ("Оба независимых итоговых аудита сохранены и прошли." if final else
        "Отчёт предварительный: финальный статус требует независимых аудитов данных и результата."), ""]
    if result:
        text += ["Подтверждение отдельных целей: " + "; ".join(f"**{goal}: {'подтверждена' if result['confirmed_by_goal'][goal] else 'не подтверждена'}**" for goal in GOALS) +
                 ". Общий PASS означает хотя бы одну подтверждённую цель; он не заменяет исход каждой цели.", ""]
        if result["confirmed_by_goal"]["A_matched_quality"]:
            text += ["Основной подтверждённый вывод: этот encoder улучшает Macro-F1 масочно-геометрических классов "
                     "относительно узкой Plain12 C4 при заранее согласованных свободных параметрах и MAC "
                     "на новых полях той же ИС. Вывод относится к этому архитектурному сравнению и протоколу.", ""]
        if not result["confirmed_by_goal"]["B_full_efficiency"]:
            text += ["Отдельная цель B к полной Plain24 C4 не подтверждена. Даже положительная точечная разность "
                     "не заменяет требуемую нижнюю границу интервала и остальные ограничения B.", ""]
    text += [f"Новых обучений **0**; прежних различных fit **{historical_fits}**. "
        "Проверяются 18 фиксированных состояний: три модели × два бюджета 18/27 × три seed 5701/5721/5757. "
        "Шесть материализованных group-checkpoint повторно не считаются обучениями. "
        "Две заранее фиксированные калибровки используют одни предсказанные вероятности.", "",
        "[Архитектурная гипотеза](../sem_arch_groupcnn_20261102/HYPOTHESIS.md) и "
        "[результаты разработки с полной историей попыток](../sem_arch_groupcnn_20261102/REPORT.md) "
        "сохраняют исходы на прежней VAL. Этот отчёт не переписывает их новым исходом.", "",
        "## Две отдельные цели подтверждения", "",
        "**A_matched_quality:** Group6 против Plain12 C4 при близких свободных параметрах и MAC (различия ≤3%). "
        "Средний ΔF1 ≥+0,5 п.п.; нижняя граница 97,5%-интервала **строго >0**; "
        "каждый бюджет ΔF1 ≥−0,25 п.п.; средний ΔIoU ≥−0,2 п.п.", "",
        "Порог +0,5 п.п. — практический порог **наблюдаемой точечной средней оценки**. "
        "Условие CI lower >0 поддерживает положительный эффект в заданной схеме пересэмплирования; "
        "оно не доказывает, что минимальный популяционный прирост составляет ≥0,5 п.п.", "",
        "**B_full_efficiency:** Group6 против Plain24 C4. Средний ΔF1 ≥−0,5 п.п.; "
        "нижняя граница 97,5%-интервала **строго >−0,5 п.п.**; каждый бюджет ΔF1 ≥−1 п.п.; "
        "средний ΔIoU ≥−0,2 п.п.; отношение MAC ≤0,30. Это отдельная цель допустимой потери качества и экономии вычислений. "
        "Успех B не переименовывается в превосходство A.", "",
        "Каждая цель дополнительно требует своего прежнего допуска на **всех трёх и новых двух seed одновременно**. "
        "Основные пороги выбраны до новых изображений на прежней VAL-сетке 81 значения. "
        "Семиточечные пороги — только вторичная чувствительность, без возможности заменить первичный исход.", "",
        "Интервалы: 10 000 парных crossed-bootstrap повторов, seed 202611027. Сначала выбираются три seed с возвращением, "
        "затем 16 кадров с возвращением; одни индексы общие для моделей, бюджетов, калибровок и сравнений. "
        "Для каждого выбранного seed/бюджета confusion суммируется по выбранным кадрам, затем вычисляется Macro-F1; "
        "потом усредняются F1 по seed и бюджетам. Confusion разных seed не объединяются перед F1. "
        "Центральные квантили [0,0125; 0,9875] дают поправку Бонферрони для двух заранее заданных сравнений. "
        "Пиксели и фрагменты не считаются независимыми bootstrap-единицами.", ""]
    if result:
        for mode, label in [("primary_fine", "Первичная оценка"), ("secondary_seven", "Вторичная чувствительность: 7 порогов")]:
            text += [f"## {label}", "",
                "| Цель | Group F1, % | Контроль F1, % | ΔF1, п.п. | 97,5% CI ΔF1, п.п. | ΔF1: 18 / 27, п.п. | ΔIoU, п.п. | Все численные условия | Подтверждение |",
                "|---|---:|---:|---:|---|---|---:|---|---|"]
            for goal in GOALS:
                row = result["analyses"][mode]["goals"][goal]
                text.append(f"| {goal} | {pct(row['group_F1'])} | {pct(row['reference_F1'])} | {pp(row['delta'])} | "
                    f"[{pp(row['CI97_5'][0])}; {pp(row['CI97_5'][1])}] | {' / '.join(pp(v) for v in row['budget_deltas'])} | "
                    f"{pp(row['IoU_delta'])} | {passed(row['passed'])} | " +
                    (("да" if row["confirmed"] else "нет") if mode == "primary_fine" else "не применяется") + " |")
            text += ["", "| Цель | Проверка | Исход |", "|---|---|---|"]
            for goal in GOALS:
                row = result["analyses"][mode]["goals"][goal]
                for name, value in row["checks"].items():
                    text.append(f"| {goal} | {name} | {passed(value)} |")
                text.append(f"| {goal} | development_eligible | {passed(row['development_eligible'])} |")
            text += [""]
        text += ["![Отдельные A/B и интервалы](figures/confirmation_intervals.png)", "",
            "Точки — средние приросты, линии — процентильные интервалы. У A порог среднего +0,5 п.п. "
            "отличается от строгого порога нижней границы 0. У B оба численных уровня −0,5 п.п., "
            "но нижняя граница должна быть строго выше. Вторичные интервалы не меняют подтверждение.", "",
            "## Абсолютное качество и вычисления", "",
            "| Маршрут | Macro-F1, % | IoU, % | Свободных параметров | Коэффициентов хранения | MAC / изображение | Видов |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for i, route in enumerate(ROUTES):
            text.append(f"| {route} | {pct(mean(result, route))} | {pct(mean(result, route, 'IoU'))} | {PARAMETERS[i]} | "
                        f"{10681 if i in [0, 2] else 2749} | {MAC[i]} | {1 if i == 0 else 4} |")
        text += ["", "![F1, IoU и вычислительные бюджеты](figures/confirmation_quality_compute.png)", "",
            "Качество показано при первичных фиксированных порогах. Значения по бюджетам усредняются по трём seed; "
            "это не ансамбль трёх моделей. У group 10 681 материализованный коэффициент связан с 2 671 свободным "
            "параметром. MAC-отношение к полному C4 равно 0,25; оно не равно отношению wall-clock времени.", "",
            "Рандомизированный CNN-only latency benchmark выполнен при разработке и сохранён "
            "[отдельно](../sem_arch_groupcnn_20261102/REPORT.md). Он включает повороты и pool, но исключает "
            "материализацию, данные, host-transfer и CPU geometric readout. End-to-end время здесь не заявляется. "
            "Времена последовательной проверки новых данных не заменяют тот парный benchmark.", "",
            "## Все зафиксированные модели", "",
            "| Маршрут | Бюджет | Seed | Порог primary / secondary | Primary F1, % | Primary IoU, % | Secondary F1, % |",
            "|---|---:|---:|---|---:|---:|---:|"]
        for row in result["rows"]:
            p, s = [row["metrics"][mode] for mode in MODES]
            text.append(f"| {row['route']} | {row['budget']} | {row['seed']} | {p['threshold']:.2f} / {s['threshold']:.2f} | "
                        f"{pct(p['mask_F1'])} | {pct(p['IoU'])} | {pct(s['mask_F1'])} |")
        text += ["", f"Полное время оценки и статистики {result['seconds']:.2f} с. "
            "[PREDICTIONS.npz](PREDICTIONS.npz) сохраняет вероятности, классы, упакованные маски и IoU; "
            "[BOOTSTRAP.npz](BOOTSTRAP.npz) — frame-confusion, общие seed/frame-индексы и все bootstrap-разности. "
            "[RESULT.json](RESULT.json) хранит полный численный исход, включая не пройденные условия.", ""]
        text += ["## Описательная проверка ошибок направлений", "",
            "Этот разбор добавлен после оценки и **не является заранее заданным критерием, доказательством механизма "
            "или производственной частотой дефектов**. Используются только сохранённые первичные классы: "
            "`LR=y//3`, `TB=y%3`. Состояние 0 — отсутствие граничного соединения по прежнему геометрическому правилу; "
            "1 и 2 — две ненулевые градации ответа. Это состояние вычисленной маски, не электрическое измерение.", "",
            f"В таблице — среднее число событий по шести состояниям seed×бюджет, из {diagnostic['directions_per_state']} "
            "решений LR/TB в каждом. Один фрагмент может дать ошибку обоих направлений. Отрицательная разность означает "
            "меньше таких ошибок у Group6. Это разложение ошибок направлений, а не математическое разложение Macro-F1 девяти классов.", "",
            "| Ошибка | GT → prediction | Group6 | Plain12 C4 | Δ Group−Plain12 | Δ LR | Δ TB |",
            "|---|---|---:|---:|---:|---:|---:|"]
        for name, label, transition in [("false_connection", "Ложное граничное соединение", "0 → 1/2"),
                                        ("missed_connection", "Пропущенное соединение", "1/2 → 0"),
                                        ("grade_confusion", "Путаница ненулевых градаций", "1 ↔ 2")]:
            item = diagnostic["categories"][name]
            text.append(f"| {label} | {transition} | {item['group_mean_count']:.3f} | {item['narrow_mean_count']:.3f} | "
                        f"{item['delta_group_minus_narrow']:+.3f} | {item['LR_delta']:+.3f} | {item['TB_delta']:+.3f} |")
        text += ["", "Числа описывают, какие типы несовпадений изменились в этой фиксированной выборке. "
            "Они не устанавливают причинный механизм улучшения и не использовались для выбора модели или порога.", ""]
    text += ["## Фиксированная выборка и происхождение", "",
        "16 новых полей съёмки, по 256 фрагментов, всего 4 096. До скачивания исключены 94 прежних кадра; "
        "точные D4-повторы исходных интенсивностей и масок удалены по шести прежним манифестам и внутри нового набора. "
        "Среди оставшихся фрагментов ранжирование использует только SHA интенсивностей; маска и класс не входят в ranking. "
        "Нужно ≥8 фрагментов и ≥3 кадров каждого класса. Недостаточное покрытие означает недостаточность фиксированного "
        "набора, без замены кадров или пересэмплирования.", ""]
    if data_manifest:
        text += [f"Полный новый [TEST.npz](TEST.npz) содержит **{data_manifest['examples']}** уникальных фрагментов "
                 "только новых кадров, без добавления TRAIN/VAL. Для модельной проверки используется заранее "
                 "заданный EVAL, а не весь TEST. Остаток тех же кадров не является ещё одним независимым подтверждением.", ""]
    if plan:
        text += [f"Фиксированные кадры: **{plan['test_frames']}**. "
                 f"[SOURCE_PLAN.json](SOURCE_PLAN.json), SHA-256 `{sha(ROOT / 'SOURCE_PLAN.json')}`.", ""]
    if index:
        text += [f"Data gate: `{index['status']}`; фрагментов {index['examples']}. "
            "[EVAL_INDEX.json](EVAL_INDEX.json) · [DATA_MANIFEST.json](DATA_MANIFEST.json).", "",
            "| Класс | Фрагментов | Различных кадров |", "|---:|---:|---:|"]
        for cls, number in enumerate(index["class_counts"]):
            text.append(f"| {cls} | {number} | {len(index['class_frames'][str(cls)])} |")
        text.append("")
    if selection:
        text += [f"До новых изображений [SELECTION.json](SELECTION.json), SHA-256 `{sha(ROOT / 'SELECTION.json')}`, "
            "связывает все 18 весов, оба порога каждого маршрута, код и "
            "[MODEL_LOCK.json](MODEL_LOCK.json). Перед первым новым предсказанием отдельно сохраняется "
            "[EVALUATION_FROZEN.json](EVALUATION_FROZEN.json). Аудит данных связывает SHA проверенного EVAL, "
            "индекса и манифеста с тем набором, который получает оценщик.", ""]
    text += ["| Независимая проверка | Статус | Проверок |", "|---|---|---|"]
    audit_names = ["PRE_LOCK_AUDIT.json", "PRE_DOWNLOAD_AUDIT.json", "DATA_AUDIT.json", "AUDIT.json"]
    for name in audit_names:
        item = read(ROOT / name)
        if item:
            text.append(f"| [{name}]({name}) | {item['status']} | {item.get('checks_passed', '?')} / {item.get('checks_total', '?')} |")
        else:
            text.append(f"| {name} | ещё не сохранён | — |")
    text += ["", "Технический PASS отделён от научных целей A/B. До скачивания выполнена настоящая проверка "
        "18 маршрутов на 17 прежних VAL-примерах с батчем 16, включая неполный батч, и независимая синтетическая "
        "проверка bootstrap. Это техническая проверка прежних данных, не новое подтверждение качества. "
        "[OLD_VAL_PREFLIGHT.json](OLD_VAL_PREFLIGHT.json). Предыдущие версии аудитов и source-снимки сохраняются.", ""]
    for path in sorted(ROOT.glob("*AUDIT_PREVIOUS_*.json")):
        text += [f"Сохранённый предыдущий аудит: [{path.name}]({path.name}).", ""]
    failures = sorted((DEV / "trials").glob("*/FAILURE.json"))
    if failures:
        text += ["Сохранённые технические остановки разработки: " + "; ".join(f"[{p.parent.name}]({rel(p, ROOT)})" for p in failures) +
            ". Первое обучение унаследовано по SHA без повторного fit; цепочка учёта остаётся в отчёте разработки.", ""]
    for name in ["EVALUATION_FAILURE.json", "FAIL_EVAL_DATA_GATE.json"]:
        if (ROOT / name).exists():
            text += [f"Сохранённая остановка текущей проверки: [{name}]({name}).", ""]
    text += ["## Границы вывода", "",
        "Это новые поля **той же физической ИС**, а не независимые изделия. Фрагменты одного кадра зависимы; "
        "D4-дедупликация не устраняет все похожие структуры и пространственные связи. Разброс dense-подвыборок "
        "не оценивался: использован один розыгрыш 5303. Отбор архитектуры на прежней VAL был адаптивным. "
        "Новая проверка относится к замороженному encoder и порогам в этом протоколе.", "",
        "Известная C4 group-CNN не объявляется новым общим методом. Перенос между изделиями и публикационная "
        "готовность для Q1 не установлены. Прежний отдельный FAIL_PILOT равномерного C4 сохранён "
        "[в его отчёте](../sem_arch_orbit_confirmation_20261102/REPORT.md); он не заменяется исходом этой архитектуры.", "",
        "[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [SELECTION.json](SELECTION.json) · "
        "[evaluate.py](evaluate.py) · [confirmation_stats.py](confirmation_stats.py) · [audit.py](audit.py).", ""]
    (ROOT / "REPORT.md").write_text("\n".join(text))
    goals_text = "; ".join(f"{g}: {'подтверждена' if result['confirmed_by_goal'][g] else 'не подтверждена'}" for g in GOALS) if result else "исходов пока нет"
    (ROOT / "README.md").write_text("# Закрытое подтверждение C4-групповой CNN\n\n" +
        f"Первичный исход: **{status}**. {goals_text}. " +
        ("Независимые аудиты данных и результата PASS.\n\n" if final else "Отчёт предварительный; финальные независимые аудиты ещё требуются.\n\n") +
        "16 новых полей одной ИС, 4 096 фрагментов, 18 фиксированных моделей. "
        f"Новых fit 0; прежних уникальных fit {historical_fits}. `mask_F1` означает Macro-F1 девяти масочно-геометрических классов; IoU — сегментация.\n\n"
        "A проверяет качество к узкой C4 при близких параметрах/MAC; B отдельно проверяет допустимую потерю к полной C4 "
        "при MAC 0,25. 10 000 парных crossed-bootstrap повторов по seed/кадрам, интервалы 97,5%. "
        "Вторичные пороги не заменяют первичный исход. End-to-end latency не измерена.\n\n"
        "[Отчёт](REPORT.md) · [Протокол](PROTOCOL.md) · [Модели и пороги](SELECTION.json) · "
        "[Разработка](../sem_arch_groupcnn_20261102/REPORT.md) · [Гипотеза](../sem_arch_groupcnn_20261102/HYPOTHESIS.md).\n")
    names = ["CONFIG.json", "PROTOCOL.md", "MODEL_LOCK.json", "SOURCE_PLAN.json", "SELECTION.json", "EVALUATION_FROZEN.json",
        "DATA_MANIFEST.json", "EVAL_INDEX.json", "TEST.npz", "EVAL.npz", "DATA_AUDIT.json", "AUDIT.json", "RESULT.json", "BOOTSTRAP.npz", "PREDICTIONS.npz",
        "OLD_VAL_PREFLIGHT.json", "OLD_VAL_PREFLIGHT.npz", "PRE_LOCK_AUDIT.json", "PRE_DOWNLOAD_AUDIT.json", "PREPARE_CHECK.json",
        "DOWNLOAD.json", "EVALUATION_FAILURE.json", "FAIL_EVAL_DATA_GATE.json", "confirmation_common.py", "confirmation_stats.py",
        "selection.py", "prepare.py", "evaluate.py", "audit.py"]
    inputs = [ROOT / name for name in names if (ROOT / name).exists()] + [HELPER, DEV / "REPORT.md", DEV / "HYPOTHESIS.md", DEV / "REPORT_MANIFEST.json"] + failures
    if selection:
        inputs += [REPO / name for name in selection["code_sha256"]]
        inputs += [REPO / row[prefix + "_file"] for row in selection["rows"] for prefix in ["free_model", "prediction_model"]]
        inputs += [REPO / source[prefix + "_file"] for source in selection["development"]["source_trials"] for prefix in ["result", "audit"]]
    if (ROOT / "EVALUATION_FROZEN.json").exists():
        inputs += [REPO / name for name in read(ROOT / "EVALUATION_FROZEN.json")["input_sha256"]]
    inputs += sorted((ROOT / "audit_sources").glob("*.py")) + sorted(ROOT.glob("*AUDIT_PREVIOUS_*.json"))
    outputs = [ROOT / "REPORT.md", ROOT / "README.md"] + images
    manifest = dict(status="FINAL_AUDITED_REPORT" if final else "PRELIMINARY_REPORT_WAITING_FOR_AUDITS",
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), scope="LOCKED_NEW_FIELDS_SAME_IC_GROUP_CNN_CONFIRMATION",
        report_script_sha256=sha(Path(__file__)), primary_status=status, confirmed_by_goal=result["confirmed_by_goal"] if result else None,
        final_independent_audits_passed=final, new_training_fits=0, historical_unique_training_fits=historical_fits,
        source_model_count=18, evaluated_routes=18 if result else 0, calibrated_metric_sets=36 if result else 0,
        deployment_artifacts_are_new_fits=False, secondary_can_replace_primary=False,
        confidence_interval_quantiles=config["CI_quantiles"], bootstrap_repeats=config["bootstrap_repeats"],
        directional_error_diagnostic=diagnostic,
        scientific_inputs_sha256={rel(p, ROOT): sha(p) for p in sorted(set(inputs))}, outputs_sha256={rel(p, ROOT): sha(p) for p in outputs})
    (ROOT / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ["scientific_inputs_sha256", "outputs_sha256", "directional_error_diagnostic"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
