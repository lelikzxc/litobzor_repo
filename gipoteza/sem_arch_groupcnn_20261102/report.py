"""Report saved group-CNN evidence; never import a model or evaluate data."""
from pathlib import Path
import datetime
import importlib.util
import json
import re

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "sem_arch_anchored_readout_20261102"
HELPER = ROOT.parent / "sem_arch_joint_adaptation_20261102" / "report.py"
spec = importlib.util.spec_from_file_location("groupcnn_report_helpers", HELPER)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)
read, sha, rel, discover, views = h.read, h.sha, h.rel, h.discover, h.views
pct, pp, passed, plt, np = h.percent, h.pp, h.passed, h.plt, h.np
ARMS = ["group6", "plain12_C4pooled", "plain24_C4pooled"]
ROUTES = ["group6_single", "plain12_identity", "plain12_C4", "plain24_identity", "plain24_C4"]
MAIN = ["group6_single", "plain12_C4", "plain24_C4"]
FREE = dict(group6=2671, plain12_C4pooled=2749, plain24_C4pooled=10681)
MAC = dict(zip(ROUTES, [1527552, 390528, 1562112, 1527552, 6110208]))
COLOR = dict(group6_single="#b25a43", plain12_C4="#3c78a8", plain24_C4="#4d8d63")
LABEL = dict(group6_single="Group6 single", plain12_C4="Plain12 C4", plain24_C4="Plain24 C4")
HISTORY = [
    ("sem_arch_iterative_20261102", "η2: direct / split / FiLM / gated; отрицательные исходы"),
    ("sem_arch_amplitude4_20261102", "η4 FiLM; физический критерий и сравнение с дообученной CNN не пройдены"),
    ("sem_arch_joint_adaptation_20261102", "Совместная адаптация; A/B не пройдены"),
    ("sem_arch_early_skip_20261102", "Early skip; A/B/C не пройдены"),
    ("sem_arch_native_detail_20261102", "Native48 detail; A/B/C не пройдены"),
    ("sem_arch_orbit_pool_20261102", "Известный равномерный C4: старые VAL-критерии пройдены; 4 новых CNN-контроля"),
    ("sem_arch_orbit_confirmation_20261102", "Закрытое C4-подтверждение на 8 полях: FAIL_PILOT"),
    ("sem_arch_learned_orbit_20261102", "Обучаемые веса видов; A/B/D не пройдены"),
    ("sem_arch_orbit_residual_20261102", "Residual-поправка маски; A/B/C не пройдены"),
    ("sem_arch_classification_readout_20261102", "Прямая flat9-классификация; A/B/C не пройдены"),
    ("sem_arch_directional_readout_20261102", "Прямая directional-классификация; A/B/C не пройдены"),
    ("sem_arch_anchored_readout_20261102", "Прямой anchored-readout; CNN_SUCCESS не пройден; техническая остановка сохранена"),
]


def count(value):
    return f"{value:,}".replace(",", " ")


def average(result, route, metric="mask_F1", budget=None, calibration="selected"):
    rows = [r for r in result["rows"] if r["route"] == route and (budget is None or r["budget"] == budget)]
    return float(np.mean([(r["selected"] if calibration == "selected" else r[calibration]["selected"])[metric] for r in rows]))


def fit_key(row):
    return row["architecture"], int(row["budget"]), int(row["seed"])


def execution_inventory(trials):
    """Count final trained weights, never route evaluations or materialized copies."""
    executions = {}; references = 0; records = 0
    pattern = re.compile(r"^(group6|plain12_C4pooled|plain24_C4pooled)_b(\d+)_s(\d+)\.pt$")
    for trial in trials:
        path = trial["path"]; result = trial["result"] or {}
        fits = result.get("fits", read(path / "FIT_RESULTS.json", []))
        provenance = {fit_key(item): item for item in result.get("fit_provenance", [])}
        recovery = trial["frozen"].get("recovery", {})
        if recovery.get("inherited_fit"):
            provenance.setdefault(fit_key(recovery["inherited_fit"]),
                                  dict(source_trial=recovery["original_failed_trial"]))
        recorded = set()
        for row in fits:
            key = fit_key(row); recorded.add(key); records += 1; origin = path
            if key in provenance:
                origin = (ROOT / provenance[key]["source_trial"]).resolve()
            assert origin.is_relative_to(ROOT / "trials")
            assert sha(path / row["model_file"]) == row["model_sha256"]
            if origin != path:
                references += 1
                assert sha(origin / row["model_file"]) == row["model_sha256"]
                inherited = recovery["inherited_fit"]
                assert key == fit_key(inherited)
                assert all(row[name] == value for name, value in inherited["fitting_metadata"].items())
            execution = (str(origin), key)
            if execution in executions:
                assert executions[execution]["model_sha256"] == row["model_sha256"]
            executions[execution] = row
        # A failure after saving weights but before assembling its JSON remains a completed fit.
        for model in sorted((path / "models").glob("*.pt")):
            match = pattern.fullmatch(model.name)
            if match:
                key = (match[1], int(match[2]), int(match[3]))
                if key not in recorded:
                    records += 1; origin = (ROOT / provenance[key]["source_trial"]).resolve() if key in provenance else path
                    if origin != path:
                        references += 1
                        assert sha(origin / "models" / model.name) == sha(model)
                    executions.setdefault((str(origin), key), dict(architecture=key[0], budget=key[1], seed=key[2],
                        fit_count=1, optimizer_updates=2000, model_file=rel(model, path), model_sha256=sha(model),
                        metadata_source="saved_final_checkpoint_before_row_assembly"))
    return executions, records, references


def verify(result):
    units = {(r["budget"], r["seed"]) for r in result["rows"]}
    assert len(result["rows"]) == len(units) * 5
    assert len(result["fits"]) == len(units) * 3
    assert len({(r["route"], r["budget"], r["seed"]) for r in result["rows"]}) == len(result["rows"])
    for unit in units:
        fits = [r for r in result["fits"] if (r["budget"], r["seed"]) == unit]
        assert {r["architecture"] for r in fits} == set(ARMS)
        assert len({r["batch_sequence_sha256"] for r in fits}) == 1
        for fit in fits:
            assert fit["fresh_initialization"] and fit["fit_count"] == 1
            assert fit["steps"] == fit["optimizer_updates"] == 2000
            assert fit["warmup_steps"] == 400 and fit["joint_steps"] == 1600
            assert fit["free_parameters"] == FREE[fit["architecture"]]
            assert fit["initial_state_sha256"] != fit["final_state_sha256"]
            assert [probe["step"] for probe in fit["gradient_probes"]] == [1, 401]
        for row in [r for r in result["rows"] if (r["budget"], r["seed"]) == unit]:
            assert row["free_parameters"] == FREE[row["architecture"]]
            assert row["MACs_per_image"] == MAC[row["route"]]
            assert row["selected"] == row["primary_fine"]["selected"]
            assert len(row["primary_fine"]["all_thresholds"]) == 81
            assert len(row["secondary_seven"]["all_thresholds"]) == 7
            assert row["CNN_batch_calls"] == 1
    a, b = result["gates"]["A_matched_quality"], result["gates"]["B_full_efficiency"]
    assert a["passed"] == all(a["checks"].values()) and b["passed"] == all(b["checks"].values())
    assert result["gates"]["ANY_GOAL_PASS"] == (a["passed"] or b["passed"])
    if "robust_primary_pass" in result:
        assert result["robust_primary_pass"] == (a["passed"] and result["new_seed_gates"]["A_matched_quality"]["passed"])
        assert result["robust_secondary_pass"] == (b["passed"] and result["new_seed_gates"]["B_full_efficiency"]["passed"])
        assert result["robust_ANY_GOAL_PASS"] == (result["robust_primary_pass"] or result["robust_secondary_pass"])


def figures(result_views):
    if not result_views:
        return []
    folder = ROOT / "figures"; folder.mkdir(exist_ok=True)
    points = [(label, budget, result) for label, result in result_views for budget in [18, 27]]
    x = np.arange(len(points)); fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for offset, route in zip([-.24, 0, .24], MAIN):
        for axis, metric in zip(axes[0], ["mask_F1", "IoU"]):
            bars = axis.bar(x + offset, [100 * average(r, route, metric, b) for _, b, r in points], .22,
                            label=LABEL[route], color=COLOR[route])
            axis.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
        if route != "group6_single":
            values = [100 * (average(r, "group6_single", budget=b) - average(r, route, budget=b)) for _, b, r in points]
            axis = axes[1, 0] if route == "plain12_C4" else axes[1, 1]
            axis.scatter(x, values, color=COLOR[route], s=55)
    for axis, metric in zip(axes[0], ["Macro-F1", "Mask IoU"]):
        axis.set_ylabel(f"{metric}, %"); axis.set_ylim(0, 100); axis.legend(fontsize=8, loc="lower left")
    axes[0, 0].set_title("Mask-derived classification; common old loss")
    axes[0, 1].set_title("Segmentation quality")
    for axis, title, mean_limit, budget_limit in [
        (axes[1, 0], "A: group − narrow C4 (matched free parameters / MAC)", .5, -.25),
        (axes[1, 1], "B: group − full C4 (separate tolerance goal)", -.5, -1.)]:
        axis.set_title(title, fontsize=10);axis.set_ylabel("F1 difference, percentage points")
        axis.axhline(0, color="black", lw=.8)
        axis.axhline(mean_limit, color="#777777", ls="--", lw=.8, label="Required mean")
        axis.axhline(budget_limit, color="#aaaaaa", ls=":", lw=.8, label="Budget floor")
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xticks(x, [f"{label}\nbudget {b}" for label, b, _ in points]);axis.set_xlim(-.6, len(points) - .4)
    fig.suptitle("Known C4 group encoder on reused VAL; quality goals stay separate", fontsize=11)
    fig.tight_layout(); quality = folder / "groupcnn_quality.png";fig.savefig(quality, dpi=180, bbox_inches="tight");plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for offset, route in zip([-.24, 0, .24], MAIN):
        medians = []; lows = []; highs = []
        for _, budget, result in points:
            bench = [b["routes"][route] for b in result["benchmarks"] if b["budget"] == budget]
            medians.append(1000 * np.median([b["median_seconds"] for b in bench]))
            lows.append(1000 * np.median([b["p10_seconds"] for b in bench]))
            highs.append(1000 * np.median([b["p90_seconds"] for b in bench]))
        axes[0].bar(x + offset, medians, .22, color=COLOR[route], label=LABEL[route],
                    yerr=[np.array(medians) - lows, np.array(highs) - medians], capsize=2)
    for route, field in [("plain12_C4", "paired_group_over_narrow_C4_median"), ("plain24_C4", "paired_group_over_full_C4_median")]:
        values = [np.median([b[field] for b in r["benchmarks"] if b["budget"] == budget]) for _, budget, r in points]
        axes[1].scatter(x + (-.08 if route == "plain12_C4" else .08), values, color=COLOR[route], s=50, label=f"Group / {LABEL[route]}")
    axes[0].set_ylabel("Median latency per batch32, ms");axes[0].set_title("CNN routes only; summaries across seeds")
    axes[1].set_ylabel("Median paired latency ratio");axes[1].set_title("Measured ratio; lower means faster group route")
    axes[1].axhline(1, color="black", lw=.8)
    for axis in axes:
        axis.set_xticks(x, [f"{label}\nbudget {b}" for label, b, _ in points]);axis.legend(fontsize=8)
    fig.suptitle("Descriptive MPS latency; data, materialization and CPU readout excluded", fontsize=11)
    fig.tight_layout(); timing = folder / "groupcnn_latency.png";fig.savefig(timing, dpi=180, bbox_inches="tight");plt.close(fig)
    return [quality, timing]


def gate_lines(result):
    text = ["| Цель | ΔF1, п.п. | ΔF1: 18 / 27, п.п. | ΔIoU, п.п. | Исход | Не пройдено |",
            "|---|---:|---|---:|---|---|"]
    for name in ["A_matched_quality", "B_full_efficiency"]:
        gate = result["gates"][name]
        failed = ", ".join(k for k, value in gate["checks"].items() if not value) or "—"
        text.append(f"| {name} | {pp(gate['delta'])} | {' / '.join(pp(v) for v in gate['budget_deltas'])} | "
                    f"{pp(gate['IoU_delta'])} | {passed(gate['passed'])} | {failed} |")
    return text + [""]


def build():
    trials = discover(ROOT);complete = [t for t in trials if t["result"]];failures = [t for t in trials if t["failure"]]
    pending = [t for t in trials if not t["result"] and not t["failure"]]
    executions, fit_records, inherited_refs = execution_inventory(trials)
    fits = len(executions);updates = sum(r["optimizer_updates"] for r in executions.values())
    previous = read(PRIOR / "REPORT_MANIFEST.json");prior_fits = previous["linked_stage_unique_completed_fits"]
    result_views = [("screen" if r.get("phase") == "screen" else label, r) for label, r in views(trials)]
    for trial in complete:
        assert trial["result"]["frozen_sha256"] == sha(trial["path"] / "FROZEN.json")
        assert trial["frozen"]["fresh_test_access"] is False
        assert sha(trial["path"] / "CONFIG_SNAPSHOT.json") == trial["frozen"]["config_sha256"] == sha(ROOT / "CONFIG.json")
        assert sha(trial["path"] / "PROTOCOL.md") == trial["frozen"]["protocol_sha256"] == sha(ROOT / "PROTOCOL.md")
        for name in ["model.py", "runner.py", "recovery.py"]:
            if name in trial["frozen"]["source_sha256"]:
                assert sha(ROOT / name) == trial["frozen"]["source_sha256"][name]
                assert sha(trial["path"] / "source" / name) == trial["frozen"]["source_sha256"][name]
        verify(trial["result"])
        if trial["result"].get("recovery"):
            result = trial["result"]
            assert result["recovery_proof_sha256"] == sha(trial["path"] / "RECOVERY_PROOF.json")
            assert result["actual_new_fits"] + result["inherited_fits"] == result["total_distinct_fits"] == result["fitted_models"]
            assert result["actual_new_updates"] + result["inherited_updates"] == result["updates"]
            assert len(result["fit_provenance"]) == len(result["fits"])
        if trial["aggregate"]:
            verify(trial["aggregate"])
    images = figures(result_views)
    lead = []
    if result_views:
        lead_result = result_views[0][1]
        lead_label = "Все три seed" if "robust_primary_pass" in lead_result else "Скрининг"
        lead = [f"**{lead_label}:** Macro-F1 group6 **{pct(average(lead_result, 'group6_single'))}%**, "
                f"Plain12 C4 {pct(average(lead_result, 'plain12_C4'))}%, Plain24 C4 {pct(average(lead_result, 'plain24_C4'))}%. "
                f"A {passed(lead_result['gates']['A_matched_quality']['passed'])}; "
                f"B {passed(lead_result['gates']['B_full_efficiency']['passed'])}. "
                "Это результат разработки на прежней VAL; нового закрытого подтверждения encoder пока нет.", ""]
        if "robust_primary_pass" in lead_result:
            lead += [f"Та же цель одновременно на всех трёх и новых двух seed: "
                     f"robust A **{passed(lead_result['robust_primary_pass'])}**, "
                     f"robust B **{passed(lead_result['robust_secondary_pass'])}**. "
                     f"Основной ΔF1: все три {pp(lead_result['gates']['A_matched_quality']['delta'])} п.п.; "
                     f"новые два {pp(lead_result['new_seed_gates']['A_matched_quality']['delta'])} п.п.", ""]
    text = ["# C4-групповая CNN вместо обычного encoder", "",
        "Изменяется encoder исходной масочной задачи: **SEM 12×12 → lifting 6×4 ориентации → "
        "group conv 6×4 (dilation 2) → group conv 6×4 (dilation 4) → среднее по ориентациям, 6 каналов → "
        "1×1 → вероятность металла → бинарная маска → прежнее геометрическое правило класса**. "
        "После каждого слоя 3×3 применяется GELU. Это возврат к классификации, производной от предсказанной маски; "
        "прямые классовые головы этапов 9–11 остаются отдельными отрицательными ветками.", "",
        "**Метрики:** историческое поле JSON `mask_F1` означает **Macro-F1 девяти классов, "
        "вычисленных из бинарной маски**, а не пиксельный F1 сегментации. "
        "Сегментация отдельно оценивается средним IoU предсказанной маски. "
        "В таблицах ниже Macro-F1 всегда относится к классификации.", "",
        *lead,
        "Использована известная конструкция C4 group-CNN с разделяемыми повёрнутыми весами "
        "([Cohen и Welling, 2016](https://proceedings.mlr.press/v48/cohenc16.html)). "
        "Сравнение при близком вычислительном бюджете мотивировано работой "
        "[Roos и Kroon, 2024](https://proceedings.mlr.press/v251/roos24a.html). "
        "Новый общий оператор или глобальная оригинальность не заявляются.", "",
        "## Общий протокол обучения и бюджет", "",
        "Все три CNN обучаются с нуля, без чтения прежних весов. Для каждого бюджета и seed: "
        "400 шагов dense BCE, затем 1 600 шагов dense BCE + прежняя geometry_loss с весом 1. "
        "AdamW: lr=0,0005, weight_decay=0,0001. На каждом шаге общий RNG выбирает 8 dense и 8 weak; "
        "во время warmup weak-индексы выбираются, но не входят в loss. Dense-бюджеты 18 и 27 "
        "(по 2 или 3 примера каждого класса), weak 1 152, VAL 695, subset seed 5303. "
        "Weak-маски не входят в оптимизатор. Новая функция потерь не исследуется.", "",
        "Dense-примеры дают пиксельные SVG-маски для BCE. Weak-примеры дают только прежние "
        "метки двух направлений, полученные из классов 0…8: они входят в geometry_loss через "
        "предсказанную маску и её мягкую проводимость. Прямой обучаемой классовой головы здесь нет; "
        "при оценке класс вновь определяется жёстким геометрическим правилом бинарной маски.", "",
        "Обе обычные CNN обучаются через тот же четырёхвидовой probability-pool, который используется при C4-инференсе. "
        "Четыре поворота объединены в один CNN-вызов с внутренним батчем 4N, затем вероятности обратно "
        "поворачиваются и усредняются. Group6 использует один вид и один проход. Identity-маршруты "
        "обычных CNN приведены как диагностика тех же весов и не являются дополнительными fit.", "",
        "| Модель / основной маршрут | Свободных параметров | Коэффициентов при исполнении | MAC на изображение | Видов |",
        "|---|---:|---:|---:|---:|",
        "| Group6 single | 2 671 | 10 681 связанных коэффициентов | 1 527 552 | 1 |",
        "| Plain12 C4 | 2 749 | 2 749, одни веса для 4 видов | 1 562 112 | 4 |",
        "| Plain24 C4 | 10 681 | 10 681, одни веса для 4 видов | 6 110 208 | 4 |", "",
        "Материализация group6 выполняется один раз: 10 681 коэффициент не означает 10 681 "
        "независимый обучаемый параметр. Group6 / Plain12: параметры −2,84%, MAC −2,21%; "
        "Group6 / Plain24 C4: MAC 0,25. MAC не отождествляется с измеренной задержкой.", "",
        "Численная C4-эквивариантность и совпадение свободной/материализованной модели проверяются отдельно. "
        "Заранее заданный допуск вероятностей обученных моделей — 1e−5; допуск preflight — 1e−6. "
        "Фактические ошибки каждого маршрута приведены в таблицах и независимом аудите. "
        "Identity-маршруты обычных CNN не обязаны быть эквивариантными.", "",
        "## Две заранее раздельные цели", "",
        "**A_matched_quality (основная):** group6 против Plain12 C4 при различиях параметров и MAC не более 3%. "
        "Средний ΔMacro-F1 ≥+0,5 п.п.; каждый бюджет ≥−0,25 п.п.; средний ΔIoU ≥−0,2 п.п.", "",
        "**B_full_efficiency (вторичная):** group6 против Plain24 C4. Средний ΔF1 ≥−0,5 п.п.; "
        "каждый бюджет ≥−1 п.п.; средний ΔIoU ≥−0,2 п.п.; MAC-отношение ≤0,30. "
        "Это заранее заданная допустимая потеря на использованной VAL, а не статистическое доказательство noninferiority. "
        "Прохождение B не переименовывается в превосходство A.", "",
        "Порог каждой маски выбирается по общей сетке 0,10…0,90 с шагом 0,01 (81 значение); "
        "ранжирование: Macro-F1, затем IoU, близость к 0,5, затем меньший порог. "
        "Прежняя сетка из 7 порогов — только вторичная диагностика и не заменяет основную.", "",
        "После независимого технического PASS любой пройденный критерий разрешает расширение seed. "
        "Для каждой заявляемой цели затем требуются **и все три seed, и отдельно новые два**: "
        "`(A_all3 AND A_new2) OR (B_all3 AND B_new2)`. Смешивать A одной группы seed с B другой нельзя.", "",
        f"Начато запусков {len(trials)}, завершено {len(complete)}, технических остановок {len(failures)}, "
        f"пока незавершённых {len(pending)}. "
        f"Различных завершённых обучений здесь **{fits}**, обновлений **{count(updates)}**. "
        f"Прежних обучений {prior_fits}; всего **{prior_fits + fits}**. "
        f"Сохранённых fit-записей {fit_records}, унаследованных ссылок {inherited_refs}; "
        "оценочные маршруты, агрегаты и материализованные копии повторно не считаются.", ""]
    for label, result in result_views:
        text += [f"## {label}: seed {result['seeds']}", "",
                 f"Основная A: **{passed(result['gates']['A_matched_quality']['passed'])}**; "
                 f"вторичная B: **{passed(result['gates']['B_full_efficiency']['passed'])}**.", ""]
        text += gate_lines(result)
        text += ["| Бюджет | Маршрут | Macro-F1, % | IoU, % | Macro-F1 по 7 порогам, % |", "|---:|---|---:|---:|---:|"]
        for budget in [18, 27]:
            for route in ROUTES:
                text.append(f"| {budget} | {route} | {pct(average(result, route, budget=budget))} | "
                            f"{pct(average(result, route, 'IoU', budget))} | {pct(average(result, route, budget=budget, calibration='secondary_seven'))} |")
        text.append("")
        if result.get("phase") == "screen" and not result["gates"]["ANY_GOAL_PASS"]:
            text += ["Ни одна заранее заданная цель не пройдена; условие расширения по seed не выполнено.", ""]
        if "robust_primary_pass" in result:
            text += [f"Одновременно все три и новые два seed: robust A **{passed(result['robust_primary_pass'])}**, "
                     f"robust B **{passed(result['robust_secondary_pass'])}**; "
                     f"robust ANY_GOAL_PASS **{passed(result['robust_ANY_GOAL_PASS'])}**.", ""]
    if not result_views:
        text += ["Завершённого научного результата пока нет. Критерии качества не объявляются пройденными.", ""]
    if images:
        text += ["![Качество и отдельные цели](figures/groupcnn_quality.png)", "",
                 "Точки показывают бюджетные разности; пунктир — требуемое среднее, точки не должны "
                 "сравниваться с ним как отдельные бюджетные пороги. Бюджетный предел отмечен отдельно. "
                 "Средние и все guard-критерии приведены в таблице; доверительные интервалы не оценивались.", ""]
    text += ["## Измеренная задержка — описательный результат", "",
        "MPS, batch 32, 10 прогревочных батчей на маршрут и 50 парных повторов со случайным порядком "
        "пяти маршрутов. Синхронизация выполняется до и после каждого измерения. Для обычной C4-CNN "
        "повороты и probability-pool входят во время, четыре вида выполняются одним вызовом с батчем 128. "
        "Материализация group-весов, загрузка данных, host-transfer и общий CPU geometric readout исключены. "
        "End-to-end задержка не измерена. Скорость не входит в A/B и не выводится только из MAC.", ""]
    for label, result in result_views:
        text += [f"**{label}:**", "", "| Бюджет | Seed | Маршрут | Median batch32, мс | p10 / p90, мс |", "|---:|---:|---|---:|---|"]
        for bench in result["benchmarks"]:
            for route in ROUTES:
                values = bench["routes"][route]
                text.append(f"| {bench['budget']} | {bench['seed']} | {route} | {1000 * values['median_seconds']:.3f} | "
                            f"{1000 * values['p10_seconds']:.3f} / {1000 * values['p90_seconds']:.3f} |")
        text += ["", "| Бюджет | Seed | Парный Group / Plain12 C4 | Парный Group / Plain24 C4 | Материализация вне окна, с |", "|---:|---:|---:|---:|---:|"]
        for bench in result["benchmarks"]:
            text.append(f"| {bench['budget']} | {bench['seed']} | {bench['paired_group_over_narrow_C4_median']:.3f} | "
                        f"{bench['paired_group_over_full_C4_median']:.3f} | {bench['materialization_setup_seconds']:.6f} |")
        text.append("")
    if images:
        text += ["![Описательная задержка](figures/groupcnn_latency.png)", "",
                 "Для нескольких seed столбцы и концы диапазонов — медианы соответствующих seed-сводок, "
                 "а не объединённый доверительный интервал. Парное отношение — медиана 50 отношений в каждом seed; "
                 "при агрегировании показана медиана seed-сводок.", ""]
    text += ["## Все сохранённые запуски", ""]
    for trial in trials:
        path = trial["path"];result = trial["result"]
        text += [f"### {path.name}", "", f"[FROZEN.json]({rel(path / 'FROZEN.json', ROOT)}).", ""]
        if trial["failure"]:
            failure = trial["failure"]
            text += [f"Сохранённая техническая остановка `{failure['error_type']}`: "
                     f"{str(failure['error']).replace(chr(10), ' ')}. [FAILURE.json]({rel(path / 'FAILURE.json', ROOT)}). "
                     "Она не является результатом A/B; сохранённые веса учитываются по исходному выполнению.", ""]
        if not result:
            completed_in_trial = sum(origin == str(path) for origin, _ in executions)
            if not trial["failure"]:
                text += ["Запуск выполняется; научный итог этой попытки пока не сохранён.", ""]
            text += [f"Завершённых обучений с сохранёнными итоговыми весами: {completed_in_trial}. "
                     f"Сохранённых оценочных строк: {len(trial['partial'])}.", ""];continue
        text += [f"`{result['status']}`; {result['phase']}; seed {result['seeds']}; "
                 f"{result['fitted_models']} различных моделей в результате, {count(result['updates'])} суммарных обновлений, "
                 f"{len(result['rows'])} оценочных строк; {result['seconds']:.2f} с этого запуска. "
                 f"[RESULT.json]({rel(path / 'RESULT.json', ROOT)}), SHA-256 `{sha(path / 'RESULT.json')}`.", ""]
        if result.get("recovery"):
            text += [("Восстановление скрининга: " if result["inherited_fits"] else "Расширение через recovery.py: ") +
                     f"унаследованных обучений {result['inherited_fits']}, "
                     f"выполненных здесь {result['actual_new_fits']}; новых обновлений {count(result['actual_new_updates'])}. "
                     "Повторная оценка прежней модели без оптимизатора не считается новым обучением. "
                     f"[RECOVERY_MANIFEST.json]({rel(path / 'RECOVERY_MANIFEST.json', ROOT)}) · "
                     f"[RECOVERY_PROOF.json]({rel(path / 'RECOVERY_PROOF.json', ROOT)}). "
                     "Исправление относится к сборке метаданных; модель, данные, loss, батчи и научные критерии сохранены.", ""]
        text += ["| Бюджет | Seed | Архитектура | Шагов | Warmup / joint | Время обучения, с |", "|---:|---:|---|---:|---|---:|"]
        for fit in result["fits"]:
            text.append(f"| {fit['budget']} | {fit['seed']} | {fit['architecture']} | {fit['steps']} | "
                        f"{fit['warmup_steps']} / {fit['joint_steps']} | {fit['training_seconds']:.2f} |")
        text += ["", "| Бюджет | Seed | Маршрут | Порог | Macro-F1, % | IoU, % | C4 max error |", "|---:|---:|---|---:|---:|---:|---:|"]
        for row in result["rows"]:
            text.append(f"| {row['budget']} | {row['seed']} | {row['route']} | {row['selected']['threshold']:.2f} | "
                        f"{pct(row['selected']['mask_F1'])} | {pct(row['selected']['IoU'])} | {row['C4_probability_max_error']:.3g} |")
        text += [""] + gate_lines(result)
        audit = trial["audit"]
        text += [(f"Независимый CPU-аудит: **{audit['status']}**, {audit.get('checks_passed', '?')} / {audit.get('checks_total', '?')}. "
                  f"[AUDIT.json]({rel(path / 'AUDIT.json', ROOT)})." if audit else "Независимый аудит ещё не сохранён."), ""]
        if trial["aggregate"]:
            text += [f"[AGGREGATE_RESULT.json]({rel(path / 'AGGREGATE_RESULT.json', ROOT)}) содержит исходные и новые seed; "
                     "его fit и маршруты повторно не суммируются.", ""]
        for old in sorted(path.glob("AUDIT_PREVIOUS_*.json")):
            text += [f"Предыдущий аудит [{old.name}]({rel(old, ROOT)}): `{read(old).get('status')}`; исходный файл сохранён.", ""]
    text += ["## История попыток и границы вывода", "",
        "Число запусков приведено из manifest каждого этапа; общие CNN-контроли входят в число fit "
        "только в месте их выполнения. Закрытое подтверждение — отдельная оценка без обучения.", "",
        "| Этап | Запусков | Новых различных обучений | Сохранённый исход |", "|---|---:|---:|---|"]
    history_inputs = [];history_fits = 0
    for folder, outcome in HISTORY:
        directory = ROOT.parent / folder;manifest = read(directory / "REPORT_MANIFEST.json")
        n = manifest.get("actual_completed_fits", manifest.get("new_training_fits", 0));history_fits += n
        history_inputs += [directory / "REPORT_MANIFEST.json", directory / "REPORT.md"]
        text.append(f"| [{folder}]({rel(directory / 'REPORT.md', ROOT)}) | {manifest.get('trial_count', 'оценка')} | {n} | {outcome} |")
    assert history_fits == prior_fits
    text += [f"| Текущая групповая CNN | {len(trials)} | {fits} | Отдельные A/B и технические запуски показаны выше |", "",
        "Это многократно использованные TRAIN/VAL одной серии, без нового закрытого подтверждения group-CNN. "
        "Прежний FAIL_PILOT равномерного C4 на восьми свежих полях той же ИС сохранён и не переопределён "
        "как проверка нового encoder. Данные этого закрытого подтверждения здесь не использовались. "
        "Новые seed повторяют обучение на той же VAL и не являются независимым тестом.", "",
        "Классы получены из SVG-масок прежним бинарным геометрическим правилом, а не экспертной типологией "
        "дефектов или измерением электрического состояния. Технический PASS не заменяет критерий качества. "
        "Перенос между изделиями, статистическое закрытое подтверждение и готовность результата для Q1 не установлены.", "",
        "[PROTOCOL.md](PROTOCOL.md) · [CONFIG.json](CONFIG.json) · [LEDGER.json](LEDGER.json) · "
        "[model.py](model.py) · [исходный runner.py](runner.py) · [исполняемый recovery.py](recovery.py) · [audit.py](audit.py).", ""]
    (ROOT / "REPORT.md").write_text("\n".join(text))
    latest = result_views[0][1] if result_views else None
    state = (("Скрининг на прежней VAL: " if latest.get("phase") == "screen" else "Разработка на прежней VAL, все три seed: ") +
             f"A: **{passed(latest['gates']['A_matched_quality']['passed'])}**; B: **{passed(latest['gates']['B_full_efficiency']['passed'])}**. " if latest else "Завершённого результата пока нет. ")
    if latest and "robust_primary_pass" in latest:
        state += f"Одна и та же цель на всех трёх и новых двух seed: robust A **{passed(latest['robust_primary_pass'])}**, robust B **{passed(latest['robust_secondary_pass'])}**. "
    if latest:
        state += f"Macro-F1 group6 {pct(average(latest, 'group6_single'))}%, Plain12 C4 {pct(average(latest, 'plain12_C4'))}%, Plain24 C4 {pct(average(latest, 'plain24_C4'))}%. "
    audit_state = "; ".join(f"{t['path'].name}: {t['audit']['status']} ({t['audit'].get('checks_passed', '?')}/{t['audit'].get('checks_total', '?')})" for t in complete if t["audit"])
    closed_root = ROOT.parent / "sem_arch_groupcnn_confirmation_20261102"
    closed_result = read(closed_root / "RESULT.json");closed_audit = read(closed_root / "AUDIT.json")
    closed_data_audit = read(closed_root / "DATA_AUDIT.json")
    closed_final = bool(closed_result and closed_audit and closed_data_audit and
        closed_audit["status"].startswith("PASS") and closed_data_audit["status"].startswith("PASS") and
        all(closed_audit["checks"].values()) and all(closed_data_audit["checks"].values()) and
        closed_audit["result_sha256"] == sha(closed_root / "RESULT.json"))
    closed_note = "На этапе этой разработки закрытого подтверждения нового encoder ещё не было."
    if closed_final:
        outcomes = "; ".join(f"{name}: {'подтверждена' if closed_result['confirmed_by_goal'][goal] else 'не подтверждена'}"
                             for name, goal in [("A при близких параметрах/MAC", "A_matched_quality"), ("B против полной C4", "B_full_efficiency")])
        closed_note += f" Последующая [закрытая проверка]({rel(closed_root / 'REPORT.md', ROOT)}): **{outcomes}** "
        closed_note += "на 4 096 новых фрагментах 16 полей той же ИС, без новых fit. Её исход не меняет исторические VAL-таблицы."
    (ROOT / "README.md").write_text(
        "# C4-групповая CNN для исходной масочной задачи\n\n"
        "SEM 12×12 → lifting 6×4 → две group-conv с dilation 2/4 → среднее ориентаций → 1×1 маска → "
        "прежнее бинарное геометрическое правило класса. Все три модели с нуля: 400 BCE + 1 600 BCE/legacy geometry.\n\n"
        "`mask_F1` в JSON — Macro-F1 девяти классов из бинарной маски, не пиксельный F1. "
        "Отдельная метрика сегментации — IoU.\n\n"
        "Group6: 2 671 свободный параметр, 1,528 M MAC. Основной контроль Plain12 C4: 2 749, 1,562 M MAC; "
        "полный Plain24 C4: 10 681, 6,110 M MAC. A проверяет качество при близком бюджете; B отдельно проверяет "
        "допустимую потерю к полной модели. Задержка MPS описательная, end-to-end не измерена.\n\n"
        + state + f"Различных обучений здесь {fits}; с прежними **{prior_fits + fits}**. "
        f"Незавершённых запусков {len(pending)}. "
        "Технические остановки и наследование весов сохранены в отчёте. Известный метод; новая loss или общая новизна не заявлены.\n\n"
        + (f"Независимый аудит: {audit_state}.\n\n" if audit_state else "")
        + closed_note + "\n\n"
        +
        "Исполняемый entrypoint обучения — [recovery.py](recovery.py); исходный runner.py с ошибкой сборки метаданных сохранён. "
        "Расширение seed разрешается только после независимого аудита и прохождения A или B.\n\n"
        "[Исторический отчёт разработки](REPORT.md) · [Текущая гипотеза](HYPOTHESIS.md) · "
        "[Протокол](PROTOCOL.md) · [Конфигурация](CONFIG.json) · [Журнал](LEDGER.json). "
        "Числа разработки относятся только к прежним TRAIN/VAL.\n\n"
        "```bash\n.venv/bin/python -B gipoteza/sem_arch_groupcnn_20261102/report.py\n```\n")
    inputs = [HELPER] + history_inputs + [ROOT / name for name in ["CONFIG.json", "PROTOCOL.md", "LEDGER.json", "PREFLIGHT.json", "model.py", "runner.py", "audit.py", "recovery.py", "RECOVERY_PREFLIGHT.json"] if (ROOT / name).exists()]
    if closed_final:
        inputs += [closed_root / name for name in ["RESULT.json", "AUDIT.json", "DATA_AUDIT.json"]]
    for trial in trials:
        inputs += [path for path in trial["path"].rglob("*") if path.is_file() and path.suffix in {".json", ".npz", ".pt", ".py", ".md"}]
    outputs = [ROOT / "REPORT.md", ROOT / "README.md"] + images
    manifest = dict(status="REPORT_GENERATED", UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="FRESH_KNOWN_C4_GROUP_ENCODER_OLD_TRAIN_VAL", report_script_sha256=sha(Path(__file__)),
        trial_count=len(trials), completed_trials=len(complete), failed_trials=len(failures),
        pending_trials=len(pending),
        architecture_candidate_count=1, actual_completed_fits=fits, completed_optimizer_updates=updates,
        saved_fit_records=fit_records, inherited_fit_references=inherited_refs,
        prior_unique_completed_fits=prior_fits, linked_stage_unique_completed_fits=prior_fits + fits,
        evaluation_rows=sum(len(t["result"]["rows"]) if t["result"] else len(t["partial"]) for t in trials),
        unique_execution_origins={f"{rel(origin, ROOT)}:{list(key)}": row["model_sha256"] for (origin, key), row in executions.items()},
        inference_route_rows_are_fits=False, materialized_copies_are_fits=False,
        primary_and_secondary_goals_separate=True, replication_success_rule=read(ROOT / "CONFIG.json")["replication_success_rule"],
        primary_pass=latest["gates"]["A_matched_quality"]["passed"] if latest else None,
        secondary_pass=latest["gates"]["B_full_efficiency"]["passed"] if latest else None,
        robust_primary_pass=latest.get("robust_primary_pass") if latest else None,
        robust_secondary_pass=latest.get("robust_secondary_pass") if latest else None,
        robust_ANY_GOAL_PASS=latest.get("robust_ANY_GOAL_PASS") if latest else None,
        technical_audit_status={t["path"].name: (t["audit"] or {}).get("status") for t in complete},
        development_results_preserved_as_historical=True,
        subsequent_confirmation_metadata_linked=closed_final,
        subsequent_confirmation_status=closed_result["status"] if closed_final else None,
        subsequent_confirmed_by_goal=closed_result["confirmed_by_goal"] if closed_final else None,
        end_to_end_latency_measured=False, fresh_test_access=False,
        scientific_inputs_sha256={rel(path, ROOT): sha(path) for path in sorted(set(inputs))},
        outputs_sha256={rel(path, ROOT): sha(path) for path in outputs})
    (ROOT / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ["scientific_inputs_sha256", "outputs_sha256", "unique_execution_origins"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
