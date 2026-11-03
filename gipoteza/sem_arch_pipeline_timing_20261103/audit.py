"""Reconstruct timing summaries/coverage and verify immutable measurement inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    d = json.loads((ROOT / 'TIMING.json').read_text())
    lock = json.loads((ROOT / 'LOCK.json').read_text())
    pre = json.loads((ROOT / 'PREFLIGHT.json').read_text())
    checks = dict(
        locked_inputs_unchanged=all(sha(REPO / p) == h for p, h in lock['input_sha256'].items()),
        measurement_binds_lock=d['lock_sha256'] == sha(ROOT / 'LOCK.json'),
        measurement_input_bindings=d['input_sha256'] == lock['input_sha256'],
        CPU_preflight_bound=lock['preflight_sha256'] == sha(ROOT / 'PREFLIGHT.json') and
                            pre['input_sha256'] == lock['input_sha256'] and all(pre['checks'].values()),
        same_96_selected_rows=d['selected_indices'] == pre['indices'] and len(set(d['selected_indices'])) == 96,
        MPS_without_fallback=d['device'] == 'mps' and not d['hardware']['mps_fallback'],
        all_correctness_assertions_completed=d['correctness_checks_all_passed'],
        positive_finite_durations=all(np.isfinite(r['nanoseconds']) and r['nanoseconds'] > 0 for r in d['measurements']),
        no_new_fits=d['new_fits'] == 0,
    )
    coverage, summaries = True, True
    for batch in [1, 16]:
        for cache in ['cold', 'warm']:
            for arch in ['group6', 'untied24_single']:
                rows = [r for r in d['measurements'] if (r['batch'], r['cache'], r['architecture']) == (batch, cache, arch)]
                expected = {(seed, repeat, start) for seed in [5701, 5721, 5757]
                            for repeat in range(3) for start in range(0, 96, batch)}
                observed = [(r['seed'], r['repeat'], r['start']) for r in rows]
                coverage &= len(observed) == len(expected) and set(observed) == expected
                values = np.array([r['nanoseconds'] / 1e6 for r in rows])
                summary = next(r for r in d['summary'] if (r['batch'], r['cache'], r['architecture']) == (batch, cache, arch))
                computed = dict(median_ms=np.median(values), p95_ms=np.quantile(values, .95), mean_ms=values.mean(),
                                median_ms_per_patch=np.median(values) / batch,
                                p95_request_ms_per_patch=np.quantile(values, .95) / batch)
                summaries &= summary['requests'] == len(expected) and all(np.isclose(summary[k], v, rtol=0, atol=1e-12) for k, v in computed.items())
    checks.update(exact_request_coverage=bool(coverage) and len(d['measurements']) == 3672,
                  all_summaries_reconstructed=bool(summaries),
                  archived_probability_tolerance=all(r['probability_error'] <= 1e-5 for r in d['measurements']))
    profile_coverage = {(arch, seed, repeat, start) for arch in ['group6', 'untied24_single']
                        for seed in [5701, 5721, 5757] for repeat in range(3) for start in range(0, 96, 16)}
    checks['separate_profile_coverage'] = len(d['profiles']) == 108 and {
        (r['architecture'], r['seed'], r['repeat'], r['start']) for r in d['profiles']} == profile_coverage
    checks = {k: bool(v) for k, v in checks.items()}
    assert all(checks.values()), checks
    report = dict(status='PASS_PIPELINE_TIMING_CHECKS', checks=checks, checks_passed=len(checks),
                  checks_total=len(checks), source_inputs_checked=len(lock['input_sha256']),
                  timing_sha256=sha(ROOT / 'TIMING.json'), lock_sha256=sha(ROOT / 'LOCK.json'),
                  audit_source_sha256=sha(Path(__file__)),
                  scope='Artifact/coverage/statistic checks; root-authored. Separate read-only code review found no blocker; not an independent rerun of hardware timing.')
    (ROOT / 'CHECKS.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
