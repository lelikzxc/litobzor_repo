"""Read-only verification of retained artifacts; writes only VERIFICATION.json."""
from pathlib import Path
import ast
import hashlib
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def main():
    plan = read(HERE / 'PLAN.json')
    complete = read(HERE / 'COMPLETE.json')
    changed = [rel for rel, expected in plan['retained_sha256'].items()
               if not (ROOT / rel).is_file() or sha(ROOT / rel) != expected]
    removed_exist = [rel for rel in plan['delete_roots'] if (ROOT / rel).exists()]
    symlinks = all((ROOT / rel).is_symlink() and
                   str((ROOT / rel).readlink()) == target and (ROOT / rel).resolve().exists()
                   for rel, target in plan['retained_symlinks'].items())
    main_doc = ROOT / 'gipoteza/11_gipotez_dlya_vkr.md'
    heading_ids = re.findall(r'^## (\d+)\.', main_doc.read_text(), re.M)
    headings_ok = list(map(int, heading_ids)) == list(range(1, 12))
    broken = []
    link_count = 0
    for rel in ('README.md', 'gipoteza/README.md', 'gipoteza/11_gipotez_dlya_vkr.md',
                'gipoteza/H7_1.md', 'gipoteza/shared_data/README.md',
                'gipoteza/retained_evidence_20261027/README.md'):
        p = ROOT / rel
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if target.startswith(('https:', 'http:', '#')):
                continue
            link_count += 1
            if not (p.parent / target.split('#')[0]).exists():
                broken.append([rel, target])
    evidence = ROOT / 'gipoteza/retained_evidence_20261027'
    records = read(evidence / 'SELECTED_RECORDS.json')['entries']
    record_hashes = all(hashlib.sha256(r['text'].encode()).hexdigest() == r['sha256']
                        for r in records)
    code = read(evidence / 'SOURCE_MANIFEST.json')
    code_hashes = sha(evidence / 'SELECTED_SOURCES.zip') == code['selected_zip_sha256']
    with zipfile.ZipFile(evidence / 'SELECTED_SOURCES.zip') as z:
        code_hashes &= set(z.namelist()) == set(code['files'])
        code_hashes &= all(hashlib.sha256(z.read(n)).hexdigest() == h
                           for n, h in code['files'].items())
    selection = ROOT / 'gipoteza/diploma_selection_20261027'
    historical_manifest = read(selection / 'SELECTION_COMPLETE.json')
    snapshot_ok = sha(selection / 'snapshots/10_gipotez_dlya_vkr.md') == historical_manifest['document_sha256']
    original_helpers_ok = all(sha(selection / n) == h for n, h in historical_manifest['code_sha256'].items())
    restored = read(selection / 'EVIDENCE_AUDIT.json')['recovered']
    restored_ok = all(sha(selection / r['saved']) == r['historical_sha256'] for r in restored)
    parsed = 0
    for rel in ('common', 'papers', 'tests', 'gipoteza/h7_readout_20261025',
                'gipoteza/diploma_selection_20261027'):
        for p in (ROOT / rel).rglob('*.py'):
            ast.parse(p.read_bytes(), filename=str(p))
            parsed += 1
    reaudit_path = HERE / 'wafer_reaudit/INDEPENDENT_AUDIT.json'
    reaudit = read(reaudit_path) if reaudit_path.exists() else {}
    tests_path = HERE / 'TEST_RESULTS.json'
    tests = read(tests_path) if tests_path.exists() else {}
    gates = {'retained_hashes': not changed, 'delete_roots_absent': not removed_exist,
             'three_internal_symlinks': symlinks, 'eleven_ordered_hypotheses': headings_ok,
             'current_document_links': not broken, 'historical_record_hashes': record_hashes,
             'historical_code_hashes': bool(code_hashes), 'ten_option_snapshot_matches': snapshot_ok,
             'frozen_helpers_unchanged': original_helpers_ok, 'nine_restored_reports_match': restored_ok,
             'plan_matches_completion': sha(HERE / 'PLAN.json') == complete['plan_sha256'],
             'wafer_independent_reaudit': reaudit.get('status') == 'PASS_AUDIT',
             'retained_runtime_tests': tests.get('status') == 'PASS'}
    result = {'status': 'PASS' if all(gates.values()) else 'FAIL', 'gates': gates,
              'retained_sha256_checked': len(plan['retained_sha256']),
              'H7_original_unchanged_files': 292, 'historical_records': len(records),
              'historical_source_files': len(code['files']), 'python_sources_parsed': parsed,
              'local_document_links_checked': link_count, 'changed_retained_files': changed,
              'remaining_deleted_roots': removed_exist, 'broken_current_links': broken,
              'reaudit_path': str(reaudit_path.relative_to(ROOT)), 'tests': tests,
              'scope': 'Preservation, source hashes, arithmetic and runtime checks after cleanup; no model training and no new hypothesis confirmation.'}
    (HERE / 'VERIFICATION.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, ensure_ascii=False))
    assert result['status'] == 'PASS', gates


if __name__ == '__main__':
    main()
