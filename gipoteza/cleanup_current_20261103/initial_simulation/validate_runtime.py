"""Read-only runtime checks, optionally denying access to every planned deletion."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO, G = HERE.parents[1], HERE.parent


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name, p):
    spec = importlib.util.spec_from_file_location(name, p)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def child(name, planned):
    if planned:
        deny = {str((REPO / r['path']).absolute()) for r in read(HERE / 'PLAN.json')['remove']}
        def hook(event, args):
            if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
                p = os.path.abspath(os.fsdecode(args[0]))
                if p in deny: raise RuntimeError('Attempt to read a planned deletion: ' + p)
        sys.addaudithook(hook)
    import numpy as np
    if name == 'freeze':
        p = G / 'sem_model_freeze_20261103'; c = module('cleanup_readonly_freeze', p / 'freeze.py')
        d = c.verify(read(p / 'FROZEN_MODEL.json')); assert d['status'].startswith('PASS')
    elif name in ['seed', 'calibrated', 'dev', 'untied', 'group_confirmation', 'bce', 'factorial']:
        paths = dict(seed='sem_arch_seed_extension_20261103', calibrated='sem_arch_calibrated_confirmation_20261103',
            dev='sem_arch_calibration_dev_20261103', untied='sem_arch_untied_control_20261103',
            group_confirmation='sem_arch_groupcnn_confirmation_20261102',
            bce='sem_arch_bce_confirmation_20261103', factorial='sem_arch_loss_factorial_20261103')
        p = G / paths[name]; sys.path.insert(0, str(p))
        c = module('cleanup_readonly_' + name, p / ('confirmation_common.py' if name == 'group_confirmation' else 'common.py'))
        c.verify_lock()
    elif name == 'timing':
        p = G / 'sem_arch_pipeline_timing_20261103'; c = module('cleanup_readonly_timing', p / 'benchmark.py')
        assert c.sources() == read(p / 'LOCK.json')['input_sha256']
    elif name == 'EDA':
        pilot = G / 'sem_architecture_pilot_20261102'; manifest = read(pilot / 'DATA_MANIFEST.json')
        assert sha(pilot / 'DATA.npz') == manifest['DATA_sha256']
        with np.load(pilot / 'DATA.npz', allow_pickle=False) as z:
            frames = np.unique(z['frame']); assert len(z['X']) == 15105 and len(frames) == 16
        from PIL import Image
        for frame in frames:
            directory = G / 'sem_confirmation_20261028/sources' if frame < 8 else pilot / 'sources'
            assert Image.open(directory / f'sem{frame:04}.png').size == (4096, 3536)
            assert (directory / f'label{frame:04}.svg').is_file()
        notebook = read(G / 'EDA_IC_SEM_M2.ipynb')
        assert sum(c['cell_type'] == 'code' and c.get('execution_count') is not None for c in notebook['cells']) == 17
    elif name == 'inference':
        import torch
        torch.set_num_threads(2)
        p = G / 'sem_arch_seed_extension_20261103'; old = G / 'sem_arch_calibrated_confirmation_20261103'
        factory = module('cleanup_readonly_model_factory', G / 'sem_arch_groupcnn_20261102/model.py')
        old_rows = [r for r in read(old / 'MODEL_LOCK.json')['rows'] if r['architecture'] in ['group6', 'untied24_single']]
        new_rows = read(p / 'CALIBRATION.json')['rows']
        rows = old_rows + new_rows; assert len(rows) == 10
        with np.load(old / 'EVAL.npz') as z: x = torch.tensor(5 * (z['X'][:17] - .2))
        with np.load(p / 'PREDICTIONS.npz') as z: probability = z['probability'][:, :, :17]; seeds=z['seeds'].tolist()
        errors = []
        for row in rows:
            model = factory.make_plain(24)
            model.load_state_dict(torch.load(REPO / row['prediction_model_file'], map_location='cpu', weights_only=True)['model'])
            model.eval()
            with torch.inference_mode(): out = model(x).sigmoid().numpy()[:, 0]
            a = ['group6', 'untied24_single'].index(row['architecture']); s=seeds.index(row['seed'])
            error=float(np.max(np.abs(out - probability[a, s]))); assert error <= 1e-5, (row['seed'], row['architecture'], error)
            errors.append(error)
        print(json.dumps(dict(models=10, examples=17, maximum_probability_error=max(errors))), flush=True)
    print('PASS ' + name, flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--planned', action='store_true'); parser.add_argument('--child')
    a=parser.parse_args()
    if a.child: child(a.child, a.planned); return
    checks={}; outputs={}
    for name in ['freeze','seed','calibrated','dev','untied','group_confirmation','bce','factorial','timing','EDA','inference']:
        cmd=[sys.executable,'-B',str(Path(__file__)),'--child',name] + (['--planned'] if a.planned else [])
        r=subprocess.run(cmd,cwd=REPO,capture_output=True,text=True)
        outputs[name]=r.stdout+r.stderr; checks[name]=r.returncode==0
        print(json.dumps(dict(check=name,passed=checks[name],details=outputs[name][-3000:])),flush=True)
    result=dict(status='PASS_CLEANUP_RUNTIME' if all(checks.values()) else 'FAIL_CLEANUP_RUNTIME',
        planned_deletions_denied=a.planned,checks=checks,checks_passed=sum(checks.values()),checks_total=len(checks),
        new_training_fits=0,scientific_records_changed=False,outputs=outputs,source_sha256=sha(Path(__file__)))
    (HERE / ('PRE_RUNTIME.json' if a.planned else 'POST_RUNTIME.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    assert all(checks.values()), checks


if __name__ == '__main__': main()
