"""Frozen historical TRAIN/VAL sources for a same-compute weight-tying ablation."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
STAGE = ROOT.parent / 'sem_arch_groupcnn_20261102'
PILOT = ROOT.parent / 'sem_architecture_pilot_20261102'
AGG = STAGE/'trials/groupcnn_replicate_s5721-5757_metadata_recovery_27b9055b42/AGGREGATE_RESULT.json'
ARMS = ['group6', 'untied24_single', 'plain12_C4pooled']
SEEDS = [5701, 5721, 5757]
MACS = {'group6': 1527552, 'untied24_single': 1527552, 'plain12_C4pooled': 1562112}
PARAMETERS = {'group6': 2671, 'untied24_single': 10681, 'plain12_C4pooled': 2749}


def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


model=module('untied_frozen_group_model',STAGE/'model.py')
previous=sys.modules.get('model');sys.modules['model']=model
try: historical=module('untied_frozen_group_runner',STAGE/'runner.py')
finally:
    if previous is None:del sys.modules['model']
    else:sys.modules['model']=previous


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def relative(path):return str(Path(path).resolve().relative_to(REPO))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+f'.{os.getpid()}.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)


def config():
    value=read(ROOT/'CONFIG.json')
    assert value['budget']==27 and value['seeds']==SEEDS
    assert value['steps']==2000 and value['warmup_steps']==400 and value['batch_size']==8
    assert value['learning_rate']==.0005 and value['weight_decay']==.0001 and value['physics_weight']==1.
    assert value['batch_seed_offset']==1000 and value['diagnostic_threshold']==.5
    assert value['thresholds']==[round(.1+.01*i,2) for i in range(81)]
    assert value['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
    return value


def old_rows():
    aggregate=read(AGG);result=[]
    assert aggregate['seeds']==SEEDS
    for name in aggregate['source_trials']:
        folder=STAGE/name;old=read(folder/'RESULT.json');audit=read(folder/'AUDIT.json')
        assert audit['checks_passed']==audit['checks_total'] and all(audit['checks'].values())
        for fit in old['fits']:
            if fit['budget']!=27 or fit['architecture'] not in ['group6','plain12_C4pooled']:continue
            arch=fit['architecture'];route='group6_single' if arch=='group6' else 'plain12_C4'
            rows=[r for r in old['rows'] if (r['architecture'],r['budget'],r['seed'],r['route'])==(arch,27,fit['seed'],route)]
            assert len(rows)==1
            final,initial=folder/fit['model_file'],folder/fit['initial_model_file']
            assert sha(final)==fit['model_sha256'] and sha(initial)==fit['initial_model_sha256']
            result.append(dict(architecture=arch,route=route,budget=27,seed=fit['seed'],row=rows[0],fit=fit,
                folder=folder,model=final,initial=initial,
                probability=folder/'val_probs'/f'{route}_b27_s{fit["seed"]}.npz'))
    assert len(result)==6 and len({(x['architecture'],x['seed']) for x in result})==6
    return result


def matching_old(architecture,seed):
    values=[x for x in old_rows() if x['architecture']==architecture and x['seed']==seed]
    assert len(values)==1
    return values[0]


def group_initial(seed):
    old=matching_old('group6',seed)
    payload=torch.load(old['initial'],map_location='cpu',weights_only=True)
    group=model.GroupCNN();group.load_state_dict(payload['model']);group.eval()
    assert model.state_digest(group.state_dict())==old['fit']['initial_state_sha256']
    return group,old


def initial_control(seed):
    group,old=group_initial(seed)
    control=group.deploy().requires_grad_(True)
    assert model.parameters(control)==10681 and all(p.requires_grad for p in control.parameters())
    return control,group,old


def tie_residual(state):
    """Maximum deviation from the original C4/shared-head linear constraints."""
    errors=[]
    lift=state['0.weight'].reshape(6,4,1,3,3)
    for r in range(1,4):errors.append((torch.rot90(lift[:,r],-r,(-2,-1))-lift[:,0]).abs().max())
    for layer in ['0','2','4']:
        bias=state[layer+'.bias'].reshape(6,4)
        errors.append((bias-bias[:,:1]).abs().max())
    for layer in ['2','4']:
        weight=state[layer+'.weight'].reshape(6,4,6,4,3,3)
        reference=weight[:,0]
        for r in range(1,4):
            canonical=torch.stack([torch.rot90(weight[:,r,:, (q+r)%4],-r,(-2,-1)) for q in range(4)],dim=2)
            errors.append((canonical-reference).abs().max())
    head=state['6.weight'].reshape(1,6,4,1,1)
    errors.append((head-head[:,:,:1]).abs().max())
    return max(float(x.detach().cpu()) for x in errors)


def protected_inputs():
    paths=[ROOT/name for name in ['common.py','runner.py','CONFIG.json','PROTOCOL.md']]
    paths += [STAGE/name for name in ['model.py','runner.py','recovery.py','CONFIG.json','PROTOCOL.md']]
    paths += [PILOT/'TRAIN_VAL.npz',PILOT/'SUBSETS.json',AGG]
    paths += [STAGE/name for name in historical.source_hashes()]
    for old in old_rows():
        paths += [old['model'],old['initial'],old['probability']]
        paths += [old['folder']/name for name in ['RESULT.json','AUDIT.json','FROZEN.json','CONFIG_SNAPSHOT.json','PROTOCOL.md']]
        paths += [old['folder']/p['file'] for p in old['fit']['gradient_probes']]
        if old['architecture']=='group6':paths.append(old['model'].with_name(old['model'].stem+'_deployed.pt'))
    return {relative(path):dict(sha256=sha(path),bytes=path.stat().st_size) for path in sorted(set(p.resolve() for p in paths))}


def verify_lock():
    lock=read(ROOT/'LOCK.json')
    assert lock['inputs']==protected_inputs() and lock['preflight_sha256']==sha(ROOT/'PREFLIGHT.json')
    assert read(ROOT/'PREFLIGHT.json')['status']=='PASS_UNTIED_CPU_PREFLIGHT'
    return lock


def data(device):
    settings=read(STAGE/'CONFIG.json');settings['budgets']=[27]
    return historical.inputs(settings,device)
