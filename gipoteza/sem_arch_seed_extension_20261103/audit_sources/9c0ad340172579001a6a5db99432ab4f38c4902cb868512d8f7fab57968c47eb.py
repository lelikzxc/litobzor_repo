"""Independent CPU audit of two additional paired training seeds.

No optimizer steps, downloads, or MPS calls. Numerical references are separate
from the producer's metrics/bootstrap implementation. The 24-frame test has
already been observed; this audit does not turn replication into a new cohort.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
G, REPO = ROOT.parent, ROOT.parents[1]
OLD = G / 'sem_arch_calibrated_confirmation_20261103'
DEV = G / 'sem_arch_calibration_dev_20261103'
STAGE = G / 'sem_arch_groupcnn_20261102'
PILOT = G / 'sem_architecture_pilot_20261102'
NEW_SEEDS = [5801, 5821]
SEEDS = [5701, 5721, 5757] + NEW_SEEDS
ARMS = ['group6', 'untied24_single']
THRESHOLDS = np.array([round(.1 + .01*i, 2) for i in range(81)])


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rel(path): return str(Path(path).resolve().relative_to(REPO))
def read(path): return json.loads(Path(path).read_text())
def load(path): return torch.load(path, map_location='cpu', weights_only=True)
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


ref = module('seed_extension_independent_historical_reference', STAGE / 'audit.py')


def save(name, checks, details, paths):
    assert all(checks.values()), {k:v for k,v in checks.items() if not v}
    snapshot = ROOT / 'audit_sources'; snapshot.mkdir(exist_ok=True)
    source = Path(__file__).resolve(); source_hash = sha(source)
    target = snapshot / (source_hash + '.py')
    if not target.exists(): shutil.copy2(source, target)
    value = dict(status='PASS_SEED_EXTENSION_' + ('PRE_AUDIT' if name == 'PRE_AUDIT.json' else 'AUDIT'),
        checks=checks, checks_passed=len(checks), checks_total=len(checks), details=details,
        input_sha256={rel(p):sha(p) for p in sorted(set(Path(p).resolve() for p in paths))},
        auditor_sha256=source_hash, optimizer_updates=0, MPS_calls=0,
        interpretation='Initialization replication on an already seen 24-frame test, not a new cohort')
    path=ROOT/name; assert not path.exists(), 'Audits are immutable'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(status=value['status'],checks_passed=len(checks),checks_total=len(checks))),flush=True)
    return value


def macro(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = matrix.sum(-1) + matrix.sum(-2)
    return np.divide(2*np.diagonal(matrix,axis1=-2,axis2=-1), denominator,
        out=np.zeros_like(denominator), where=denominator>0).mean(-1)


def cm(truth, prediction):
    return np.bincount(9*np.asarray(truth,dtype=np.int64)+np.asarray(prediction,dtype=np.int64),minlength=81).reshape(9,9)


def rank_ap(truth, scores):
    truth, scores = np.asarray(truth,bool).ravel(), np.asarray(scores).ravel()
    positive=int(truth.sum())
    if not positive:return 0.
    order=np.argsort(-scores,kind='stable'); ranked=scores[order]; labels=truth[order]
    ends=np.flatnonzero(np.r_[ranked[1:]!=ranked[:-1],True]); tp=np.cumsum(labels,dtype=np.int64)[ends]
    return float(np.dot(np.diff(np.r_[0,tp])/positive,tp/(ends+1)))


def base():
    c=module('seed_extension_subject_common_audited', ROOT/'common.py')
    config=c.configuration(); lock=read(ROOT/'LOCK.json')
    checks={};details={}
    checks['fixed_two_new_seeds_four_fits_and_five_total']=(config['new_seeds']==NEW_SEEDS and config['old_seeds']==SEEDS[:3]
        and config['architectures']==ARMS and config['budget']==27)
    checks['all_lock_source_and_data_bindings_unchanged']=all(sha(REPO/p)==h for p,h in lock['input_sha256'].items())
    checks['preflight_bound_and_all_checks_pass']=lock['preflight_sha256']==sha(ROOT/'PREFLIGHT.json') and all(read(ROOT/'PREFLIGHT.json')['checks'].values())
    old_audit=read(OLD/'AUDIT.json')
    checks['historical_closed_result_and_all_audited_inputs_unchanged']=all(old_audit['checks'].values()) and all(
        sha(REPO/p)==h for p,h in old_audit['input_sha256'].items())
    checks['unchanged_optimizer_phase_and_sampling_configuration']=(config['steps']==2000 and config['warmup_steps']==400
        and config['learning_rate']==.0005 and config['weight_decay']==.0001 and config['physics_weight']==1
        and config['batch_size']==8 and config['batch_seed_offset']==1000 and config['thresholds']==THRESHOLDS.tolist())
    values,lineage=c.data('cpu'); values=values[27]
    subsets=read(PILOT/'SUBSETS.json')
    with np.load(PILOT/'TRAIN_VAL.npz',allow_pickle=False) as z:
        dense=np.array(subsets['dense_indices']['27']); weak=np.flatnonzero(z['role']=='weak')
        checks['same27dense_and1152weak_class_only_with_exact_normalization']=(len(dense)==27 and len(weak)==1152
            and 'mask' not in values['weak'] and values['dense']['indices']==dense.tolist()
            and values['weak']['indices']==weak.tolist()
            and np.array_equal(values['dense']['x'].numpy(),5*(z['X'][dense]-.2))
            and np.array_equal(values['weak']['x'].numpy(),5*(z['X'][weak]-.2)))
    final=read(OLD/'FINAL_CALIBRATION.json')
    with np.load(DEV/'DATA.npz',allow_pickle=False) as z:
        frames=z['frame']; hashes=z['raw_D4_sha256']; truth=z['label']
        chosen=sorted(np.unique(frames).tolist(),key=lambda f:hashlib.sha256(('SEM-C4-FINAL-CALIB-FRAME-20261103/'+str(f)).encode()).hexdigest())[:32]
        ids=[]
        for frame in chosen:
            candidates=np.flatnonzero(frames==frame).tolist()
            candidates.sort(key=lambda i:hashlib.sha256(('SEM-C4-CALIB-ROW-20261103/'+str(hashes[i])).encode()).hexdigest())
            ids.extend(candidates[:16])
        ids=np.array(sorted(ids),np.int64)
    checks['same512_calibration_indices_from_independent_SHA_rule']=(len(ids)==len(set(ids.tolist()))==512
        and final['data_indices']==ids.tolist() and final['frame_ids']==chosen)
    checks['class_only_threshold_tie_rule_integer_hundredths']=all(max(range(81),key=lambda i:(float(scores[i]),-abs(i-40),-i))==expected
        for scores,expected in [(np.ones(81),40),(np.where(np.isin(np.arange(81),[20,60]),1.,0.),20),
        (np.where(np.isin(np.arange(81),[39,42]),1.,0.),39)])
    pair_errors=[]
    fixture=torch.linspace(-1,1,7*144).reshape(7,1,12,12)
    for seed in NEW_SEEDS:
        group,untied=c.pair(seed); state=group.state_dict(); expected=ref.reference_deploy_state(state)
        exact=all(torch.equal(v,untied.state_dict()[k]) for k,v in expected.items())
        with torch.no_grad():err=float((ref.reference_group(fixture,state)-ref.reference_plain(fixture,untied.state_dict())).abs().max())
        checks[f's{seed}_independent_deployment_initial_same_function_and_parameter_counts']=(exact and err<=1e-6
            and sum(v.numel() for v in state.values())==2671 and sum(v.numel() for v in untied.state_dict().values())==10681
            and all(p.requires_grad for p in untied.parameters()))
        pair_errors.append(dict(seed=seed,independent_initial_logits_error=err))
    details.update(initial_function=pair_errors,calibration_indices=ids.tolist(),calibration_class_counts=np.bincount(truth[ids],minlength=9).tolist())
    paths=[REPO/p for p in lock['input_sha256']]+[ROOT/'LOCK.json',ROOT/'PREFLIGHT.json']
    return c,config,checks,details,paths


def training(c, checks, details, paths):
    result=read(ROOT/'TRAINING_RESULT.json'); fits=result['fits']
    assert len(fits)==4 and {(r['architecture'],r['seed']) for r in fits}=={(a,s) for a in ARMS for s in NEW_SEEDS}
    checks['four_fits_exactly8000_updates_with_no_seed_selection']=(result['new_fits']==4 and result['optimizer_updates']==8000
        and all(r['optimizer_updates']==2000 and r['warmup_steps']==400 and r['joint_steps']==1600 for r in fits))
    paths += [ROOT/'TRAINING_RESULT.json',ROOT/'FIT_RESULTS.json',ROOT/'MPS_PREFLIGHT.json']
    probes=[]
    for fit in fits:
        arch,seed=fit['architecture'],fit['seed']
        initial_file,final_file=REPO/fit['initial_file'],REPO/fit['model_file']
        initial,final=load(initial_file)['model'],load(final_file)['model']
        group,untied=c.pair(seed)
        expected=group.state_dict() if arch=='group6' else ref.reference_deploy_state(group.state_dict())
        batch_hash,batches=ref.expected_training_batches(27,seed)
        checks[f'{arch}_{seed}_fresh_initializer_and_saved_hashes_match']=(sha(initial_file)==fit['initial_sha256']
            and sha(final_file)==fit['model_sha256'] and all(torch.equal(v,initial[k]) for k,v in expected.items())
            and ref.digest(initial)==fit['initial_state_sha256'] and ref.digest(final)==fit['final_state_sha256']
            and ref.digest(initial)!=ref.digest(final))
        checks[f'{arch}_{seed}_sampler_sequence_and_fit_parameters_match']=(fit['batch_sequence_sha256']==batch_hash
            and fit['free_parameters']==(2671 if arch=='group6' else 10681) and fit['MACs_per_image']==1527552
            and fit['weak_masks_loaded'] is False and len(fit['gradient_probes'])==2)
        prediction_file=REPO/fit['prediction_model_file'];prediction=load(prediction_file)['model']
        deployed=ref.reference_deploy_state(final) if arch=='group6' else final
        checks[f'{arch}_{seed}_final_deployed_state_exact']=sha(prediction_file)==fit['prediction_model_sha256'] and all(
            torch.equal(v,prediction[k]) for k,v in deployed.items())
        if arch=='untied24_single':
            checks[f'{arch}_{seed}_independent_weights_left_tied_subspace']=c.legacy.tie_residual(final)>1e-8
        paths += [initial_file,final_file,prediction_file]
        for entry in fit['gradient_probes']:
            path=REPO/entry['file'];probe=load(path);step=entry['step'];joint=step==401
            assert step in [1,401] and sha(path)==entry['sha256'];paths.append(path)
            checks[f'{arch}_{seed}_{step}_saved_dense_then_weak_draws_and_phase']=(
                np.array_equal(probe['dense_indices'].numpy(),batches[step]['dense'])
                and np.array_equal(probe['weak_indices'].numpy(),batches[step]['weak'])
                and probe['weak_loss_used']==joint and probe['optimizer_step_not_yet_applied'])
            states={k:v.detach().clone().requires_grad_() for k,v in probe['state_before'].items()}
            dx,wx=probe['dense_x'],probe['weak_x'];inputs=torch.cat((dx,wx)) if joint else dx
            logits=(ref.reference_group if arch=='group6' else ref.reference_plain)(inputs,states)
            dense=logits[:len(dx)];weak=logits[len(dx):] if joint else None
            bce=F.binary_cross_entropy_with_logits(dense,probe['dense_mask'])
            geometry=None
            if joint:
                labels=probe['weak_label'];geometry=c.historical.physics.geometry_loss(weak,torch.stack((labels//3,labels%3),1))
            loss=bce if geometry is None else bce+geometry;loss.backward()
            logit_error=float((dense.detach()-probe['dense_logits']).abs().max())
            gradient_error=max(float((states[k].grad-probe['gradients'][k]).abs().max()) for k in states)
            gradient_relative=max(float((states[k].grad-probe['gradients'][k]).norm()/(states[k].grad.norm()+1e-10)) for k in states)
            checks[f'{arch}_{seed}_{step}_CPU_reference_forward_backward_agree']=(logit_error<=2e-4
                and gradient_error<=2e-4 and gradient_relative<=.01
                and abs(float(loss.detach())-probe['total_loss'])<=2e-4
                and all(torch.isfinite(states[k].grad).all() for k in states))
            dense_gradient=(probe['dense_logits'].sigmoid()-probe['dense_mask'])/probe['dense_mask'].numel()
            checks[f'{arch}_{seed}_{step}_exact_dense_BCE_gradient_and_active_weak_gradient']=(
                float((dense_gradient-probe['dense_logits_gradient']).abs().max())<=1e-7
                and (probe['weak_logits_gradient'] is None if not joint else
                    bool(torch.isfinite(probe['weak_logits_gradient']).all() and probe['weak_logits_gradient'].abs().sum()>0)))
            probes.append(dict(architecture=arch,seed=seed,step=step,CPU_logit_max_error=logit_error,
                CPU_parameter_gradient_max_error=gradient_error,CPU_gradient_max_relative_norm=gradient_relative))
    details['gradient_probe_CPU_replay']=probes
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pre',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        c,config,checks,details,paths=base()
        if args.pre:save('PRE_AUDIT.json',checks,details,paths)
        else:raise RuntimeError('Final numerical audit is added after producer artifact schema is fixed')


if __name__=='__main__':main()
