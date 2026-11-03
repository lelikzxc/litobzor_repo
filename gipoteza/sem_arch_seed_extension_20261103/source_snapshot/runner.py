"""Four paired fresh fits, fixed before optimization; no test inference here."""
import argparse
import datetime
import hashlib
import json
import shutil
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import common as c

def scalar(x): return float(x.detach().cpu())

def preflight(device):
    settings = c.configuration(); data, lineage = c.data(device); values = data[27]
    if device == 'mps': assert torch.backends.mps.is_available()
    torch.manual_seed(202611035); fixture = torch.randn(5,1,12,12, device=device)
    checks, rows = {}, []
    for seed in c.NEW_SEEDS:
        group, untied = c.pair(seed)
        assert c.model.state_digest(untied.state_dict()) == c.model.state_digest(group.deploy().state_dict())
        checks[f'{seed}_exact_initial_constraints'] = c.legacy.tie_residual(untied.state_dict()) == 0.
        for net in [group,untied]: net.to(device)
        with torch.no_grad(): error = scalar((group(fixture)-untied(fixture)).abs().max())
        assert error <= 1e-6
        checks[f'{seed}_same_initial_function'] = True
        for arch, network in zip(c.ARCHITECTURES,[group,untied]):
            initial = c.model.state_digest(network.state_dict())
            for joint in [False,True]:
                network.zero_grad(set_to_none=True)
                total, dense, weak, bce, geometry = c.objective(network,
                    values['dense']['x'][:2], values['weak']['x'][:2], values['dense']['mask'][:2], values['weak']['label'][:2], joint)
                dense.retain_grad()
                if weak is not None: weak.retain_grad()
                total.backward()
                assert torch.isfinite(total) and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in network.parameters())
                assert dense.grad.abs().sum()>0 and (weak is None or weak.grad.abs().sum()>0)
                assert c.model.state_digest(network.state_dict()) == initial
                residual = c.legacy.tie_residual({k:p.grad for k,p in network.named_parameters()}) if arch=='untied24_single' else None
                if residual is not None: assert residual>1e-8
                checks[f'{seed}_{arch}_{joint}_finite_gradients_unchanged_state'] = True
                rows.append(dict(seed=seed,architecture=arch,joint=joint,logit_error=error,
                    BCE=scalar(bce),geometry=None if geometry is None else scalar(geometry),untied_gradient_tie_residual=residual))
        rng = np.random.default_rng(seed+1000); sampler = hashlib.sha256()
        for _ in range(2000):
            for length in [27,1152]: sampler.update(rng.integers(0,length,8).astype(np.int64).tobytes())
        rows.append(dict(seed=seed,expected_batch_sequence_sha256=sampler.hexdigest()))
    assert all(checks.values())
    result=dict(status='PASS_SEED_REPLICATION_PREFLIGHT',device=device,checks=checks,rows=rows,
        checks_passed=len(checks),checks_total=len(checks),lineage=lineage,input_sha256=c.protected_inputs(),optimizer_updates=0)
    c.save(c.ROOT/('PREFLIGHT.json' if device=='cpu' else 'MPS_PREFLIGHT.json'),result)
    print(json.dumps(dict(status=result['status'],device=device,checks_passed=len(checks),checks_total=len(checks))),flush=True)

def freeze():
    assert not (c.ROOT/'LOCK.json').exists()
    pre=c.read(c.ROOT/'PREFLIGHT.json')
    assert pre['status']=='PASS_SEED_REPLICATION_PREFLIGHT' and pre['input_sha256']==c.protected_inputs()
    source=c.ROOT/'source_snapshot';source.mkdir(exist_ok=False)
    for name in ['common.py','runner.py','evaluate.py','report.py','CONFIG.json','PROTOCOL.md']: shutil.copy2(c.ROOT/name,source/name)
    c.save(c.ROOT/'LOCK.json',dict(status='FROZEN_SEED_REPLICATION_BEFORE_TRAINING',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),input_sha256=c.protected_inputs(),
        preflight_sha256=c.sha(c.ROOT/'PREFLIGHT.json'),new_seeds=c.NEW_SEEDS,old_seeds=c.OLD_SEEDS,
        new_fits=4,optimizer_updates=8000,new_independent_cohort=False))

def progress(**fields):
    c.save(c.ROOT/'PROGRESS.json',fields)
    with (c.ROOT/'progress.jsonl').open('a') as handle:handle.write(json.dumps(fields)+'\n')
    print(json.dumps(fields),flush=True)

def fit(net,arch,seed,values,index,started):
    cfg=c.configuration();net.train();optimizer=torch.optim.AdamW(net.parameters(),lr=.0005,weight_decay=.0001)
    initial=c.model.state_digest(net.state_dict());rng=np.random.default_rng(seed+1000);batch_digest=hashlib.sha256();probes=[]
    start=time.monotonic()
    for step in range(1,2001):
        ai=rng.integers(0,27,8);wi=rng.integers(0,1152,8)
        batch_digest.update(ai.astype(np.int64).tobytes());batch_digest.update(wi.astype(np.int64).tobytes())
        ai,wi=torch.tensor(ai,device='mps'),torch.tensor(wi,device='mps')
        dx,wx=values['dense']['x'][ai],values['weak']['x'][wi]
        target,labels=values['dense']['mask'][ai],values['weak']['label'][wi]
        optimizer.zero_grad(set_to_none=True)
        total,dense,weak,bce,geometry=c.objective(net,dx,wx,target,labels,step>400)
        if step in [1,401]:
            dense.retain_grad()
            if weak is not None:weak.retain_grad()
        assert torch.isfinite(total);total.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        if step in [1,401]:
            path=c.ROOT/'trial/gradient_probes'/f'{arch}_b27_s{seed}_step{step}.pt'
            torch.save(dict(architecture=arch,budget=27,seed=seed,step=step,
                state_before={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                gradients={k:p.grad.detach().cpu().clone() for k,p in net.named_parameters()},
                dense_indices=ai.cpu(),weak_indices=wi.cpu(),dense_x=dx.cpu(),weak_x=wx.cpu(),
                dense_mask=target.cpu(),weak_label=labels.cpu(),dense_logits=dense.detach().cpu(),
                weak_logits=None if weak is None else weak.detach().cpu(),dense_logits_gradient=dense.grad.detach().cpu(),
                weak_logits_gradient=None if weak is None else weak.grad.detach().cpu(),
                BCE=scalar(bce),geometry_loss=None if geometry is None else scalar(geometry),
                total_loss=scalar(total),optimizer_step_not_yet_applied=True,weak_loss_used=step>400),path)
            probes.append(dict(file=c.relative(path),sha256=c.sha(path),step=step))
        optimizer.step()
        if step%200==0:progress(status='TRAINING',architecture=arch,seed=seed,fit_index=index,target_fits=4,
            step=step,target_steps=2000,completed_updates=(index-1)*2000+step,total_updates=8000,
            BCE=scalar(bce),geometry_loss=None if geometry is None else scalar(geometry),seconds=time.monotonic()-started)
    torch.mps.synchronize();final=c.model.state_digest(net.state_dict())
    expected=next(row['expected_batch_sequence_sha256'] for row in c.read(c.ROOT/'PREFLIGHT.json')['rows'] if row.get('seed')==seed and 'expected_batch_sequence_sha256' in row)
    assert batch_digest.hexdigest()==expected and final!=initial
    return dict(architecture=arch,seed=seed,budget=27,optimizer_updates=2000,warmup_steps=400,joint_steps=1600,
        initial_state_sha256=initial,final_state_sha256=final,batch_sequence_sha256=expected,
        free_parameters=c.model.parameters(net),MACs_per_image=1527552,weak_masks_loaded=False,
        training_seconds=time.monotonic()-start,gradient_probes=probes,
        final_tie_residual=c.legacy.tie_residual(net.state_dict()) if arch=='untied24_single' else None)

def run():
    c.verify_lock();assert torch.backends.mps.is_available() and c.os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
    assert c.read(c.ROOT/'MPS_PREFLIGHT.json')['status']=='PASS_SEED_REPLICATION_PREFLIGHT'
    preaudit=c.read(c.ROOT/'PRE_AUDIT.json');assert preaudit['status'].startswith('PASS') and all(preaudit['checks'].values())
    trial=c.ROOT/'trial';trial.mkdir(exist_ok=False)
    for name in ['models','gradient_probes']:(trial/name).mkdir()
    values=c.data('mps')[0][27];fits=[];started=time.monotonic()
    try:
        for seed in c.NEW_SEEDS:
            group,untied=c.pair(seed)
            initial_group_digest=c.model.state_digest(group.state_dict())
            for arch,net in zip(c.ARCHITECTURES,[group,untied]):
                index=len(fits)+1;initial=trial/'models'/f'{arch}_b27_s{seed}_initial.pt';final=trial/'models'/f'{arch}_b27_s{seed}.pt'
                torch.save(dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},architecture=arch,seed=seed,
                    group_initial_state_sha256=initial_group_digest),initial)
                fitting=fit(net.to('mps'),arch,seed,values,index,started)
                torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},architecture=arch,seed=seed,settings=fitting),final)
                row=dict(fitting,initial_file=c.relative(initial),initial_sha256=c.sha(initial),model_file=c.relative(final),model_sha256=c.sha(final))
                if arch=='group6':
                    deployed=trial/'models'/f'{arch}_b27_s{seed}_deployed.pt'
                    torch.save(dict(model={k:v.detach().cpu() for k,v in net.deploy().state_dict().items()},free_model_sha256=c.sha(final)),deployed)
                    row.update(prediction_model_file=c.relative(deployed),prediction_model_sha256=c.sha(deployed))
                else:row.update(prediction_model_file=c.relative(final),prediction_model_sha256=c.sha(final))
                fits.append(row);c.save(c.ROOT/'FIT_RESULTS.json',fits)
                progress(status='FIT_COMPLETE',architecture=arch,seed=seed,completed_fits=len(fits),target_fits=4,
                    completed_updates=2000*len(fits),total_updates=8000,seconds=time.monotonic()-started)
                del net
        c.verify_lock();assert len(fits)==4
        c.save(c.ROOT/'TRAINING_RESULT.json',dict(status='COMPLETE_FOUR_SEED_REPLICATION_FITS',fits=fits,
            new_fits=4,optimizer_updates=8000,lock_sha256=c.sha(c.ROOT/'LOCK.json'),seconds=time.monotonic()-started))
    except BaseException as exc:
        c.save(c.ROOT/'TRAINING_FAILURE.json',dict(error=str(exc),type=type(exc).__name__,traceback=traceback.format_exc(),completed_fits=len(fits)));raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--preflight',choices=['cpu','mps']);p.add_argument('--freeze',action='store_true');p.add_argument('--run',action='store_true');a=p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):
        if a.preflight:preflight(a.preflight)
        elif a.freeze:freeze()
        elif a.run:run()
        else:p.error('select --preflight/--freeze/--run')
if __name__=='__main__':main()
