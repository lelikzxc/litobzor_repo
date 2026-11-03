"""Three fresh untied single-pass controls from archived GROUP INITIAL states."""
import argparse
import datetime
import hashlib
import json
import shutil
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
import common as c


def scalar(value):return float(value.detach().cpu())


def objective(net,dx,wx,target,labels,joint):
    logits=net(torch.cat((dx,wx)) if joint else dx)
    dense=logits[:len(dx)];weak=logits[len(dx):] if joint else None
    bce=F.binary_cross_entropy_with_logits(dense,target)
    geometry=None if not joint else c.historical.physics.geometry_loss(weak,torch.stack((labels//3,labels%3),dim=1))
    return bce if geometry is None else bce+geometry,dense,weak,bce,geometry


def preflight(settings):
    data,lineage=c.data('cpu');values=data[27];checks={};details=[]
    torch.manual_seed(5913);fixture=torch.randn(5,1,12,12)
    for seed in c.SEEDS:
        net,group,old=c.initial_control(seed)
        expanded=group.deploy()
        assert c.model.state_digest(net.state_dict())==c.model.state_digest(expanded.state_dict())
        with torch.no_grad():
            logits_error=float((net(fixture)-group(fixture)).abs().max())
            probability_error=float((net(fixture).sigmoid()-group(fixture).sigmoid()).abs().max())
        assert logits_error<=1e-6 and probability_error<=1e-6 and c.tie_residual(net.state_dict())==0.
        final_group=c.model.GroupCNN();final_group.load_state_dict(torch.load(old['model'],map_location='cpu',weights_only=True)['model'])
        deployed=torch.load(old['model'].with_name(old['model'].stem+'_deployed.pt'),map_location='cpu',weights_only=True)
        assert c.model.state_digest(final_group.deploy().state_dict())==c.model.state_digest(deployed['model'])
        assert deployed['free_model_sha256']==c.sha(old['model'])
        checks[f's{seed}_archived_INITIAL_expansion_and_same_function']=True
        checks[f's{seed}_all10681_parameters_trainable_initial_ties_exact']=True
        checks[f's{seed}_historical_final_deployment_is_not_initializer']=True
        for joint in [False,True]:
            net,_,_=c.initial_control(seed)
            dx,wx=values['dense']['x'][:2],values['weak']['x'][:2]
            target=values['dense']['mask'][:2];labels=values['weak']['label'][:2]
            loss,dense,weak,bce,geometry=objective(net,dx,wx,target,labels,joint)
            dense.retain_grad()
            if weak is not None:weak.retain_grad()
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in net.parameters())
            assert dense.grad.abs().sum()>0 and (weak is None or weak.grad.abs().sum()>0)
            residual=c.tie_residual({k:p.grad for k,p in net.named_parameters()})
            assert residual>1e-8,'Independent gradients must leave tied-weight subspace'
            checks[f's{seed}_phase{401 if joint else 1}_finite_all_gradients_and_free_tie_gradient']=True
            details.append(dict(seed=seed,joint=joint,logit_error=logits_error,probability_error=probability_error,
                BCE=scalar(bce),geometry_loss=None if geometry is None else scalar(geometry),gradient_tie_residual=residual))
    protected=c.protected_inputs()
    result=dict(status='PASS_UNTIED_CPU_PREFLIGHT',checks=checks,checks_passed=len(checks),checks_total=len(checks),
        details=details,inputs_sha256=c.historical.digest(protected),lineage=lineage,optimizer_updates=0,
        new_fits_planned=3,reused_fits=6,fresh_test_access=False)
    c.save(c.ROOT/'PREFLIGHT.json',result)
    print(json.dumps(dict(status=result['status'],checks_passed=len(checks),checks_total=len(checks),optimizer_updates=0)),flush=True)


def freeze():
    assert not (c.ROOT/'LOCK.json').exists()
    pre=c.read(c.ROOT/'PREFLIGHT.json');protected=c.protected_inputs()
    assert pre['status']=='PASS_UNTIED_CPU_PREFLIGHT' and pre['inputs_sha256']==c.historical.digest(protected)
    snapshot=c.ROOT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in ['common.py','runner.py','CONFIG.json','PROTOCOL.md']:shutil.copy2(c.ROOT/name,snapshot/name)
    c.save(c.ROOT/'LOCK.json',dict(scope='OLD_TRAIN_VAL_SAME_WIDTH_SAME_MAC_UNTIED_CONTROL',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=protected,
        preflight_sha256=c.sha(c.ROOT/'PREFLIGHT.json'),new_fits=3,reused_fits=6,
        optimizer_updates_planned=6000,fresh_test_access=False))
    print(json.dumps(dict(status='FROZEN',inputs=len(protected),lock_sha256=c.sha(c.ROOT/'LOCK.json'))),flush=True)


def progress(trial,**fields):
    c.save(trial/'PROGRESS.json',fields);c.save(c.ROOT/'PROGRESS.json',fields)
    with (trial/'progress.jsonl').open('a') as handle:handle.write(json.dumps(fields)+'\n')
    print(json.dumps(fields),flush=True)


def fit(settings,net,values,seed,trial,index,start):
    net.train();old=c.matching_old('group6',seed);initial=c.model.state_digest(net.state_dict())
    optimizer=torch.optim.AdamW(net.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    rng=np.random.default_rng(seed+settings['batch_seed_offset']);batches=hashlib.sha256();probes=[];started=time.monotonic()
    for step in range(1,settings['steps']+1):
        ai=rng.integers(0,len(values['dense']['x']),settings['batch_size']);wi=rng.integers(0,len(values['weak']['x']),settings['batch_size'])
        batches.update(ai.astype(np.int64).tobytes());batches.update(wi.astype(np.int64).tobytes())
        ai=torch.tensor(ai,device='mps');wi=torch.tensor(wi,device='mps')
        dx,wx=values['dense']['x'][ai],values['weak']['x'][wi]
        target,labels=values['dense']['mask'][ai],values['weak']['label'][wi]
        optimizer.zero_grad(set_to_none=True);joint=step>settings['warmup_steps']
        loss,dense,weak,bce,geometry=objective(net,dx,wx,target,labels,joint)
        if step in [1,401]:
            dense.retain_grad()
            if weak is not None:weak.retain_grad()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in net.parameters())
        if step in [1,401]:
            probe=dict(architecture='untied24_single',budget=27,seed=seed,step=step,
                state_before={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
                gradients={k:p.grad.detach().cpu().clone() for k,p in net.named_parameters()},
                dense_indices=ai.cpu(),weak_indices=wi.cpu(),dense_x=dx.cpu(),weak_x=wx.cpu(),
                dense_mask=target.cpu(),weak_label=labels.cpu(),dense_logits=dense.detach().cpu(),
                weak_logits=None if weak is None else weak.detach().cpu(),dense_logits_gradient=dense.grad.detach().cpu(),
                weak_logits_gradient=None if weak is None else weak.grad.detach().cpu(),
                BCE=scalar(bce),geometry_loss=None if geometry is None else scalar(geometry),total_loss=scalar(loss),
                weak_loss_used=joint,optimizer_step_not_yet_applied=True)
            path=trial/'gradient_probes'/f'untied24_single_b27_s{seed}_step{step}.pt';torch.save(probe,path)
            probes.append(dict(file=str(path.relative_to(trial)),sha256=c.sha(path),step=step))
        optimizer.step()
        if step%200==0:progress(trial,status='TRAINING',architecture='untied24_single',budget=27,seed=seed,
            fit_index=index,target_fits=3,step=step,target_steps=2000,completed_updates=(index-1)*2000+step,total_updates=6000,
            BCE=scalar(bce),geometry_loss=None if geometry is None else scalar(geometry),seconds=time.monotonic()-start)
    torch.mps.synchronize();final=c.model.state_digest(net.state_dict());assert final!=initial
    batch_digest=batches.hexdigest();assert batch_digest==old['fit']['batch_sequence_sha256']
    residual=c.tie_residual(net.state_dict());assert residual>1e-8
    return dict(steps=2000,optimizer_updates=2000,warmup_steps=400,joint_steps=1600,fit_count=1,
        initial_state_sha256=initial,final_state_sha256=final,batch_sequence_sha256=batch_digest,
        group_initial_file=c.relative(old['initial']),group_initial_sha256=c.sha(old['initial']),
        fresh_initialization=True,pretrained_weights_loaded=False,weak_masks_loaded=False,
        free_parameters=10681,MACs_per_image=1527552,training_route='single_untied24',
        final_tie_residual=residual,gradient_probes=probes,training_seconds=time.monotonic()-started)


def historical_rows():
    result=[]
    for old in c.old_rows():
        row=old['row'];fine=row['primary_fine'];fixed=next(x for x in fine['all_thresholds'] if x['threshold']==.5)
        result.append(dict(architecture=old['architecture'],route=old['route'],budget=27,seed=old['seed'],
            selected=row['selected'],all_thresholds=fine['all_thresholds'],primary_fine=fine,fixed05=fixed,
            model_file=c.relative(old['model']),model_sha256=c.sha(old['model']),
            probability_file=c.relative(old['probability']),probability_sha256=c.sha(old['probability']),
            free_parameters=c.PARAMETERS[old['architecture']],MACs_per_image=c.MACS[old['architecture']],
            origin='immutable_historical_fit',historical_trial=c.relative(old['folder'])))
    return result


def summarize(rows,mode):
    lookup={(x['architecture'],x['seed']):x[mode] for x in rows};assert len(lookup)==9
    means={arch:{m:float(np.mean([lookup[arch,s][m] for s in c.SEEDS])) for m in ['mask_F1','IoU']} for arch in c.ARMS}
    def difference(reference):
        per=[dict(seed=s,**{m:lookup['group6',s][m]-lookup[reference,s][m] for m in ['mask_F1','IoU']}) for s in c.SEEDS]
        return dict(mean={m:float(np.mean([x[m] for x in per])) for m in ['mask_F1','IoU']},per_seed=per)
    return dict(means=means,contrasts=dict(group_minus_untied=difference('untied24_single'),group_minus_narrow=difference('plain12_C4pooled')))


def gates(primary,fixed,settings):
    limits=settings['development_gate'];effect=primary['contrasts']['group_minus_untied']
    guards=dict(mean_F1=effect['mean']['mask_F1']>=limits['minimum_mean_F1_gain'],
        seed_F1=min(x['mask_F1'] for x in effect['per_seed'])>=limits['minimum_seed_F1_delta'],
        mean_IoU=effect['mean']['IoU']>=limits['minimum_mean_IoU_delta'],
        matched_MACs=abs(c.MACS['group6']/c.MACS['untied24_single']-1)<=limits['maximum_MAC_relative_difference'],
        initial_function_match=all(v['logit_error']<=settings['initial_function_absolute_tolerance'] and
            v['probability_error']<=settings['initial_function_absolute_tolerance']
            for v in c.read(c.ROOT/'PREFLIGHT.json')['details']))
    if 'minimum_fixed05_F1_delta' in limits:
        guards['fixed05_F1']=fixed['contrasts']['group_minus_untied']['mean']['mask_F1']>=limits['minimum_fixed05_F1_delta']
    return dict(passed=bool(all(guards.values())),guards={k:bool(v) for k,v in guards.items()},
        failed=[k for k,v in guards.items() if not v],interpretation='descriptive_reused_VAL_mechanism_gate_not_independent_confirmation')


def admission(outcome,fixed,settings):
    effect=fixed['contrasts']['group_minus_untied']['mean'];limits=settings['confirmation_admission']
    guards=dict(primary_gate=outcome['passed'],fixed05_F1=effect['mask_F1']>=limits['minimum_fixed05_F1_delta'],
        fixed05_IoU=effect['IoU']>=limits['minimum_fixed05_IoU_delta'])
    return dict(passed=bool(all(guards.values())),guards={k:bool(v) for k,v in guards.items()},
        failed=[k for k,v in guards.items() if not v],interpretation='Primary gate AND both fixed0.5 safeguards; no secondary rescue')


def run(settings):
    c.verify_lock();assert torch.backends.mps.is_available() and c.os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
    trial=c.ROOT/'trial';assert not trial.exists(),'Immutable trial; no overwrite or implicit resume'
    for name in ['','models','val_probs','gradient_probes']:(trial/name).mkdir(exist_ok=False)
    data,lineage=c.data('mps');values=data[27];start=time.monotonic();rows=historical_rows();fits=[]
    c.save(trial/'FROZEN.json',dict(lock_sha256=c.sha(c.ROOT/'LOCK.json'),lineage=lineage,new_fits=3,reused_fits=6,
        new_optimizer_updates=6000,fresh_test_access=False))
    c.save(trial/'PARTIAL_RESULTS.json',rows)
    try:
        for index,seed in enumerate(c.SEEDS,1):
            net,group,old=c.initial_control(seed);net=net.to('mps')
            initial=trial/'models'/f'untied24_single_b27_s{seed}_initial.pt';final=trial/'models'/f'untied24_single_b27_s{seed}.pt'
            torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},architecture='untied24_single',seed=seed,
                source_group_initial_file=c.relative(old['initial']),source_group_initial_sha256=c.sha(old['initial'])),initial)
            fitting=fit(settings,net,values,seed,trial,index,start)
            torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},settings=fitting,
                architecture='untied24_single',budget=27,seed=seed),final)
            fitted=dict(fitting,architecture='untied24_single',budget=27,seed=seed,
                model_file=c.relative(final),model_sha256=c.sha(final),initial_file=c.relative(initial),initial_sha256=c.sha(initial))
            fits.append(fitted);c.save(trial/'FIT_RESULTS.json',fits)
            net.eval()
            with torch.no_grad():probability=np.concatenate([net(x).sigmoid().cpu().numpy() for x in values['val']['x'].split(16)])
            evaluation,arrays=c.historical.score(probability,values['val'],settings['thresholds'])
            fixed=next(x for x in evaluation['all_thresholds'] if x['threshold']==.5)
            path=trial/'val_probs'/f'untied24_single_b27_s{seed}.npz'
            np.savez_compressed(path,probability=probability,truth=values['val']['label'].cpu().numpy(),
                target=values['val']['mask'][:,0].cpu().numpy().astype(bool),thresholds=np.array(settings['thresholds']),**arrays)
            row=dict(architecture='untied24_single',route='untied24_single',budget=27,seed=seed,
                selected=evaluation['selected'],all_thresholds=evaluation['all_thresholds'],primary_fine=evaluation,fixed05=fixed,
                free_parameters=10681,MACs_per_image=1527552,model_file=c.relative(final),model_sha256=c.sha(final),
                probability_file=c.relative(path),probability_sha256=c.sha(path),origin='new_untied_fit')
            rows.append(row);c.save(trial/'PARTIAL_RESULTS.json',rows)
            progress(trial,status='EVALUATED',architecture='untied24_single',seed=seed,fit_index=index,target_fits=3,
                completed_updates=index*2000,total_updates=6000,selected=row['selected'],seconds=time.monotonic()-start)
        assert len(rows)==9 and len(fits)==3;c.verify_lock()
        primary,fixed=summarize(rows,'selected'),summarize(rows,'fixed05');outcome=gates(primary,fixed,settings)
        result=dict(status='COMPLETE_UNTIED_CONTROL_DEVELOPMENT',rows=rows,fits=fits,primary=primary,fixed05_diagnostic=fixed,
            primary_development_gate=outcome,confirmation_admission=admission(outcome,fixed,settings),
            new_fits=3,reused_fits=6,new_optimizer_updates=6000,
            lock_sha256=c.sha(c.ROOT/'LOCK.json'),source_and_old_inputs_unchanged=True,fresh_test_access=False,
            seconds=time.monotonic()-start,
            caveats=['reused_adaptive_VAL_not_independent_confirmation','same_compute_different_free_parameter_count',
                'auxiliary_supervision_shared_by_all_arms','calibration_on_same_VAL_is_optimistic','no_new_operator_novelty_claim'])
        c.save(trial/'RESULT.json',result);c.save(c.ROOT/'RESULT.json',result)
        progress(trial,status=result['status'],fit_index=3,target_fits=3,completed_updates=6000,total_updates=6000,
            primary_development_gate=outcome,contrasts=primary['contrasts'],seconds=result['seconds'])
    except BaseException as error:
        c.save(trial/'FAILURE.json',dict(status='TECHNICAL_FAILURE',type=type(error).__name__,error=str(error),
            traceback=traceback.format_exc(),completed_fits=len(fits),preserve_all_artifacts=True));raise


def main():
    parser=argparse.ArgumentParser();action=parser.add_mutually_exclusive_group()
    action.add_argument('--preflight',action='store_true');action.add_argument('--freeze',action='store_true')
    args=parser.parse_args();settings=c.config();torch.set_num_threads(2)
    if args.preflight:preflight(settings)
    elif args.freeze:freeze()
    else:run(settings)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
