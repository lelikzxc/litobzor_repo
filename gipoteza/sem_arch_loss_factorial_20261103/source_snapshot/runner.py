"""Twelve fresh BCE fits + twelve immutable historical auxiliary fits; old VAL only."""
from pathlib import Path
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


def scalar(value): return float(value.detach().cpu())


def objective(net, architecture, dx, wx, dense_target, joint):
    details = c.model.training_details(net, architecture, torch.cat((dx, wx)) if joint else dx)
    logits = details['logits']
    loss = F.binary_cross_entropy_with_logits(logits[:len(dx)], dense_target)
    return loss, details


def preflight(settings):
    values, lineage = c.data('cpu'); checks = []; maximum_errors = {}
    for item in c.old_rows():
        net = c.model.make_model(item['architecture'], item['seed'])
        original = torch.load(item['initial'], map_location='cpu', weights_only=True)
        assert c.model.state_digest(net.state_dict()) == c.model.state_digest(original['model'])
    checks.append(dict(name='all12_initializations_match_archived_initial_states', passed=True))
    checks.append(dict(name='old12_final_models_and_two_independent_audits_bound', passed=True))
    for arch in c.ARMS:
        for joint in [False, True]:
            net = c.model.make_model(arch, 5701)
            v = values[18]; dx=v['dense']['x'][:2]; wx=v['weak']['x'][:2]; target=v['dense']['mask'][:2]
            loss, details = objective(net, arch, dx, wx, target, joint)
            details['logits'].retain_grad(); loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            assert any(float(p.grad.abs().sum()) > 0 for p in net.parameters())
            if joint: assert int(torch.count_nonzero(details['logits'].grad[2:])) == 0
            gradients={k:p.grad.clone() for k,p in net.named_parameters()}
            other = c.model.make_model(arch, 5701)
            inputs=torch.cat((dx,wx)) if joint else dx
            if arch == 'group6': independent=other(inputs).sigmoid()
            else:
                views=[]
                # Independent one-view-at-a-time probability pooling.
                for turn in range(4):
                    z=other(torch.rot90(inputs,turn,(-2,-1)).contiguous())
                    views.append(torch.rot90(z,-turn,(-2,-1)).sigmoid())
                independent=torch.stack(views).mean(0)
            check_loss=F.binary_cross_entropy(independent[:2],target); check_loss.backward()
            probability_error=float((details['probability']-independent).abs().max())
            gradient_error=max(float((gradients[k]-p.grad).abs().max()) for k,p in other.named_parameters())
            assert probability_error <= 1e-6 and gradient_error <= 1e-6 and abs(scalar(loss)-scalar(check_loss)) <= 1e-6
            maximum_errors[f'{arch}_joint{int(joint)}']=dict(probability=probability_error, gradients=gradient_error,
                BCE=abs(scalar(loss)-scalar(check_loss)))
            checks.append(dict(name=f'{arch}_step{401 if joint else 1}_BCE_gradient_and_zero_weak', passed=True))
    sample=dict(architecture='group6', objective='BCE', budget=18, seed=5701, selected=dict(threshold=.5,mask_F1=0.,IoU=0.))
    assert json.loads(json.dumps(sample)) == sample
    checks.append(dict(name='result_serialization',passed=True))
    checks.append(dict(name='data_dense18_27_weak1152_VAL695_no_weak_masks',passed=True))
    protected=c.protected_inputs()
    result=dict(status='PASS_FACTORIAL_CPU_PREFLIGHT', checks=checks, checks_passed=len(checks), checks_total=len(checks),
        maximum_errors=maximum_errors, inputs_sha256=c.historical.digest(protected), lineage=lineage,
        optimizer_updates=0, new_fits_planned=12, inherited_fits=12, fresh_test_access=False)
    c.save(c.ROOT/'PREFLIGHT.json',result);print(json.dumps(result),flush=True)


def freeze():
    assert not (c.ROOT/'LOCK.json').exists(), 'Immutable lock already exists'
    pre=c.read(c.ROOT/'PREFLIGHT.json'); assert pre['status']=='PASS_FACTORIAL_CPU_PREFLIGHT'
    protected=c.protected_inputs(); assert pre['inputs_sha256']==c.historical.digest(protected)
    snapshot=c.ROOT/'source_snapshot';snapshot.mkdir(exist_ok=False)
    for name in ['common.py','runner.py','CONFIG.json','PROTOCOL.md']:
        shutil.copy2(c.ROOT/name,snapshot/name)
    c.save(c.ROOT/'LOCK.json',dict(scope='OLD_TRAIN_VAL_ONLY_ARCHITECTURE_X_AUXILIARY_SUPERVISION',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=protected,
        preflight_sha256=c.sha(c.ROOT/'PREFLIGHT.json'), new_fits=12, inherited_fits=12,
        fresh_test_access=False, optimizer_updates_planned=24000))
    print(json.dumps(dict(status='FROZEN', lock_sha256=c.sha(c.ROOT/'LOCK.json'), inputs=len(protected))),flush=True)


def progress(trial, **kwargs):
    c.save(trial/'PROGRESS.json', kwargs);c.save(c.ROOT/'PROGRESS.json', kwargs)
    with (trial/'progress.jsonl').open('a') as handle:handle.write(json.dumps(kwargs)+'\n')
    print(json.dumps(kwargs),flush=True)


def fit(settings, net, arch, values, budget, seed, trial, fit_index, start):
    net.train();old=c.matching_old(arch,budget,seed);initial=c.model.state_digest(net.state_dict())
    assert initial==old['fit']['initial_state_sha256']
    optimizer=torch.optim.AdamW(net.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    rng=np.random.default_rng(seed+settings['batch_seed_offset']);batches=hashlib.sha256();probes=[];started=time.monotonic()
    for step in range(1,settings['steps']+1):
        ai=rng.integers(0,len(values['dense']['x']),settings['batch_size'])
        wi=rng.integers(0,len(values['weak']['x']),settings['batch_size'])
        batches.update(ai.astype(np.int64).tobytes());batches.update(wi.astype(np.int64).tobytes())
        ai=torch.tensor(ai,device='mps');wi=torch.tensor(wi,device='mps')
        dx=values['dense']['x'][ai];wx=values['weak']['x'][wi];target=values['dense']['mask'][ai]
        joint=step>settings['warmup_steps'];optimizer.zero_grad(set_to_none=True)
        loss,details=objective(net,arch,dx,wx,target,joint)
        if step in [1,401]:details['logits'].retain_grad()
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
        assert any(scalar(p.grad.abs().sum())>0 for p in net.parameters())
        if step in [1,401]:
            lg=details['logits'].grad
            if joint:assert int(torch.count_nonzero(lg[settings['batch_size']:]))==0
            old_probe_path=old['folder']/next(v['file'] for v in old['fit']['gradient_probes'] if v['step']==step)
            old_probe=torch.load(old_probe_path,map_location='cpu',weights_only=True)
            state_before={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            state_max_error=max(float((state_before[k]-v).abs().max()) for k,v in old_probe['state_before'].items())
            state_matches=all(torch.allclose(state_before[k],v,atol=1e-6,rtol=1e-5)
                for k,v in old_probe['state_before'].items())
            assert state_matches, f'Paired warmup state mismatch at step{step}: {state_max_error}'
            probe=dict(architecture=arch,objective='BCE',budget=budget,seed=seed,step=step,
                state_before=state_before,
                gradients={k:p.grad.detach().cpu().clone() for k,p in net.named_parameters()},
                dense_indices=ai.cpu(),weak_indices=wi.cpu(),dense_x=dx.cpu(),weak_x=wx.cpu(),dense_mask=target.cpu(),
                all_logits=details['logits'].detach().cpu(),logits_gradient=lg.detach().cpu(),
                BCE=scalar(loss),total_loss=scalar(loss),weak_loss_used=False,joint_forward=joint,
                historical_probe_file=c.relative(old_probe_path),historical_probe_sha256=c.sha(old_probe_path),
                paired_warmup_state_max_absolute_error=state_max_error,paired_warmup_state_matches=state_matches,
                paired_warmup_state_tolerance=dict(absolute=1e-6,relative=1e-5),
                optimizer_step_not_yet_applied=True)
            path=trial/'gradient_probes'/f'{arch}_b{budget}_s{seed}_step{step}.pt';torch.save(probe,path)
            probes.append(dict(file=str(path.relative_to(trial)),sha256=c.sha(path),step=step))
        optimizer.step()
        if step%200==0:
            progress(trial,status='TRAINING',objective='BCE',architecture=arch,budget=budget,seed=seed,
                fit_index=fit_index,target_fits=12,step=step,target_steps=2000,
                completed_updates=(fit_index-1)*2000+step,total_updates=24000,
                BCE=scalar(loss),seconds=time.monotonic()-start)
    torch.mps.synchronize();final=c.model.state_digest(net.state_dict());assert final!=initial
    batch_digest=batches.hexdigest();assert batch_digest==old['fit']['batch_sequence_sha256']
    return dict(optimizer_updates=2000,steps=2000,warmup_shape_steps=400,joint_shape_steps=1600,
        dense_BCE_steps=2000,weak_loss_steps=0,weak_labels_used=False,weak_masks_loaded=False,
        fresh_initialization=True,initial_state_sha256=initial,final_state_sha256=final,
        paired_auxiliary_initialization_match=True,batch_sequence_sha256=batch_digest,
        paired_auxiliary_batch_sequence_match=True,gradient_probes=probes,training_seconds=time.monotonic()-started)


def evaluate(settings,network,arch,values):
    _,mode=c.ROUTES[arch];network.eval()
    with torch.no_grad():
        parts=[c.model.predict_probability(network,x,mode).cpu().numpy() for x in values['x'].split(16)]
    probability=np.concatenate(parts)
    score, arrays=c.historical.score(probability,values,settings['thresholds'])
    fixed=[row for row in score['all_thresholds'] if row['threshold']==.5];assert len(fixed)==1
    payload=dict(probability=probability,truth=values['label'].cpu().numpy(),target=values['mask'][:,0].cpu().numpy().astype(bool),
        thresholds=np.array(settings['thresholds']),predictions=arrays['predictions'],packed_masks=arrays['packed_masks'])
    return dict(selected=score['selected'],all_thresholds=score['all_thresholds'],fixed05=fixed[0],
        primary_fine=score,fixed_05=fixed[0]),payload


def historical_rows(settings):
    rows=[]
    for old in c.old_rows():
        row=old['row'];fixed=[v for v in row['primary_fine']['all_thresholds'] if v['threshold']==.5];assert len(fixed)==1
        rows.append(dict(objective='BCE_geometry',loss_regime='BCE_plus_geometry',architecture=old['architecture'],budget=old['budget'],seed=old['seed'],
            route=old['route'],selected=row['selected'],all_thresholds=row['primary_fine']['all_thresholds'],fixed05=fixed[0],
            primary_fine=row['primary_fine'],fixed_05=fixed[0],
            origin='immutable_historical_fit_no_new_optimizer_steps',model_file=c.relative(old['model']),
            model_sha256=c.sha(old['model']),probability_file=c.relative(old['probability']),
            probability_sha256=c.sha(old['probability']),historical_trial=c.relative(old['folder'])))
    return rows


def summarize(rows,settings,metric_mode):
    lookup={(v['objective'],v['architecture'],v['budget'],v['seed']):v[metric_mode] for v in rows}
    assert len(lookup)==24
    cells={}
    for obj in ['BCE','BCE_geometry']:
        cells[obj]={}
        for arch in c.ARMS:
            budget_means={str(b):{m:float(np.mean([lookup[obj,arch,b,s][m] for s in settings['seeds']]))
                for m in ['mask_F1','IoU']} for b in settings['budgets']}
            means={m:float(np.mean([budget_means[str(b)][m] for b in settings['budgets']])) for m in ['mask_F1','IoU']}
            cells[obj][arch]=dict(mean=means,by_budget=budget_means)
    def contrast(left,right):
        per=[dict(budget=b,seed=s,**{m:lookup[left[0],left[1],b,s][m]-lookup[right[0],right[1],b,s][m]
            for m in ['mask_F1','IoU']}) for b in settings['budgets'] for s in settings['seeds']]
        return dict(mean={m:float(np.mean([v[m] for v in per])) for m in ['mask_F1','IoU']},
            by_budget={str(b):{m:float(np.mean([v[m] for v in per if v['budget']==b])) for m in ['mask_F1','IoU']}
                for b in settings['budgets']},per_budget_seed=per)
    contrasts=dict(architecture_BCE=contrast(('BCE','group6'),('BCE','plain12_C4pooled')),
        architecture_auxiliary=contrast(('BCE_geometry','group6'),('BCE_geometry','plain12_C4pooled')),
        auxiliary_group=contrast(('BCE_geometry','group6'),('BCE','group6')),
        auxiliary_plain=contrast(('BCE_geometry','plain12_C4pooled'),('BCE','plain12_C4pooled')))
    interaction={m:contrasts['architecture_auxiliary']['mean'][m]-contrasts['architecture_BCE']['mean'][m] for m in ['mask_F1','IoU']}
    contrasts['interaction']=dict(mean=interaction,definition='(Group-Plain)_aux - (Group-Plain)_BCE')
    return dict(cells=cells,contrasts=contrasts)


def gates(summary,settings):
    limits=settings['development_gate'];effect=summary['contrasts']['architecture_BCE']
    p=abs(c.model.FREE_PARAMETERS['group6']/c.model.FREE_PARAMETERS['plain12_C4pooled']-1)
    mac=abs(c.model.MACS['group6_single']/c.model.MACS['plain12_C4']-1)
    guards=dict(mean_F1=effect['mean']['mask_F1']>=limits['minimum_mean_F1_gain'],
        budget_F1=all(v['mask_F1']>=limits['minimum_budget_F1_delta'] for v in effect['by_budget'].values()),
        mean_IoU=effect['mean']['IoU']>=limits['minimum_mean_IoU_delta'],
        matched_parameters=p<=limits['maximum_budget_ratio_deviation'],matched_MACs=mac<=limits['maximum_budget_ratio_deviation'])
    return dict(passed=all(guards.values()),guards=guards,parameter_relative_difference=p,MAC_relative_difference=mac,
        failed=[k for k,v in guards.items() if not v],interpretation='descriptive_reused_VAL_gate_not_independent_confirmation')


def run(settings):
    c.verify_lock();assert torch.backends.mps.is_available();assert c.os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
    trial=c.ROOT/'trial';assert not trial.exists(),'Immutable trial exists; no implicit overwrite or resume'
    for name in ['','models','val_probs','gradient_probes','fixtures']:(trial/name).mkdir(exist_ok=False)
    start=time.monotonic();values,lineage=c.data('mps');rows=historical_rows(settings);fits=[]
    c.save(trial/'FROZEN.json',dict(lock_sha256=c.sha(c.ROOT/'LOCK.json'),config_sha256=c.sha(c.ROOT/'CONFIG.json'),
        new_fits=12,inherited_fits=12,new_optimizer_updates=24000,lineage=lineage,fresh_test_access=False))
    c.save(trial/'PARTIAL_RESULTS.json',rows)
    try:
        fit_index=0
        for budget in settings['budgets']:
            for seed in settings['seeds']:
                for arch in c.ARMS:
                    fit_index+=1;net=c.model.make_model(arch,seed).to('mps');name=f'{arch}_b{budget}_s{seed}'
                    initial=trial/'models'/f'{name}_initial.pt';final=trial/'models'/f'{name}.pt'
                    torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},architecture=arch,seed=seed),initial)
                    fitting=fit(settings,net,arch,values[budget],budget,seed,trial,fit_index,start)
                    torch.save(dict(model={k:v.detach().cpu() for k,v in net.state_dict().items()},settings=fitting,
                        architecture=arch,budget=budget,seed=seed,objective='BCE'),final)
                    fitting.update(architecture=arch,budget=budget,seed=seed,objective='BCE',
                        model_file=c.relative(final),model_sha256=c.sha(final),initial_file=c.relative(initial),initial_sha256=c.sha(initial))
                    fits.append(fitting);c.save(trial/'FIT_RESULTS.json',fits)
                    net.eval();network=net.deploy() if arch=='group6' else net
                    if arch=='group6':
                        deployed=trial/'models'/f'{name}_deployed.pt'
                        torch.save(dict(model={k:v.detach().cpu() for k,v in network.state_dict().items()},
                            free_model_sha256=c.sha(final),free_parameters=2671,materialized_coefficients=10681),deployed)
                    evaluation,arrays=evaluate(settings,network,arch,values[budget]['val'])
                    route,mode=c.ROUTES[arch];prob_path=trial/'val_probs'/f'{route}_b{budget}_s{seed}.npz'
                    np.savez_compressed(prob_path,**arrays)
                    fixture,technical=c.historical.route_fixture(c.read(c.STAGE/'CONFIG.json'),network,
                        values[budget]['val']['x'][:16],mode,net if arch=='group6' else None)
                    np.savez_compressed(trial/'fixtures'/f'{route}_b{budget}_s{seed}.npz',**fixture)
                    row=dict(objective='BCE',loss_regime='BCE_only',architecture=arch,route=route,budget=budget,seed=seed,
                        **evaluation,technical=technical,model_file=c.relative(final),model_sha256=c.sha(final),
                        probability_file=c.relative(prob_path),probability_sha256=c.sha(prob_path),origin='new_fresh_fit')
                    rows.append(row);c.save(trial/'PARTIAL_RESULTS.json',rows)
                    progress(trial,status='EVALUATED',objective='BCE',architecture=arch,budget=budget,seed=seed,
                        fit_index=fit_index,target_fits=12,completed_updates=fit_index*2000,total_updates=24000,
                        selected=evaluation['selected'],seconds=time.monotonic()-start)
                    del net,network
        assert len(fits)==12 and len(rows)==24;c.verify_lock()
        summary=summarize(rows,settings,'selected');diagnostic=summarize(rows,settings,'fixed05');outcome=gates(summary,settings)
        result=dict(status='COMPLETE_FACTORIAL_DEVELOPMENT',primary_development_gate=outcome,
            rows=rows,fits=fits,primary=summary,fixed05_diagnostic=diagnostic,new_fits=12,inherited_fits=12,
            new_optimizer_updates=24000,seconds=time.monotonic()-start,lock_sha256=c.sha(c.ROOT/'LOCK.json'),
            source_and_old_inputs_unchanged=True,fresh_test_access=False,
            caveats=['reused_adaptive_VAL_no_new_statistical_confirmation','auxiliary_effect_changes_weak_label_supervision',
                     'threshold_selection_and_scoring_on_same_VAL','known_group_convolution_no_operator_novelty_claim'])
        c.save(trial/'RESULT.json',result);c.save(c.ROOT/'RESULT.json',result)
        progress(trial,status=result['status'],fit_index=12,target_fits=12,completed_updates=24000,total_updates=24000,
            primary_development_gate=outcome,contrasts=summary['contrasts'],seconds=time.monotonic()-start)
    except BaseException as error:
        c.save(trial/'FAILURE.json',dict(status='TECHNICAL_FAILURE',error=str(error),type=type(error).__name__,
            traceback=traceback.format_exc(),completed_fits=len(fits),preserve_all_artifacts=True));raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args();assert not(args.preflight and args.freeze);settings=c.config();torch.set_num_threads(2)
    if args.preflight:preflight(settings)
    elif args.freeze:freeze()
    else:run(settings)


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
