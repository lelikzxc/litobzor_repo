"""Bounded, validation-only None-bias ablation on four completed checkpoints.

Run only after all paired MPS runs AND summarize_pairs.py have completed:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python checkpoints/semiwafernet_mps_followup_20261008/calibrate_none_bias.py

No model inference, training, checkpoint writes, or official Test images are
performed. Cached CPU logits are mandatory. This diagnostic deviation does not
change publication defaults (eval_prior_scale remains 0). The 991 validation
maps were also used for checkpoint selection, so fitted validation scores are
optimistic. The 9000 extra None maps already informed this investigation; stress
results are descriptive, not a fresh unbiased evaluation or article result.
"""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='1'

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).resolve().parent
SEEDS=[42,43]
ARMS=['natural_none','stratified_none']
GRID=[0.,.5,1.,1.5,2.,2.5,3.,3.5,4.]
CLASSES=['none','Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
PUBLISHED_NONE_PRIOR=110701/118595


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def size_bin(value):return 0 if value<700 else 1 if value<2500 else 2


def training_population():
    counts=np.zeros(9,dtype=np.int64);bins=np.zeros(3,dtype=np.int64);rows={}
    labels=ROOT/'datasets/wm811k/labels.csv';mapping={name:i for i,name in enumerate(CLASSES)}
    with labels.open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['trianTestLabel'].strip().lower()!='training':continue
            label=row['failureType'].strip()
            if not label or label.lower()=='nan':continue
            try:label=mapping[label] if label in mapping else int(label)
            except ValueError:continue
            filename=row['filename'].strip();active=int(row['dieSize'])
            if filename in rows or not 0<=label<9:raise ValueError('Invalid official Training inventory')
            rows[filename]=(label,active);counts[label]+=1
            if label==0:bins[size_bin(active)]+=1
    if np.any(counts==0) or np.any(bins==0):raise ValueError('Missing Training class or None size stratum')
    return rows,counts,bins


def prediction(logits,bias):
    changed=logits.copy();changed[:,0]+=bias
    return changed.argmax(1)


def metrics(target,pred,weights=None):
    c=np.bincount(target*9+pred,weights=weights,minlength=81).reshape(9,9)
    diagonal=np.diag(c).astype(np.float64);support=c.sum(1);predicted=c.sum(0)
    precision=np.divide(diagonal,predicted,out=np.zeros(9),where=predicted>0)
    recall=np.divide(diagonal,support,out=np.zeros(9),where=support>0)
    f1=np.divide(2*diagonal,support+predicted,out=np.zeros(9),where=support+predicted>0)
    return {'accuracy':float(diagonal.sum()/c.sum()),'macro_f1':float(f1.mean()),
        'macro_precision':float(precision.mean()),'balanced_accuracy':float(recall.mean()),
        'None_to_Scratch_rate':float(c[0,8]/support[0]),'None_to_any_defect_rate':float(1-recall[0]),
        'Scratch_recall':float(recall[8]),'Scratch_precision':float(precision[8]),
        'None_f1':float(f1[0]),'Scratch_f1':float(f1[8]),'confusion_matrix':c.tolist(),
        'per_class':[{'class':name,'support':float(support[i]),'predicted_support':float(predicted[i]),
                      'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])}
                      for i,name in enumerate(CLASSES)]}


def validation_weights(labels,names,population,counts,bins,none_prior):
    supports=np.bincount(labels,minlength=9)
    val_bins=np.array([size_bin(population[name][1]) if y==0 else -1 for name,y in zip(names,labels)])
    bin_supports=np.array([(val_bins==i).sum() for i in range(3)])
    if not np.array_equal(bin_supports,[100,100,100]):raise ValueError('Unexpected saved None validation strata')
    target_classes=np.zeros(9);target_classes[0]=none_prior
    target_classes[1:]=(1-none_prior)*counts[1:]/counts[1:].sum()
    weights=np.array([none_prior*(bins[bin_index]/bins.sum())/bin_supports[bin_index]
                      if y==0 else target_classes[y]/supports[y]
                      for y,bin_index in zip(labels,val_bins)],dtype=np.float64)
    if not math.isclose(float(weights.sum()),1.,abs_tol=1e-12):raise ValueError('Validation weights do not sum to 1')
    return weights,target_classes


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=BASE/'none_bias_calibration.json')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Use a fresh output; preserve calibration decisions')
    started=time.perf_counter();state=read(BASE/'paired_run_state.json')
    expected={(seed,arm) for seed in SEEDS for arm in ARMS}
    if state.get('status')!='complete' or {(r['seed'],r['arm']) for r in state['runs']}!=expected:
        raise RuntimeError('All four predetermined paired MPS runs must finish first')
    if any(r.get('status')!='complete' or r.get('exit_code')!=0 for r in state['runs']):raise RuntimeError('A paired run did not complete')
    builder=read(BASE/'paired_splits/builder_summary.json')
    if sha(ROOT/'datasets/wm811k/labels.csv')!=builder['labels_csv_sha256']:raise RuntimeError('Training labels changed')
    population,counts,bins=training_population()
    scenarios=[('observed_Training_prior',float(counts[0]/counts.sum())),
               ('published_None_prior_assumption',PUBLISHED_NONE_PRIOR)]
    results=[];source=[]
    for seed in SEEDS:
      for arm in ARMS:
        path=BASE/f'paired_seed{seed}/{arm}';report=read(path/'report.json');split=read(BASE/f'paired_splits/{arm}.json')
        if report['samples']!=split['samples'] or report['none_stress_extra_samples']!=split['none_stress_extra_samples']:
            raise RuntimeError('Executed report differs from declared split')
        val=split['samples']['validation'];extra=split['none_stress_extra_samples'];rows=val+extra
        names=[name for name,_ in rows];labels=np.array([label for _,label in rows],dtype=np.int64)
        if len(val)!=991 or len(extra)!=9000:raise ValueError('Unexpected holdout sizes')
        if len(set(names))!=len(names) or any(population.get(name,(None,))[0]!=label for name,label in rows):raise ValueError('Invalid official Training holdouts')
        cache=path/'cpu_holdout_logits.npz';meta=read(path/'cpu_holdout_logits.json')
        if meta['checkpoint_sha256']!=sha(path/'mid_smote.pt') or meta['npz_sha256']!=sha(cache) or meta['validation_count']!=991:
            raise RuntimeError('CPU logit cache identity mismatch')
        with np.load(cache,allow_pickle=False) as saved:
            if not np.array_equal(saved['filenames'],np.array(names)) or not np.array_equal(saved['true_labels'],labels):raise ValueError('CPU logits have different role ordering')
            logits=saved['logits'].astype(np.float64)
        if logits.shape!=(9991,9) or not np.isfinite(logits).all():raise ValueError('Invalid logits')
        n=len(val);raw=prediction(logits,0.)
        variant=next(v for v in report['variants'] if v['name']=='mid_smote')
        baseline={'validation':metrics(labels[:n],raw[:n]),'stress':metrics(labels,raw)}
        match=(baseline['validation']['confusion_matrix']==variant['best_validation']['confusion_matrix'] and baseline['stress']['confusion_matrix']==variant['none_stress_validation']['confusion_matrix'])
        source.append({'seed':seed,'arm':arm,'cache':str(cache.relative_to(ROOT)),
            'checkpoint_sha256':meta['checkpoint_sha256'],'npz_sha256':meta['npz_sha256'],
            'baseline_CPU_confusion_exactly_matches_saved_MPS':match})
        for scenario,prior in scenarios:
            weights,target_classes=validation_weights(labels[:n],names[:n],population,counts,bins,prior)
            grid=[]
            # Stress scores are intentionally unavailable during parameter selection.
            for bias in GRID:
                pred=prediction(logits[:n],bias)
                weighted=metrics(labels[:n],pred,weights);plain=metrics(labels[:n],pred)
                grid.append({'bias':bias,'weighted_validation_macro_f1':weighted['macro_f1'],
                    'unweighted_validation_macro_f1':plain['macro_f1'],'validation_Scratch_recall':plain['Scratch_recall']})
            # Fixed grid order gives the smallest bias if objectives tie.
            selected=max(grid,key=lambda row:row['weighted_validation_macro_f1'])
            bias=selected['bias'];chosen=prediction(logits,bias)
            raw_weighted=metrics(labels[:n],raw[:n],weights)
            calibrated={'validation':metrics(labels[:n],chosen[:n]),'stress':metrics(labels,chosen),
                'weighted_validation':metrics(labels[:n],chosen[:n],weights)}
            results.append({'seed':seed,'arm':arm,'scenario':scenario,'target_None_prior':prior,
                'target_class_probabilities':target_classes.tolist(),'chosen_None_logit_bias':bias,
                'chosen_at_grid_boundary':bias==GRID[-1],'validation_grid':grid,
                'selection_rule':'Maximum target-weighted validation macro F1; smallest bias on a tie; no stress fitting',
                'baseline':{**baseline,'weighted_validation':raw_weighted},'calibrated':calibrated,
                'stress_macro_f1_delta':calibrated['stress']['macro_f1']-baseline['stress']['macro_f1'],
                'stress_Scratch_recall_delta':calibrated['stress']['Scratch_recall']-baseline['stress']['Scratch_recall']})
    summary=[]
    for arm in ARMS:
      for scenario,_ in scenarios:
        group=[r for r in results if r['arm']==arm and r['scenario']==scenario]
        key_metrics=['macro_f1','accuracy','balanced_accuracy','None_to_Scratch_rate','Scratch_precision','Scratch_recall']
        summary.append({'arm':arm,'scenario':scenario,'seeds':SEEDS,
            'chosen_biases':[r['chosen_None_logit_bias'] for r in group],
            'baseline_stress':{key:{'mean':statistics.mean(r['baseline']['stress'][key] for r in group),'sample_sd':statistics.stdev(r['baseline']['stress'][key] for r in group)} for key in key_metrics},
            'calibrated_stress':{key:{'mean':statistics.mean(r['calibrated']['stress'][key] for r in group),'sample_sd':statistics.stdev(r['calibrated']['stress'][key] for r in group)} for key in key_metrics}})
    output={'scope':'Predetermined validation-only fixed-checkpoint diagnostic; no training, inference, official Test maps, source/config changes or scenario selection by stress.',
        'fixed_seeds':SEEDS,'fixed_arms':ARMS,'fixed_bias_grid':GRID,'paired_run_state_sha256':sha(BASE/'paired_run_state.json'),
        'population_source':'Only labeled official Training rows and dieSize metadata from unchanged labels.csv',
        'Training_class_counts':counts.tolist(),'Training_None_size_bin_counts':bins.tolist(),
        'validation_geometry_weighting':'None size strata are reweighted from 100/100/100 to empirical official Training bin proportions.',
        'validation_class_weighting':'Use each scenario None prior and preserve official Training conditional proportions among the 8 defect classes; this makes minority precision reflect the declared environment.',
        'primary_scenario':'observed_Training_prior','secondary_predeclared_scenario':'published_None_prior_assumption',
        'secondary_prior_source':'Existing README Table 1 count 110701/118595; a scenario assumption only, not use of official Test labels or images.',
        'evaluation_deviation':'Adds one selected scalar to logit 0 only; publication eval_prior_scale 0 remains unchanged. It is distinct from scaling all class log priors.',
        'limitations':['Validation reused for checkpoint selection and calibration: fitted validation scores are optimistic.',
            'Stress already informed hypotheses; its post-calibration numbers are descriptive, not unbiased confirmation.',
            'Training geometry/class proportions may not transfer to an unknown evaluation distribution; paper metric/protocol inconsistency remains unresolved.',
            'Near-full has only 10 validation maps; conditional metrics have substantial sampling uncertainty.',
            'A bias may improve precision while reducing defect recall. Both are reported; no scenario or seed is selected using stress.',
            'A grid-boundary winner is marked and does not trigger grid expansion or extra experiments.'],
        'sources':source,'results':results,'summary_by_arm_and_scenario':summary,'seconds':time.perf_counter()-started}
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(output,indent=2));temporary.replace(args.output)
    print('Saved',args.output,'CPU seconds',output['seconds'])
    for row in summary:print(row['arm'],row['scenario'],'biases',row['chosen_biases'],'stress F1',row['baseline_stress']['macro_f1']['mean'],'->',row['calibrated_stress']['macro_f1']['mean'],'Scratch recall',row['baseline_stress']['Scratch_recall']['mean'],'->',row['calibrated_stress']['Scratch_recall']['mean'])


if __name__=='__main__':main()
