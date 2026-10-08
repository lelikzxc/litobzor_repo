"""Build paired, official-Training-only morphology diagnostics; no training.

Run from the repository root. To reproduce without overwriting artifacts:
  python checkpoints/semiwafernet_mps_followup_20261008/paired_splits/build_paired_splits.py --output-dir /private/tmp/semi-paired-reproduction

Non-None rows retain the exact completed warmup roles. None holdouts are fresh
relative to earlier saved diagnostics. This resampling is an explicit bounded
diagnostic interpretation, not a change to the paper reproduction recipe.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from papers.wafer_encoding import decode_die_map

BASE = ROOT / 'checkpoints/semiwafernet_mps_followup_20261008'
SOURCE = BASE / 'warmup_seed42/report.json'
PREVIOUS_REPORTS = [
    ROOT / 'checkpoints/mps_diagnostics/semiwafernet_20261008/report.json',
    ROOT / 'checkpoints/mps_diagnostics/semiwafernet_extension_20261008/report.json',
    ROOT / 'checkpoints/mps_diagnostics/semiwafernet_none_diversity_20261008/report.json',
    SOURCE,
]
CLASSES = ['none','Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
SEEDS = {'source_data_seed':42, 'validation_none':420101, 'stress_none':420102,
         'natural_training_none':420103, 'stratified_training_none':420104}
BIN_NAMES = ['under700', '700_to2499', 'at_least2500']


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bin_index(active):
    return 0 if active < 700 else 1 if active < 2500 else 2


def choose(rows, size, seed):
    if len(rows) < size:
        raise ValueError(f'Insufficient unique originals: need {size}, have {len(rows)}')
    order = np.random.RandomState(seed).permutation(len(rows))[:size]
    return [rows[int(i)] for i in order]


def counts(rows):
    result = [0]*9
    for _,label in rows: result[label] += 1
    return result


def inventory():
    samples, metadata = [], {}
    mapping = {name:i for i,name in enumerate(CLASSES)}
    with (ROOT/'datasets/wm811k/labels.csv').open(encoding='utf-8',newline='') as handle:
        for row in csv.DictReader(handle):
            if row['trianTestLabel'].strip().lower() != 'training':
                continue
            name, label = row['filename'].strip(), row['failureType'].strip()
            if not label or label.lower() == 'nan': continue
            if label in mapping: value = mapping[label]
            else:
                try: value = int(label)
                except ValueError: continue
            if name in metadata: raise ValueError(f'Duplicate official Training filename: {name}')
            if not 0 <= value < 9: raise ValueError(f'Invalid label for {name}: {value}')
            samples.append((name,value))
            metadata[name]={'label':value,'dieSize':int(row['dieSize'])}
    return samples, metadata


def fresh_exclusions():
    excluded=set();sources=[]
    for path in PREVIOUS_REPORTS:
        if not path.exists(): raise FileNotFoundError(path)
        report=json.loads(path.read_text())
        roles=list(report['samples'].values()) + [report.get('none_stress_extra_samples',[]), report.get('diverse_none_extra_training_samples',[])]
        for rows in roles:
            excluded.update(name for name,label in rows if label==0)
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)})
    return excluded,sources


def raw_features(rows, metadata):
    result={}
    for name,label in rows:
        if name in result: continue
        path=ROOT/'datasets/wm811k/images'/name
        if not path.exists(): path=path.with_suffix('.png')
        with Image.open(path) as image: die=decode_die_map(image)
        height,width=die.shape
        active=int((die>0).sum());failed=int((die==2).sum())
        if active != metadata[name]['dieSize']:
            raise ValueError(f'CSV dieSize differs from categorical raw active dies: {name}')
        result[name]={'label':label,'height':height,'width':width,'raw_active_dies':active,
                      'raw_failed_dies':failed,'raw_failed_fraction':failed/active if active else 0.,
                      'size_stratum':BIN_NAMES[bin_index(active)]}
    return result


def describe(rows, features):
    none=[features[name] for name,label in rows if label==0]
    size_counts={name:0 for name in BIN_NAMES}
    for row in none:size_counts[row['size_stratum']]+=1
    values=[row['raw_failed_fraction'] for row in none]
    return {'rows':len(rows),'class_counts':counts(rows),'None_count':len(none),
            'None_size_counts':size_counts,'None_failed_fraction_under5percent':sum(v<.05 for v in values),
            'None_failed_fraction_mean':float(np.mean(values)) if values else None,
            'None_failed_fraction_quantiles_0_25_50_75_100':np.quantile(values,[0,.25,.5,.75,1]).tolist() if values else [],
            'None_shapes':dict(Counter(f"{row['height']}x{row['width']}" for row in none))}


def validate_roles(train,val,stress,training_inventory,metadata):
    seen=set()
    for role,rows in [('train',train),('validation',val),('none_stress',stress)]:
        for name,label in rows:
            if training_inventory.get(name)!=label: raise ValueError(f'{role} not official Training: {name}')
            if name in seen: raise ValueError(f'Duplicate or role overlap: {name}')
            if role=='none_stress' and label!=0: raise ValueError('Stress extra rows must be None')
            seen.add(name)
        if role!='none_stress' and {label for _,label in rows}!=set(range(9)):
            raise ValueError(f'{role} must contain all9 classes')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args();out=args.output_dir
    names=['natural_none.json','stratified_none.json','builder_summary.json']
    if any((out/name).exists() for name in names): raise FileExistsError('Use a fresh output directory; existing paired split artifacts are preserved')
    started=time.perf_counter();source=json.loads(SOURCE.read_text())
    samples,metadata=inventory();source_inventory=dict(samples)
    excluded,exclusion_sources=fresh_exclusions()
    nonnone_train=[(name,label) for name,label in source['samples']['train'] if label!=0]
    nonnone_val=[(name,label) for name,label in source['samples']['validation'] if label!=0]
    none=[row for row in samples if row[1]==0 and row[0] not in excluded]
    bins=[[row for row in none if bin_index(metadata[row[0]]['dieSize'])==i] for i in range(3)]
    # Reserve originals before either training arm is sampled.
    val_none=[]
    for i,pool in enumerate(bins):val_none.extend(choose(pool,100,SEEDS['validation_none']+i))
    held={name for name,_ in val_none}
    stress=choose([row for row in none if row[0] not in held],9000,SEEDS['stress_none'])
    held.update(name for name,_ in stress)
    candidates=[row for row in none if row[0] not in held]
    training_bins=[[row for row in candidates if bin_index(metadata[row[0]]['dieSize'])==i] for i in range(3)]
    for i,pool in enumerate(training_bins):
        if len(pool)<240: raise ValueError(f'Insufficient bin {BIN_NAMES[i]} after holdouts: {len(pool)} originals, need240')
    natural=choose(candidates,720,SEEDS['natural_training_none'])
    stratified=[]
    for i,pool in enumerate(training_bins):stratified.extend(choose(pool,240,SEEDS['stratified_training_none']+i))
    validation=val_none+nonnone_val
    features=raw_features(val_none+stress+natural+stratified,metadata)
    invariant={'source_report':str(SOURCE.relative_to(ROOT)),'source_report_sha256':sha256_file(SOURCE),
       'builder_script':str(Path(__file__).resolve().relative_to(ROOT)),
       'builder_script_sha256':sha256_file(Path(__file__)),
       'labels_csv_sha256':sha256_file(ROOT/'datasets/wm811k/labels.csv'),
       'fixed_seeds':SEEDS,'bin_definition':{'under700':'raw active dies <700','700_to2499':'700 <= raw active dies <2500','at_least2500':'raw active dies >=2500'},
       'metadata_dieSize_verified_against_raw_images_for_all_selected_None':True,
       'fresh_None_exclusion_report_hashes':exclusion_sources,'previous_None_maps_excluded':len(excluded),
       'eligible_fresh_None_pool':len(none),'eligible_fresh_None_bin_counts':{name:len(pool) for name,pool in zip(BIN_NAMES,bins)},
       'training_candidate_None_bin_counts_after_shared_holdouts':{name:len(pool) for name,pool in zip(BIN_NAMES,training_bins)},
       'shared_validation':describe(validation,features),'shared_stress_extra':describe(stress,features),
       'holdouts_never_used_in_either_training_arm':True,'no_replacement_or_duplicates':True,
       'all_rows_verified_official_Training_only':True,'non_None_train_and_validation_roles_identical_to_source':True,
       'paired_training_None_overlap_permitted':len({name for name,_ in natural}&{name for name,_ in stratified}),
       'note':'Explicit diagnostic None resampling; not the published default. Models use validation macro F1 for checkpoint selection; shared natural stress is descriptive only.'}
    artifacts={}
    for variant,none_train in [('natural_none',natural),('stratified_none',stratified)]:
        train=none_train+nonnone_train
        validate_roles(train,validation,stress,source_inventory,metadata)
        wrapper=copy.deepcopy(source)
        for key in ('torch_version','device','preparation_seconds','limits','split','variants'):
            wrapper.pop(key,None)
        wrapper.update({'seed':42,'data_seed':42,'source_data_seed':42,'diagnostic_split_only':True,
            'training_inventory_digest':hashlib.sha256(json.dumps(samples,separators=(',',':')).encode()).hexdigest(),
            'samples':{'train':train,'validation':validation},'none_stress_extra_samples':stress,
            'diverse_none_extra_training_samples':[],'diverse_none_extra_disjoint_from_base_train_validation_stress':True,
            'training_counts':counts(train),'validation_counts':counts(validation),'smote_counts':None,
            'none_stress_counts':counts(validation+stress),'none_stress_none_fraction':(len(val_none)+len(stress))/(len(validation)+len(stress)),
            'none_stress_disjoint_from_training':True,'extra_none_disjoint_from_balanced_validation':True,
            'subset_digest':hashlib.sha256(json.dumps({'train':train,'val':validation}).encode()).hexdigest(),
            'variants':[],'paired_variant':variant,'builder_metadata':{**invariant,'training':describe(train,features)},
            'split':'Paired morphology diagnostic with identical non-None roles and shared original None holdouts',
            'limitations':['Explicit diagnostic resampling, not a paper-default recipe.',
                'None validation is deliberately size-stratified and class counts differ from the completed warmup validation.',
                'The natural None stress prior is artificial and does not establish official Test performance.']})
        assert counts(train)==source['training_counts']
        expected=list(source['validation_counts']);expected[0]=300
        assert counts(validation)==expected
        if variant=='stratified_none':assert wrapper['builder_metadata']['training']['None_size_counts']==dict(zip(BIN_NAMES,[240]*3))
        artifacts[variant]=wrapper
    assert artifacts['natural_none']['samples']['validation']==artifacts['stratified_none']['samples']['validation']
    assert artifacts['natural_none']['none_stress_extra_samples']==artifacts['stratified_none']['none_stress_extra_samples']
    out.mkdir(parents=True,exist_ok=True)
    for variant,wrapper in artifacts.items():
        target=out/f'{variant}.json';temporary=target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(wrapper,indent=2));temporary.replace(target)
    summary={**invariant,'builder_seconds':time.perf_counter()-started,
        'arms':{variant:{'path':str((out/f'{variant}.json').resolve()),'sha256':sha256_file(out/f'{variant}.json'),
              'training':wrapper['builder_metadata']['training'],'training_rows':len(wrapper['samples']['train']),
              'validation_rows':len(wrapper['samples']['validation']),'stress_extra_rows':len(wrapper['none_stress_extra_samples'])}
              for variant,wrapper in artifacts.items()},'selected_None_raw_features':features}
    (out/'builder_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({key:summary[key] for key in ['builder_seconds','eligible_fresh_None_pool','eligible_fresh_None_bin_counts','previous_None_maps_excluded']},indent=2))
    for variant,wrapper in artifacts.items():print(variant,'train',wrapper['training_counts'],'val',wrapper['validation_counts'],'None strata',wrapper['builder_metadata']['training']['None_size_counts'])
    print('Paired splits validated and saved; no training or GPU operations performed.')


if __name__=='__main__':main()
