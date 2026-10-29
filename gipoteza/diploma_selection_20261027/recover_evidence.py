"""Recover historical evidence for diploma selection; never changes old experiments."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
ARCHIVES=PROJECT/'gipoteza/cleanup_history'

def sha(b):return hashlib.sha256(b).hexdigest()

def main():
    source=ROOT/'sources';source.mkdir(exist_ok=True)
    a_path=ARCHIVES/'rejected_results_20261025.json'
    b_path=ARCHIVES/'rejected_results_20261026.json'
    a=json.loads(a_path.read_text());b=json.loads(b_path.read_text())['entries']
    wanted={
      'sem_daft/pilot_v2/REPORT.md':'SEM_CELL_REPORT.md',
      'gipoteza/search_20261026/sem_graded_test/RESULT.json':'SEM_GRADED_RESULT.json',
      'gipoteza/search_20261026/sem_replication/RESULT.json':'SEM_REPLICATION_RESULT.json',
      'gipoteza/longer_20261026/evaluation/RESULT.json':'LONGER_RESULT.json',
      'gipoteza/longer_20261026/RESULT.md':'LONGER_REPORT.md',
      'gipoteza/search_next_20261026/functional_observability/ep2/core_confirmation/REPORT.md':'EP2_CORE_REPORT.md',
      'gipoteza/search_next_20261026/functional_observability/ep2/probe/REPORT.md':'EP2_PILOT_REPORT.md',
      'gipoteza/larger_separator_20261026/RESULT.md':'EP2_LARGER_REPORT.md',
      'gipoteza/NOVELTY_SCREEN_20261025.md':'NOVELTY_SCREEN.md',
    }
    rows=[{'path':p,**v,'archive':a_path.name} for p,v in a.items()]
    rows += [{**e,'archive':b_path.name} for e in b]
    manifest=[]
    for row in rows:
        if row['path'] not in wanted:continue
        data=row['text'].encode('utf-8');dest=source/wanted[row['path']]
        if dest.exists():assert dest.read_bytes()==data,'Do not replace recovered evidence'
        else:dest.write_bytes(data)
        manifest.append({'original_path':row['path'],'archive':row['archive'],
           'saved':str(dest.relative_to(ROOT)),'text_sha256':sha(data),
           'historical_sha256':row.get('sha256'),
           'text_matches_historical_sha256':sha(data)==row.get('sha256')})
    assert len(manifest)==len(wanted)
    long=json.loads((source/'LONGER_RESULT.json').read_text())
    metrics=long['results']['fresh']['fixed2000']['pooled']['metrics']
    base=metrics['base'];effects={}
    for name in ('strong_soft','connectivity','joint'):
        value=metrics[name]
        delta=[100*(v-b) for v,b in zip(value['per_seed_f1'],base['per_seed_f1'])]
        effects[name]={'macro_F1_percent':100*value['macro_f1'],'base_macro_F1_percent':100*base['macro_f1'],
            'difference_pp':100*(value['macro_f1']-base['macro_f1']),
            'per_seed_differences_pp':delta,'positive_all_eight_seeds':all(v>0 for v in delta)}
    result={'purpose':'Historical reanalysis, not new training or a new prospective hypothesis test',
        'archive_entry_count':len(rows),'source_archives':{a_path.name:sha(a_path.read_bytes()),b_path.name:sha(b_path.read_bytes())},
        'recovered':manifest,'descriptive_control_effects':effects,
        'original_joint_hypothesis_status':'FAIL unchanged',
        'warning':'Controls were not separately preregistered as new hypotheses. Positive seed differences do not establish cross-chip transfer.'}
    (ROOT/'EVIDENCE_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'archive_entries':len(rows),'recovered':len(manifest),
        'historical_hashes_match':all(r['text_matches_historical_sha256'] for r in manifest),
        'control_effects_pp':{k:v['difference_pp'] for k,v in effects.items()}},ensure_ascii=False))

if __name__=='__main__':main()
