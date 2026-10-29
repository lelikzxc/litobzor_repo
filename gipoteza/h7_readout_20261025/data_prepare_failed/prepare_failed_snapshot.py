"""Freeze a target-defined readout benchmark, then decode stored native RI."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from core import block_sum,native_label

BASE=Path(__file__).resolve().parent
OLD=BASE.parent/'h6_selection_audit_20261025/fresh_combined'
sys.path.insert(0,str(BASE.parent/'h6_revision_1'))
from local_events import observe


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def image(raw):
    with Image.open(io.BytesIO(raw)) as im:
        assert im.mode=='1' and im.size==(4096,4096)
        return np.not_equal(np.array(im,bool),False)


def main():
    out=BASE/'data';out.mkdir(exist_ok=False)
    old=json.loads((OLD/'INPUTS_FROZEN.json').read_text());labels=json.loads((OLD/'labels/MANIFEST.json').read_text())
    geometries=sorted(old['geometries'],key=lambda g:hashlib.sha256(('h7/readout-v1/'+g).encode()).hexdigest())
    records=[sorted([r for r in labels['records'] if r['target_hash']==g],key=lambda r:(r['archive'],r['group']))[0] for g in geometries]
    selected=[0,5,10,11,16,21,22,27,32]
    files=[BASE/'PROTOCOL.md',Path(__file__),BASE/'core.py',OLD/'INPUTS_FROZEN.json',OLD/'labels/MANIFEST.json']
    files.extend(OLD/f'{prefix}_{g}.{ext}' for g in geometries for prefix,ext in [('events','json'),('regions','npz')])
    frozen=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='DEVELOPMENT_READOUT_FEASIBILITY_NOT_CONFIRMATION',
        train=geometries[:4],evaluation=geometries[4:],records=records,state_indices=selected,
        geometry_selection='Fixed hash rule, one lexically first recipe per target, all56 target-defined events',
        old_geometry_separation='Inherited old fresh_combined D4/shift rejection at IoU0.9; not independent chip families',
        hashes={str(p):sha(p) for p in files},published_test_used=False)
    write(out/'FROZEN.json',frozen)
    packed=[];coarse=[];template=[];state_id=[];truth=[];geometry=[];state_truth=[];state_geometry=[]
    templates={k:[] for k in ['a','b','domain','target']};kind=[];ref=[];cases=[];sources={}
    for gi,(g,record) in enumerate(zip(geometries,records)):
        assert sha(record['file'])==record['sha256']
        with np.load(record['file']) as z:
            expected={k:z[k].copy() for k in ['state_q','state_missing','state_violation','state_y']}
        events=json.loads((OLD/f'events_{g}.json').read_text())['events']
        with np.load(OLD/f'regions_{g}.npz') as z:regions={k:z[k].astype(bool) for k in z.files}
        with tarfile.open(record['archive'],'r:') as tar:
            raw=tar.extractfile(record['group']+'/layout.png').read();layout=image(raw)
            assert hashlib.sha256(layout.tobytes()).hexdigest()==g
            for ei,e in enumerate(events):
                y,x,h,w=e['box'];templates['target'].append(np.packbits(layout[y:y+h,x:x+w].ravel()))
                for name in ['a','b','domain']:templates[name].append(np.packbits(regions[name][ei].ravel()))
                kind.append(e['kind']=='forbidden');ref.append(e['q_ref'])
            for si in selected:
                fi,di=divmod(si,11);dose=[-.1,-.08,-.06,-.04,-.02,0,.02,.04,.06,.08,.1][di];focus=[-40,0,40][fi]
                name=f'{record["group"]}/RI_dose_{dose:g}_defocus_{focus}.png'
                raw=tar.extractfile(name).read();produced=image(raw);sources[record['archive']+'::'+name]=hashlib.sha256(raw).hexdigest()
                sid=len(state_truth);state_truth.append(expected['state_y'][si]);state_geometry.append(g)
                for ei,e in enumerate(events):
                    y,x,h,w=e['box'];native=produced[y:y+h,x:x+w]
                    kk=e['kind']=='forbidden';aa,bb,dd=[regions[k][ei] for k in ['a','b','domain']]
                    observed=observe(produced,dict(e,a=aa,b=bb,domain=dd))
                    assert abs(observed['q']-expected['state_q'][si,ei])<1e-5
                    label=2 if expected['state_missing'][si,ei] else int(expected['state_violation'][si,ei])
                    assert native_label(native,aa,bb,dd,kk,e['q_ref'])==label
                    packed.append(np.packbits(native.ravel()));coarse.append(block_sum(native).astype(np.uint8))
                    template.append(gi*56+ei);state_id.append(sid);truth.append(label);geometry.append(g)
                cases.append(dict(state_id=sid,geometry=g,upstream_state_index=si,archive=record['archive'],group=record['group'],dose=dose,focus=focus))
        print(json.dumps(dict(geometry=g,states=len(cases),events=len(truth))),flush=True)
    data=dict(native=np.array(packed),coarse=np.array(coarse),template=np.array(template),state_id=np.array(state_id),
        truth=np.array(truth,dtype=np.int64),geometry=np.array(geometry),kind=np.array(kind),ref=np.array(ref,np.float32),
        state_truth=np.array(state_truth,bool),state_geometry=np.array(state_geometry),**{k:np.array(v) for k,v in templates.items()})
    np.savez_compressed(out/'benchmark.npz',**data)
    assert len(truth)==4032 and len(state_truth)==72
    assert all(sha(p)==h for p,h in frozen['hashes'].items())
    support={}
    for role,gg in [('train',geometries[:4]),('evaluation',geometries[4:])]:
        ii=np.isin(data['state_geometry'],gg);jj=np.isin(data['geometry'],gg)
        support[role]=dict(states=int(ii.sum()),events=int(jj.sum()),positive_states=data['state_truth'][ii].sum(0).tolist(),
            positive_geometries=[sum(data['state_truth'][data['state_geometry']==g,j].any() for g in gg) for j in range(3)],
            local_class_counts=np.bincount(data['truth'][jj],minlength=3).tolist())
    write(out/'MANIFEST.json',dict(status='COMPLETE',benchmark_sha256=sha(out/'benchmark.npz'),
        frozen_sha256=sha(out/'FROZEN.json'),states=cases,support=support,raw_source_members_sha256=sources,
        native_labels_match_old_cache=True,full_reveal_certificate_matches_labels=True))
    print(json.dumps(support,indent=2),flush=True)


if __name__=='__main__':main()
