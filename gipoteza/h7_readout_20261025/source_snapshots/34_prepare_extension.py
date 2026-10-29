"""Fixed all-ten legacy geometries, untouched by H7 fitting/policy development."""
import io
import json
from pathlib import Path
import sys
import tarfile
from datetime import datetime,timezone
import numpy as np
from PIL import Image
from core import block_sum,native_label
from prepare import image,sha,write

BASE=Path(__file__).resolve().parent
REV=BASE.parent/'h6_revision_1';CACHE=REV/'pilot_20261025/cache'
sys.path.insert(0,str(REV))
from local_events import observe


def main():
    out=BASE/'extension_data';out.mkdir(exist_ok=False)
    manifest=json.loads((CACHE/'manifest.json').read_text());old=json.loads((BASE/'data/FROZEN.json').read_text())
    gg=sorted({r['geometry'] for r in manifest['recipes']});assert len(gg)==10
    assert not set(gg)&set(old['train']+old['evaluation'])
    records=[next(r for r in manifest['recipes'] if r['geometry']==g) for g in gg]
    selected=[0,5,10,11,16,21,22,27,32]
    source=[Path(__file__),BASE/'core.py',BASE/'prepare.py',BASE/'COST_EXTENSION_PROTOCOL.md',CACHE/'manifest.json',BASE/'evaluation/SUMMARY.json']
    frozen=dict(created_utc=datetime.now(timezone.utc).isoformat(),purpose='PROSPECTIVE_H7_COST_ENDPOINT_ON_OTHER_KNOWN_DESIGNS',
        geometries=gg,records=records,state_indices=selected,hashes={str(p):sha(p) for p in source},
        published_test_used=False,new_training=False)
    write(out/'FROZEN.json',frozen)
    packed=[];coarse=[];ti=[];sid=[];truth=[];geo=[];sy=[];sg=[];cases=[]
    templates={k:[] for k in ['a','b','domain','target']};kind=[];ref=[];boxes=[]
    for gi,(g,record) in enumerate(zip(gg,records)):
        assert sha(CACHE/record['file'])==record['cache_sha256']
        with np.load(CACHE/record['file']) as z:expected={k:z[k].copy() for k in ['state_q','state_missing','state_violation','state_y']}
        folder=REV/record['source_folder'];ep=folder/f'events_{g}.json';rp=folder/f'regions_{g}.npz'
        assert sha(ep)==record['spec_sha256'] and sha(rp)==record['regions_sha256']
        events=json.loads(ep.read_text())['events']
        with np.load(rp) as z:regions={k:z[k].astype(bool) for k in z.files}
        with tarfile.open(record['archive'],'r:*') as tar:
            raw=tar.extractfile(record['group']+'/layout.png').read();assert sha_bytes(raw)==record['layout_file_sha256'];layout=image(raw)
            for ei,e in enumerate(events):
                y,x,h,w=e['box'];templates['target'].append(np.packbits(layout[y:y+h,x:x+w].ravel()));boxes.append(e['box'])
                for k in ['a','b','domain']:templates[k].append(np.packbits(regions[k][ei].ravel()))
                kind.append(e['kind']=='forbidden');ref.append(e['q_ref'])
            for si in selected:
                fi,di=divmod(si,11);dose=[-.1,-.08,-.06,-.04,-.02,0,.02,.04,.06,.08,.1][di];focus=[-40,0,40][fi]
                produced=image(tar.extractfile(f'{record["group"]}/RI_dose_{dose:g}_defocus_{focus}.png').read())
                state=len(sy);sy.append(expected['state_y'][si]);sg.append(g)
                for ei,e in enumerate(events):
                    y,x,h,w=e['box'];native=produced[y:y+h,x:x+w];a,b,domain=[regions[k][ei] for k in ['a','b','domain']]
                    label=2 if expected['state_missing'][si,ei] else int(expected['state_violation'][si,ei])
                    observed=observe(produced,dict(e,a=a,b=b,domain=domain))
                    assert abs(observed['q']-expected['state_q'][si,ei])<1e-5
                    assert native_label(native,a,b,domain,e['kind']=='forbidden',e['q_ref'])==label
                    packed.append(np.packbits(native.ravel()));coarse.append(block_sum(native).astype(np.uint8))
                    ti.append(gi*56+ei);sid.append(state);truth.append(label);geo.append(g)
                cases.append(dict(state_id=state,geometry=g,upstream_state_index=si))
        print(json.dumps(dict(geometry=g,states=len(sy),events=len(truth))),flush=True)
    np.savez_compressed(out/'benchmark.npz',native=np.array(packed),coarse=np.array(coarse),template=np.array(ti),state_id=np.array(sid),
        truth=np.array(truth),geometry=np.array(geo),kind=np.array(kind),ref=np.array(ref,np.float32),box=np.array(boxes),
        state_truth=np.array(sy,bool),state_geometry=np.array(sg),**{k:np.array(v) for k,v in templates.items()})
    assert len(truth)==5040 and all(sha(p)==h for p,h in frozen['hashes'].items())
    write(out/'MANIFEST.json',dict(status='COMPLETE',benchmark_sha256=sha(out/'benchmark.npz'),
        frozen_sha256=sha(out/'FROZEN.json'),states=cases,all_local_and_global_labels_match_old_cache=True))


def sha_bytes(raw):return __import__('hashlib').sha256(raw).hexdigest()
if __name__=='__main__':main()
