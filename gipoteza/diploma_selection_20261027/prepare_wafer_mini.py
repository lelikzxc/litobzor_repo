"""Outcome-blind native WM811K data preparation, grouped lots and exact/D4 purge."""
from pathlib import Path
import csv,hashlib,json,time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent;PROJECT=ROOT.parent.parent
OUT=ROOT/'wafer_data';OUT.mkdir(exist_ok=True)
CLASSES=['none','Center','Donut','Edge-Loc','Edge-Ring','Loc','Near-full','Random','Scratch']
CAPS=([1200,180,100,350,350,250,45,180,120],[40]*9,[80]*9)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(s):return hashlib.sha256(s.encode()).digest()
def canonical(a):
    variants=[]
    for k in range(4):
        for v in (np.rot90(a,k),np.fliplr(np.rot90(a,k))):
            variants.append(hashlib.sha256(np.array(v.shape,dtype='<u4').tobytes()+v.copy().tobytes()).hexdigest())
    return min(variants)
def split(lot):
    u=int.from_bytes(digest('DIPLOMA-20261027:'+lot)[:8],'big')/2**64
    return 0 if u<.6 else 1 if u<.8 else 2
def main():
    assert not (OUT/'INPUTS.npz').exists(),'Keep existing preparation unchanged'
    start=time.monotonic();csvpath=PROJECT/'datasets/wm811k/labels.csv'
    lookup={s.casefold():i for i,s in enumerate(CLASSES)};pools={}
    with csvpath.open(newline='') as f:
        for row in csv.DictReader(f):
            key=row['failureType'].casefold()
            if key not in lookup:continue
            c=lookup[key];part=split(row['lotName']);pools.setdefault((part,c),[]).append(row)
    image=[];y=[];parts=[];lots=[];records=[];flat=[];offsets=[0];shapes=[]
    seen={};excluded={};contradictions=[]
    for part in range(3):
        for c in range(9):
            count=0
            for row in sorted(pools.get((part,c),[]),key=lambda r:digest(r['filename'])):
                if count>=CAPS[part][c]:break
                p=PROJECT/'datasets/wm811k/images'/row['filename']
                if not p.is_file():excluded['missing_file']=excluded.get('missing_file',0)+1;continue
                a=np.array(Image.open(p).convert('L'))
                if not set(np.unique(a)).issubset({0,1,2}):raise ValueError('Unexpected source encoding')
                key=canonical(a)
                if key in seen:
                    prev=seen[key]
                    if prev['class']!=c:contradictions.append([prev['filename'],row['filename']])
                    excluded['exact_D4']=excluded.get('exact_D4',0)+1;continue
                seen[key]={'class':c,'filename':row['filename']}
                small=np.asarray(Image.fromarray(a).resize((32,32),Image.Resampling.NEAREST))
                image.append(np.stack([small==2,small!=0]).astype(np.uint8))
                y.append(c);parts.append(part);lots.append(row['lotName']);shapes.append(a.shape)
                flat.append(a.ravel());offsets.append(offsets[-1]+a.size)
                records.append({'filename':row['filename'],'lot':row['lotName'],'class':c,'split':part,
                    'native_shape':list(a.shape),'source_file_sha256':sha(p),'exact_D4_sha256':key})
                count+=1
    assert not contradictions,contradictions
    y=np.array(y,np.int64);parts=np.array(parts,np.int8);lot=np.array(lots)
    sizes={str(p):np.bincount(y[parts==p],minlength=9).tolist() for p in range(3)}
    assert min(sizes['1'])>=8 and min(sizes['2'])>=10,sizes
    assert not set(lot[parts==0])&set(lot[parts==1]);assert not set(lot[parts==0])&set(lot[parts==2])
    np.savez_compressed(OUT/'INPUTS.npz',x=np.stack(image),y=y,split=parts,lots=lot,
        native_flat=np.concatenate(flat),native_offsets=np.array(offsets,np.int64),native_shapes=np.array(shapes,np.int16))
    report={'status':'PREPARED_NO_PREDICTIONS','source':'Native PNG from local WM811K','classes':CLASSES,
        'counts':sizes,'lots':{str(p):len(np.unique(lot[parts==p])) for p in range(3)},
        'excluded':excluded,'contradictory_exact_D4_groups':contradictions,
        'csv_sha256':sha(csvpath),'protocol_sha256':sha(ROOT/'NEW_MINI_PROTOCOL.md'),
        'preparer_sha256':sha(Path(__file__)),'input_sha256':sha(OUT/'INPUTS.npz'),
        'records':records,'seconds':time.monotonic()-start,
        'scope':'New grouped development split of previously used public corpus; not new chips or globally untouched test.'}
    (OUT/'MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
