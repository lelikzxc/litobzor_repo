"""Native wafer geometry baselines. No labels enter feature construction."""
from pathlib import Path
import hashlib,json,time,resource
import joblib,numpy as np
from scipy import ndimage as ndi
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'wafer_features';OUT.mkdir(exist_ok=True)
SEEDS=(3101,3102,3103)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,v):(OUT/name).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def features(a):
    mask=a!=0;fail=a==2;h,w=a.shape;N=int(mask.sum());n=int(fail.sum());yy,xx=np.indices(a.shape)
    cy,cx=yy[mask].mean(),xx[mask].mean();scale=max(np.sqrt((yy[mask]-cy)**2+(xx[mask]-cx)**2).max(),1)
    y=(yy-cy)/scale;x=(xx-cx)/scale;r=np.hypot(y,x);angle=np.arctan2(y,x)
    rb=np.minimum((r*6).astype(int),5);ab=np.minimum(((angle+np.pi)/(2*np.pi)*8).astype(int),7)
    gb=(np.minimum(yy*4//h,3)*4+np.minimum(xx*4//w,3))
    bins=[rb==i for i in range(6)]+[ab==i for i in range(8)]+[gb==i for i in range(16)]
    count=np.array([np.count_nonzero(fail&b) for b in bins],float)
    size=np.array([np.count_nonzero(mask&b) for b in bins],float);p=size/N
    variance=n*p*(1-p)*max(N-n,0)/max(N-1,1)
    z=np.divide(count-n*p,np.sqrt(variance),out=np.zeros_like(count),where=variance>1e-12)
    raw=[np.log1p(h),np.log1p(w),np.log1p(N),N/(h*w),n/N]
    raw.extend(count/max(n,1));raw.extend(size/N)
    # Add deterministic rotation-insensitive distributions without estimated labels.
    raw.extend(sorted((count[6:14]/max(n,1)).tolist()))
    raw.extend(sorted((count[14:]/max(n,1)).tolist()))
    for structure in (ndi.generate_binary_structure(2,1),ndi.generate_binary_structure(2,2)):
        labels,k=ndi.label(fail,structure)
        sizes=np.bincount(labels.ravel())[1:]
        raw.extend([k/max(n,1),float(sizes.max())/max(n,1) if k else 0,
                    float(np.quantile(sizes,.9))/max(n,1) if k else 0])
    if n:
        ys=y[fail];xs=x[fail];rs=r[fail];cov=np.cov(np.stack([ys,xs])) if n>1 else np.zeros((2,2))
        vals=np.linalg.eigvalsh(cov)
        distance=ndi.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1][fail]/scale
        raw.extend([ys.mean(),xs.mean(),ys.std(),xs.std(),cov[0,1],vals[0],vals[1],
                    (vals[1]-vals[0])/max(vals.sum(),1e-12),rs.mean(),rs.std(),*np.quantile(rs,[.1,.5,.9]),
                    ys.max()-ys.min(),xs.max()-xs.min(),distance.mean(),np.quantile(distance,.1)])
    else:raw.extend([0.]*17)
    out=np.array(raw,float);assert np.isfinite(out).all() and np.isfinite(z).all()
    return out,np.concatenate([out,z])
def main():
    assert not (OUT/'FROZEN.json').exists()
    start=time.monotonic()
    with np.load(ROOT/'wafer_data/INPUTS.npz',allow_pickle=False) as f:
        y=f['y'];split=f['split'];lots=f['lots'];flat=f['native_flat'];offset=f['native_offsets'];shape=f['native_shapes']
    rows=[features(flat[offset[i]:offset[i+1]].reshape(shape[i])) for i in range(len(y))]
    raw=np.stack([r for r,z in rows]);rz=np.stack([z for r,z in rows])
    np.savez_compressed(OUT/'FEATURES.npz',raw=raw,raw_z=rz,truth=y,split=split,lots=lots)
    prep=time.monotonic()-start
    params={'n_estimators':256,'max_depth':14,'min_samples_leaf':2,'class_weight':'balanced','n_jobs':2}
    frozen={'script_sha256':sha(__file__),'protocol_sha256':sha(ROOT/'NEW_MINI_PROTOCOL.md'),
        'input_sha256':sha(ROOT/'wafer_data/INPUTS.npz'),'feature_sha256':sha(OUT/'FEATURES.npz'),
        'params':params,'seeds':list(SEEDS),'test_predictions_computed':False,'feature_dimensions':[raw.shape[1],rz.shape[1]]}
    dump('FROZEN.json',frozen);models=[];tests=[];cals=[]
    for name,x in (('raw',raw),('raw_z',rz)):
        ps=[];pcs=[]
        for seed in SEEDS:
            begin=time.monotonic();model=ExtraTreesClassifier(**params,random_state=seed)
            model.fit(x[split==0],y[split==0]);fit_time=time.monotonic()-begin
            cp=model.predict_proba(x[split==1]);cal=f1_score(y[split==1],cp.argmax(1),labels=np.arange(9),average='macro',zero_division=0)
            if cal<.45:
                dump('STATUS.json',{'status':'INCONCLUSIVE_LEARNABILITY','name':name,'seed':seed,'test_evaluated':False,'calibration_F1':cal});return
            begin=time.monotonic();tp=model.predict_proba(x[split==2]);predtime=time.monotonic()-begin
            joblib.dump(model,OUT/f'{name}_{seed}.joblib',compress=3)
            models.append({'method':name,'seed':seed,'calibration_F1':float(cal),'fit_seconds':fit_time,'test_predict_seconds':predtime})
            ps.append(tp);pcs.append(cp);print(json.dumps(models[-1]),flush=True)
        tests.append(ps);cals.append(pcs)
    np.savez_compressed(OUT/'PREDICTIONS.npz',test_probabilities=np.array(tests),cal_probabilities=np.array(cals),
        test_truth=y[split==2],cal_truth=y[split==1],test_lots=lots[split==2],cal_lots=lots[split==1],methods=np.array(['raw','raw_z']),seeds=np.array(SEEDS))
    dump('STATUS.json',{'status':'PREDICTIONS_COMPLETE','test_evaluated':True,'models':models,'feature_preparation_seconds':prep,
        'total_seconds':time.monotonic()-start,'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'script_unchanged':sha(__file__)==frozen['script_sha256']})
if __name__=='__main__':main()
