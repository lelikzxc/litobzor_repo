"""Independent grouped-confusion-matrix audit of saved mini outcomes."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from sklearn.metrics import f1_score
import torch
from wafer_cnn_mini import Net,inputs

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'wafer_analysis'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def main():
    start=time.monotonic();torch.set_num_threads(2)
    result=read(OUT/'RESULT.json');cf=read(ROOT/'wafer_cnn/FROZEN.json');ff=read(ROOT/'wafer_features/FROZEN.json')
    hashes={
      'cnn_script':sha(ROOT/'wafer_cnn_mini.py')==cf['script_sha256'],
      'cnn_protocol':sha(ROOT/'NEW_MINI_PROTOCOL.md')==cf['protocol_sha256'],
      'cnn_input':sha(ROOT/'wafer_data/INPUTS.npz')==cf['input_sha256'],
      'feature_script':sha(ROOT/'wafer_feature_mini.py')==ff['script_sha256'],
      'feature_protocol':sha(ROOT/'NEW_MINI_PROTOCOL.md')==ff['protocol_sha256'],
      'feature_input':sha(ROOT/'wafer_data/INPUTS.npz')==ff['input_sha256'],
      'feature_values':sha(ROOT/'wafer_features/FEATURES.npz')==ff['feature_sha256'],
      'analysis_script':sha(ROOT/'analyse_wafer_mini.py')==result['analysis_script_sha256'],
      'cnn_predictions':sha(ROOT/'wafer_cnn/TEST_PREDICTIONS.npz')==result['CNN_predictions_sha256'],
      'feature_predictions':sha(ROOT/'wafer_features/PREDICTIONS.npz')==result['feature_predictions_sha256'],
    }
    with np.load(ROOT/'wafer_cnn/TEST_PREDICTIONS.npz',allow_pickle=False) as f:
        prob=f['probabilities'];y=f['truth'];lots=f['lots'];seeds=f['seeds'];source=f['source_indices'];arms=f['arms'].tolist()
    with np.load(ROOT/'wafer_features/PREDICTIONS.npz',allow_pickle=False) as f:
        assert np.array_equal(y,f['test_truth']) and np.array_equal(lots,f['test_lots'])
        feature=f['test_probabilities'].argmax(-1)
    pred=prob.argmax(-1);tta=prob[:,2].mean(1).argmax(-1)
    with np.load(ROOT/'wafer_data/INPUTS.npz',allow_pickle=False) as f:
        allx=f['x'];ally=f['y'];sp=f['split'];alllots=f['lots']
    split_pass=not set(alllots[sp==0])&set(alllots[sp==1]) and not set(alllots[sp==0])&set(alllots[sp==2]) and not set(alllots[sp==1])&set(alllots[sp==2])
    manifest=read(ROOT/'wafer_data/MANIFEST.json');keys=[r['exact_D4_sha256'] for r in manifest['records']]
    source_unique=len(keys)==len(set(keys))
    source_hash_ok=True
    for r in manifest['records']:
        p=ROOT.parent.parent/'datasets/wm811k/images'/r['filename']
        source_hash_ok &= sha(p)==r['source_file_sha256']
    # Re-evaluate saved weights on fixed examples with CPU, independent from MPS inference.
    maximum=0.;all_classes=True
    summaries=read(ROOT/'wafer_cnn/CALIBRATION.json')
    for row in summaries:
        s=list(seeds).index(row['seed']);a=arms.index(row['arm']);p=ROOT/f"wafer_cnn/{row['arm']}_{row['seed']}.pt"
        hashes[f"checkpoint_{row['arm']}_{row['seed']}"]=sha(p)==row['checkpoint_sha256']
        net=Net();net.load_state_dict(torch.load(p,weights_only=True));net.eval()
        with torch.inference_mode():
            pp=net(torch.tensor(inputs(allx[source[:24]],row['arm']))).softmax(1).numpy()
        maximum=max(maximum,float(abs(pp-prob[s,a,0,:24]).max()))
        all_classes &= np.array_equal(pp.argmax(1),pred[s,a,0,:24])
    names=[];values=[]
    for s in range(3):
        for a in range(5):
            for g in range(8):names.append(('cnn',s,a,g));values.append(pred[s,a,g])
        names.append(('tta',s));values.append(tta[s])
    for a in range(2):
        for s in range(3):names.append(('feature',a,s));values.append(feature[a,s])
    values=np.array(values);lookup={k:i for i,k in enumerate(names)}
    unique,gid=np.unique(lots,return_inverse=True);G=len(unique);K=len(values)
    matrices=np.zeros((G,K,9,9),np.int32)
    for j,p in enumerate(values):np.add.at(matrices,(gid,np.full(len(y),j),y,p),1)
    rng=np.random.default_rng(310127);sample=rng.integers(0,G,(2000,G));weights=np.zeros((2000,G),np.int32)
    for i in range(2000):weights[i]=np.bincount(sample[i],minlength=G)
    counts=(weights.astype(float)@matrices.reshape(G,-1)).reshape(2000,K,9,9)
    denominator=counts.sum(-1)+counts.sum(-2);numerator=2*np.diagonal(counts,axis1=-2,axis2=-1)
    scores=np.divide(numerator,denominator,out=np.zeros_like(numerator),where=denominator>0).mean(-1)
    point=np.array([f1_score(y,p,labels=np.arange(9),average='macro',zero_division=0) for p in values])
    def diff(keys_a,keys_b):return scores[:,[lookup[k] for k in keys_a]].mean(1)-scores[:,[lookup[k] for k in keys_b]].mean(1)
    draws={}
    for key,a,b in (('C1',1,0),('C2',2,1)):
        draws[key]=diff([('cnn',s,a,0) for s in range(3)],[('cnn',s,b,0) for s in range(3)])
    draws['C5']=diff([('cnn',s,2,g) for s in range(3) for g in range(8)],[('cnn',s,4,g) for s in range(3) for g in range(8)])
    draws['C7']=diff([('feature',1,s) for s in range(3)],[('feature',0,s) for s in range(3)])
    draws['C6']=diff([('feature',0,s) for s in range(3)],[('cnn',s,2,0) for s in range(3)])
    draws['C3_F1']=diff([('tta',s) for s in range(3)],[('cnn',s,2,0) for s in range(3)])
    with np.load(OUT/'SELECTIVE_MASKS.npz',allow_pickle=False) as f:
        keep=f['accepted'];rand=f['random_accepted'];bad=f['predictions']!=y
    risks=[]
    for s in range(3):
        den=np.bincount(gid,weights=keep[s],minlength=G);num=np.bincount(gid,weights=keep[s]&bad[s],minlength=G)
        own=(weights@num)/(weights@den)
        others=[]
        for m in rand[s]:
            d=np.bincount(gid,weights=m,minlength=G);n=np.bincount(gid,weights=m&bad[s],minlength=G)
            others.append((weights@n)/(weights@d))
        risks.append(np.mean(others,axis=0)-own)
    draws['C4']=np.mean(risks,axis=0)
    maximum_ci_error=0.;intervals={}
    for key,arr in draws.items():
        ci=np.quantile(arr,[.025,.975]);intervals[key]={'descriptive_CI95':ci.tolist(),
            'Bonferroni_7_questions_CI':np.quantile(arr,[.025/7,1-.025/7]).tolist()}
        if key in ('C1','C2','C4','C5','C7'):
            maximum_ci_error=max(maximum_ci_error,float(abs(ci-np.array(result['questions'][key]['paired_lot_bootstrap_CI95'])).max()))
    np.savez_compressed(OUT/'BOOTSTRAP_AUDIT.npz',**draws)
    # All stored source claims and reported F1 values are checked against sklearn.
    f1_error=0.
    for a,arm in enumerate(arms):
        f1_error=max(f1_error,max(abs(point[lookup[('cnn',s,a,0)]]-result['single_orientation_CNN_F1'][arm][s]) for s in range(3)))
    gates={key:bool(all(v['gates'].values())) for key,v in result['questions'].items()}
    allstatus=all((result['questions'][k]['status']=='PASS_MINI')==v for k,v in gates.items())
    audit={'status':'PASS_AUDIT' if all(hashes.values()) and split_pass and source_unique and source_hash_ok and all_classes and maximum<1e-5 and maximum_ci_error<1e-12 and f1_error<1e-12 and allstatus else 'FAIL_AUDIT',
        'hashes':hashes,'lot_split_disjoint':bool(split_pass),'exact_D4_unique':source_unique,'source_PNG_hashes_match':bool(source_hash_ok),
        'CPU_MPS_max_probability_error':maximum,'CPU_MPS_all_fixture_classes_equal':bool(all_classes),
        'independent_sklearn_point_error':f1_error,'independent_group_confusion_CI_max_error':maximum_ci_error,
        'reported_gate_statuses_consistent':allstatus,'intervals':intervals,'seconds':time.monotonic()-start,
        'scope':'Independent arithmetic/provenance audit; no model fitting. Simultaneous intervals supplementary, original gates unchanged.'}
    (OUT/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('hashes','intervals')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
