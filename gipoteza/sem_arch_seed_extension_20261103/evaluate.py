"""Class-only calibration then seed-replication scoring on a previously seen test set."""
import argparse
import json
import time
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
import common as c

@torch.no_grad()
def predict(row,x,device):
    payload=torch.load(c.resolve(row['prediction_model_file']),map_location='cpu',weights_only=True)
    net=c.model.make_plain(24);net.load_state_dict(payload['model']);net=net.requires_grad_(False).to(device).eval()
    tensor=torch.tensor(5*(x-.2),device=device)
    probability=np.concatenate([net(part).sigmoid().cpu().numpy() for part in tensor.split(16)])[:,0]
    assert np.isfinite(probability).all() and ((probability>=0)&(probability<=1)).all()
    return probability

def calibrate(fits,device):
    assert not (c.ROOT/'CALIBRATION.json').exists()
    metadata=c.read(c.OLD/'FINAL_CALIBRATION.json');ids=np.array(metadata['data_indices'],np.int64)
    with np.load(c.DEV/'DATA.npz') as z:x=z['X'][ids];truth=z['label'][ids];frames=z['frame'][ids]
    assert len(ids)==512 and len(np.unique(frames))==32
    thresholds=np.array(c.configuration()['thresholds']);probability=np.empty((2,2,512,12,12),np.float32)
    scores=np.empty((2,2,81));classes=np.empty((2,2,81,512),np.int64);chosen=np.empty((2,2),np.int64);rows=[]
    lookup={(r['architecture'],r['seed']):r for r in fits}
    for a,arch in enumerate(c.ARCHITECTURES):
        for s,seed in enumerate(c.NEW_SEEDS):
            row=lookup[arch,seed];probability[a,s]=predict(row,x,device)
            for t,threshold in enumerate(thresholds):
                classes[a,s,t]=c.hard_classes(probability[a,s]>=threshold)
                confusion=np.bincount(9*truth+classes[a,s,t],minlength=81).reshape(9,9)
                scores[a,s,t]=c.macro(confusion)
            chosen[a,s]=max(range(81),key=lambda t:(float(scores[a,s,t]),-abs(t-40),-t))
            rows.append(dict(row,threshold=float(thresholds[chosen[a,s]]),calibration_F1=float(scores[a,s,chosen[a,s]])))
            print(json.dumps(dict(stage='CALIBRATION',architecture=arch,seed=seed,threshold=rows[-1]['threshold'])),flush=True)
    np.savez_compressed(c.ROOT/'CALIBRATION.npz',data_indices=ids,truth=truth,frames=frames,
        thresholds=thresholds,probability=probability,classes=classes,scores=scores,selected_indices=chosen,
        selected_thresholds=thresholds[chosen],architectures=np.array(c.ARCHITECTURES),seeds=np.array(c.NEW_SEEDS))
    result=dict(status='FROZEN_FOUR_CLASS_ONLY_THRESHOLDS_BEFORE_NEW_TEST_PREDICTIONS',
        rows=rows,calibration_examples=512,calibration_frames=32,data_indices=ids.tolist(),
        source_final_calibration_sha256=c.sha(c.OLD/'FINAL_CALIBRATION.json'),
        arrays_sha256=c.sha(c.ROOT/'CALIBRATION.npz'),masks_or_IoU_used_for_selection=False,
        criterion='MacroF1 then integerhundredths distance to0.50 then lower',
        test_already_seen_in_previous_study=True,new_independent_cohort=False)
    c.save(c.ROOT/'CALIBRATION.json',result)
    return result

def describe(confusion,ious,aps,indices):
    points=c.macro(confusion.sum(2))
    cells={}
    for a,arch in enumerate(c.ARCHITECTURES):
        cells[arch]=dict(mean=dict(mask_F1=float(points[a,indices].mean()),IoU=float(ious[a,indices].mean()),mean_frame_AP=float(aps[a,indices].mean())),
            per_seed=[dict(seed=c.SEEDS[s],mask_F1=float(points[a,s]),IoU=float(ious[a,s].mean()),mean_frame_AP=float(aps[a,s].mean())) for s in indices])
    delta={metric:cells[c.ARCHITECTURES[0]]['mean'][metric]-cells[c.ARCHITECTURES[1]]['mean'][metric] for metric in ['mask_F1','IoU','mean_frame_AP']}
    per_seed=[dict(seed=c.SEEDS[s],mask_F1=float(points[0,s]-points[1,s]),IoU=float(ious[0,s].mean()-ious[1,s].mean()),mean_frame_AP=float(aps[0,s].mean()-aps[1,s].mean())) for s in indices]
    guards=dict(mean_F1=delta['mask_F1']>=.005,mean_AP=delta['mean_frame_AP']>0,
        seed_F1=min(row['mask_F1'] for row in per_seed)>=-.01,IoU_guard=delta['IoU']>=-.002,matched_MACs=True)
    return dict(cells=cells,group_minus_untied=dict(mean=delta,per_seed=per_seed),
        descriptive_point_guards=guards,descriptive_point_pass=all(guards.values()),new_independent_cohort=False)

def analyze(arrays):
    cfg=c.configuration();pred,ious,prob=arrays['mask_predictions'],arrays['IoU'],arrays['probability']
    truth,target,frames=arrays['truth'],arrays['target'],arrays['frames'];values,groups=np.unique(frames,return_inverse=True)
    confusion=np.zeros((2,2,5,24,9,9),np.int32);aps=np.empty((2,5,24));pooled=np.empty((2,5))
    for a in range(2):
        for s in range(5):
            for m in range(2):np.add.at(confusion[m,a,s],(groups,truth,pred[m,a,s]),1)
            for f,frame in enumerate(values):
                ids=frames==frame;aps[a,s,f]=average_precision_score(target[ids].reshape(-1),prob[a,s,ids].reshape(-1))
            pooled[a,s]=average_precision_score(target.reshape(-1),prob[a,s].reshape(-1))
    with np.load(c.OLD/'BOOTSTRAP.npz') as old:
        assert np.array_equal(confusion[:,:,:3],old['confusion'][:,:2])
        assert np.array_equal(aps[:,:3],old['per_frame_AP'][:2])
    rng=np.random.default_rng(cfg['bootstrap_seed']);seed_ids=rng.integers(0,5,(10000,5));frame_ids=rng.integers(0,24,(10000,24))
    boot_f1,boot_ap=np.empty((2,10000)),np.empty((2,10000))
    for a in range(2):
        for start in range(0,10000,256):
            end=min(start+256,10000)
            sampled=confusion[0,a][seed_ids[start:end,:,None],frame_ids[start:end,None,:]].sum(2)
            boot_f1[a,start:end]=c.macro(sampled).mean(1)
        boot_ap[a]=aps[a][seed_ids[:,:,None],frame_ids[:,None,:]].mean((1,2))
    df,da=boot_f1[0]-boot_f1[1],boot_ap[0]-boot_ap[1]
    cif=np.quantile(df,cfg['CI_quantiles']).tolist();cia=np.quantile(da,cfg['CI_quantiles']).tolist()
    analysis={}
    for m,mode in enumerate(c.MODES):
        analysis[mode]={name:describe(confusion[m],ious[m],aps,ids) for name,ids in [('old3',[0,1,2]),('new2',[3,4]),('all5',[0,1,2,3,4])]}
    all5=analysis['final_calibrated']['all5'];new2=analysis['final_calibrated']['new2']
    guards=dict(all5['descriptive_point_guards'],F1_CI_lower=cif[0]>0,AP_CI_lower=cia[0]>0)
    all5.update(CI97_5_F1=cif,CI97_5_AP=cia,descriptive_composite_guards=guards,descriptive_composite_pass=all(guards.values()))
    arrays=dict(confusion=confusion,point_per_seed=c.macro(confusion.sum(3)),per_frame_AP=aps,pooled_pixel_AP=pooled,
        frame_values=values,seed_ids=seed_ids,frame_ids=frame_ids,bootstrap_F1=boot_f1,bootstrap_AP=boot_ap,
        delta_F1=df,delta_AP=da,CI_quantiles=np.array(cfg['CI_quantiles']),seeds=np.array(c.SEEDS))
    return analysis,arrays,new2['descriptive_point_pass'] and all5['descriptive_composite_pass']

def run(device):
    start=time.monotonic();c.verify_lock();assert not (c.ROOT/'RESULT.json').exists()
    training=c.read(c.ROOT/'TRAINING_RESULT.json');assert training['status']=='COMPLETE_FOUR_SEED_REPLICATION_FITS'
    fits=training['fits'];assert len(fits)==4
    for row in fits:
        for k,h in [('model_file','model_sha256'),('prediction_model_file','prediction_model_sha256')]:assert c.sha(c.resolve(row[k]))==row[h]
    calibration=calibrate(fits,device)
    with np.load(c.OLD/'EVAL.npz') as z:x=z['X'].copy();target=z['Y'][:,0].astype(bool);truth=z['label'].copy();frames=z['frame'].copy()
    n=len(x);assert n==6144 and len(np.unique(frames))==24
    with np.load(c.OLD/'PREDICTIONS.npz') as old:
        assert np.array_equal(old['truth'],truth) and np.array_equal(old['target'],target) and np.array_equal(old['frames'],frames)
        probability=np.empty((2,5,n,12,12),np.float32);probability[:,:3]=old['probability'][:2]
        predictions=np.empty((2,2,5,n),np.int64);predictions[:,:,:3]=old['mask_predictions'][:,:2]
        ious=np.empty((2,2,5,n));ious[:,:,:3]=old['IoU'][:,:2]
        packed=np.empty((2,2,5,n,18),np.uint8);packed[:,:,:3]=old['packed_masks'][:,:2]
        thresholds=np.empty((2,2,5));thresholds[:,:,:3]=old['thresholds'][:,:2]
    lookup={(r['architecture'],r['seed']):r for r in calibration['rows']}
    for a,arch in enumerate(c.ARCHITECTURES):
        for s,seed in enumerate(c.NEW_SEEDS,3):
            row=lookup[arch,seed];probability[a,s]=predict(row,x,device)
            for m,threshold in enumerate([row['threshold'],.5]):
                thresholds[m,a,s]=threshold;mask=probability[a,s]>=threshold
                predictions[m,a,s]=c.hard_classes(mask);packed[m,a,s]=np.packbits(mask.reshape(n,144),axis=1)
                union=(mask|target).sum((1,2));intersection=(mask&target).sum((1,2))
                ious[m,a,s]=np.divide(intersection,union,out=np.ones(n),where=union>0)
            print(json.dumps(dict(stage='SEEN_TEST_SEED_REPLICATION',architecture=arch,seed=seed)),flush=True)
    arrays=dict(probability=probability,mask_predictions=predictions,IoU=ious,packed_masks=packed,thresholds=thresholds,
        truth=truth,target=target,frames=frames,architectures=np.array(c.ARCHITECTURES),modes=np.array(c.MODES),seeds=np.array(c.SEEDS))
    np.savez_compressed(c.ROOT/'PREDICTIONS.npz',**arrays)
    analysis,bootstrap,passed=analyze(arrays);np.savez_compressed(c.ROOT/'BOOTSTRAP.npz',**bootstrap)
    c.verify_lock()
    result=dict(status='PASS_SEED_REPLICATION' if passed else 'FAIL_SEED_REPLICATION',analyses=analysis,
        new_seeds=c.NEW_SEEDS,seeds=c.SEEDS,new_fits=4,new_optimizer_updates=8000,calibration_examples=512,
        test_examples=6144,test_frames=24,new_independent_cohort=False,
        old_predictions_preserved_exactly=True,replication_pass=bool(passed),
        new2_cannot_be_rescued_by_all5=True,bootstrap_repeats=10000,bootstrap_seed=202611035,
        CI_quantiles=[.0125,.9875],lock_sha256=c.sha(c.ROOT/'LOCK.json'),calibration_sha256=c.sha(c.ROOT/'CALIBRATION.json'),
        predictions_sha256=c.sha(c.ROOT/'PREDICTIONS.npz'),bootstrap_sha256=c.sha(c.ROOT/'BOOTSTRAP.npz'),seconds=time.monotonic()-start,
        caveats=['same_previously_seen24frames','one_IC_only','mask_derived_geometric_classes','known_C4_operator',
            'different_free_parameters_and_optimizer_geometry','approximate_descriptive_seed_frame_bootstrap','fixed05_diagnostic_only'])
    c.save(c.ROOT/'RESULT.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',action='store_true',required=True);p.add_argument('--device',choices=['mps','cpu'],default='mps');a=p.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):run(a.device)
if __name__=='__main__':main()
