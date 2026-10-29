"""Fixed MPS classifiers; TEST inference only after all calibration gates."""
from pathlib import Path
import hashlib,json,time,resource
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'wafer_cnn';OUT.mkdir(exist_ok=True)
SEEDS=(3101,3102,3103)
ARMS=('binary_natural','support_natural','support_balanced','tristate_balanced','support_noaug')
START=time.monotonic()

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def log(stage,**kw):
    row={'stage':stage,'seconds':time.monotonic()-START,**kw}
    with (OUT/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def macro(y,p):
    c=np.bincount(9*y+p,minlength=81).reshape(9,9);d=c.sum(0)+c.sum(1)
    return float(np.divide(2*np.diag(c),d,out=np.zeros(9),where=d>0).mean())
class Net(nn.Module):
    def __init__(self):
        super().__init__();layers=[];inc=2
        for width in (16,32,64):
            layers += [nn.Conv2d(inc,width,3,stride=2,padding=1),nn.GroupNorm(4,width),nn.GELU()];inc=width
        self.body=nn.Sequential(*layers);self.head=nn.Sequential(nn.Flatten(),nn.Linear(1024,64),nn.GELU(),nn.Linear(64,9))
    def forward(self,x):return self.head(self.body(x))
def inputs(x,arm):
    a=x.copy().astype(np.float32)
    if arm=='binary_natural':a[:,1]=0
    if arm=='tristate_balanced':a[:,0]=.5*(a[:,0]+a[:,1]);a[:,1]=0
    return a
def transform(x,g):
    v=torch.rot90(x,g%4,(-2,-1))
    return v.flip(-1) if g>=4 else v
def probabilities(net,x,g=0):
    rows=[]
    with torch.inference_mode():
        for i in range(0,len(x),256):
            a=transform(torch.tensor(x[i:i+256],device='mps'),g)
            rows.append(net(a).softmax(1).cpu().numpy())
    return np.concatenate(rows)
def main():
    assert torch.backends.mps.is_available(),'MPS required, no CPU fallback'
    assert not (OUT/'FROZEN.json').exists(),'Do not overwrite prospective run'
    torch.set_num_threads(2)
    with np.load(ROOT/'wafer_data/INPUTS.npz',allow_pickle=False) as f:
        x=f['x'];y=f['y'];split=f['split'];lots=f['lots']
    train=np.flatnonzero(split==0);cal=np.flatnonzero(split==1);test=np.flatnonzero(split==2)
    frozen={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'script_sha256':sha(__file__),
        'protocol_sha256':sha(ROOT/'NEW_MINI_PROTOCOL.md'),'input_sha256':sha(ROOT/'wafer_data/INPUTS.npz'),
        'manifest_sha256':sha(ROOT/'wafer_data/MANIFEST.json'),'device':'mps','seeds':list(SEEDS),
        'arms':list(ARMS),'steps':600,'test_predictions_computed':False,
        'parameters':sum(p.numel() for p in Net().parameters()),'torch':torch.__version__}
    dump('FROZEN.json',frozen);log('frozen',parameters=frozen['parameters'])
    summaries=[]
    for seed in SEEDS:
        for arm in ARMS:
            begin=time.monotonic();torch.manual_seed(seed);net=Net().to('mps')
            opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
            tx=torch.tensor(inputs(x[train],arm),device='mps');ty=torch.tensor(y[train],device='mps')
            rng=np.random.default_rng(seed+100);grng=np.random.default_rng(seed+200)
            perclass=[np.flatnonzero(y[train]==c) for c in range(9)]
            balanced=arm in ('support_balanced','tristate_balanced','support_noaug')
            history=[];net.train()
            for step in range(600):
                if balanced:
                    cs=rng.integers(0,9,32);ids=np.array([rng.choice(perclass[c]) for c in cs])
                else:ids=rng.integers(0,len(train),32)
                g=int(grng.integers(0,8));batch=tx[torch.tensor(ids,device='mps')]
                if arm!='support_noaug':batch=transform(batch,g)
                target=ty[torch.tensor(ids,device='mps')]
                loss=nn.functional.cross_entropy(net(batch),target)
                opt.zero_grad();loss.backward();opt.step()
                if (step+1)%200==0:
                    value=float(loss.detach().cpu());history.append({'step':step+1,'loss':value})
                    log('training',seed=seed,arm=arm,step=step+1,loss=value)
            torch.mps.synchronize();net.eval();p=probabilities(net,inputs(x[cal],arm))
            score=macro(y[cal],p.argmax(1));path=OUT/f'{arm}_{seed}.pt'
            torch.save(net.cpu().state_dict(),path)
            np.savez_compressed(OUT/f'{arm}_{seed}_CAL.npz',p=p,truth=y[cal],source_indices=cal,lots=lots[cal])
            row={'seed':seed,'arm':arm,'calibration_F1':score,'training_and_cal_seconds':time.monotonic()-begin,
                'checkpoint_sha256':sha(path),'history':history}
            summaries.append(row);dump('CALIBRATION.json',summaries);log('calibration',**{k:v for k,v in row.items() if k not in ('history','checkpoint_sha256')})
            del net,opt,tx,ty;torch.mps.empty_cache()
    if min(r['calibration_F1'] for r in summaries)<.45:
        dump('STATUS.json',{'status':'INCONCLUSIVE_LEARNABILITY','test_evaluated':False,
            'minimum_calibration_F1':min(r['calibration_F1'] for r in summaries)});return
    arrays=[];inference=[]
    for seed in SEEDS:
        perarm=[]
        for arm in ARMS:
            net=Net();net.load_state_dict(torch.load(OUT/f'{arm}_{seed}.pt',weights_only=True));net.to('mps').eval()
            tx=inputs(x[test],arm);begin=time.monotonic()
            orientation=np.stack([probabilities(net,tx,g) for g in range(8)])
            torch.mps.synchronize();elapsed=time.monotonic()-begin
            perarm.append(orientation);inference.append({'seed':seed,'arm':arm,'eight_orientations_seconds':elapsed,'test_images':len(tx)})
            del net;torch.mps.empty_cache()
        arrays.append(perarm);log('test_predictions_saved',seed=seed)
    np.savez_compressed(OUT/'TEST_PREDICTIONS.npz',probabilities=np.array(arrays),truth=y[test],lots=lots[test],
        source_indices=test,arms=np.array(ARMS),seeds=np.array(SEEDS))
    dump('TIMING.json',inference)
    dump('STATUS.json',{'status':'PREDICTIONS_COMPLETE','test_evaluated':True,'seconds':time.monotonic()-START,
        'script_unchanged':sha(__file__)==frozen['script_sha256'],'maxrss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory()})
    log('complete_predictions')
if __name__=='__main__':main()
