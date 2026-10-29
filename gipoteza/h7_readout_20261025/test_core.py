import itertools
import numpy as np
from core import certificate,native_label,initial_bounds,block_sum,ArrayOracle,rollout,priority


def test_exact_sums_and_bounds():
    x=np.zeros((16,16),bool);x[:8,:8]=True;x[8,8]=True
    counts=block_sum(x);lo,hi=initial_bounds(counts)
    assert counts.tolist()==[[64,0],[0,1]]
    assert np.all(lo<=x) and np.all(x<=hi)


def test_all_completions_are_enclosed_even_with_missing_gates():
    a=np.zeros((6,6),bool);b=a.copy();a[2:4,0:2]=True;b[2:4,4:6]=True
    base=np.zeros((6,6),bool);base[2:4,:]=True
    unknown=[(2,0),(3,1),(2,3),(3,3),(2,5)]
    lo=base.copy();hi=base.copy()
    for pos in unknown:lo[pos]=False;hi[pos]=True
    for k in [False,True]:
        for ref in [2.,4.,6.]:
            cert=certificate(lo,hi,a,b,np.ones_like(a),k,ref)
            for bits in itertools.product([False,True],repeat=len(unknown)):
                x=base.copy()
                for pos,value in zip(unknown,bits):x[pos]=value
                label=native_label(x,a,b,np.ones_like(a),k,ref)
                bits2=np.array([label==1,label==2])
                assert np.all(cert['lower']<=bits2) and np.all(bits2<=cert['upper'])
                if cert['label']>=0:assert cert['label']==label


def test_required_gated_violation_not_monotone_counterexample():
    a=np.zeros((8,8),bool);b=a.copy();a[3:5,:2]=True;b[3:5,6:]=True
    lo=np.zeros_like(a);hi=np.ones_like(a);middle=a|b
    domain=np.ones_like(a)
    assert native_label(lo,a,b,domain,False,2)==2
    assert native_label(hi,a,b,domain,False,2)==0
    assert native_label(middle,a,b,domain,False,2)==1
    cert=certificate(lo,hi,a,b,domain,False,2)
    assert cert['label']==-1 and cert['upper'].all()


def test_policy_does_not_receive_hidden_pixels_and_reveal_is_exact():
    x=np.zeros((16,16),bool);x[4:12,2:14]=True
    a=np.zeros_like(x);b=a.copy();a[6:10,2:4]=True;b[6:10,12:14]=True
    counts=block_sum(x);oracle=ArrayOracle(x,counts)
    for method in ['random','entropy','center','witness']:
        result=rollout(ArrayOracle(x,counts),counts,a,b,np.ones_like(x),False,4,method,budgets=(0,4))
        assert result['4']['label']==native_label(x,a,b,np.ones_like(x),False,4)
    import inspect
    assert not {'native','truth','label'} & set(inspect.signature(priority).parameters)
