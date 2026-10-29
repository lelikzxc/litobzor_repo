"""Sound interval decisions for partially observed binary local inspection.

No model weights, ground-truth labels or hidden native pixels enter policies.
Bounds ignore mixed-block cardinality beyond empty/full, conservatively.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric

BLOCK=8
FOOTPRINT=np.ones((3,3),bool)


def block_sum(image,block=BLOCK):
    h,w=image.shape
    assert h%block==w%block==0
    return image.reshape(h//block,block,w//block,block).sum((1,3))


def expand(grid,block=BLOCK):
    return grid.repeat(block,0).repeat(block,1)


def initial_bounds(counts,block=BLOCK):
    return expand(counts==block*block,block),expand(counts>0,block)


def connection(phase,a,b,threshold,return_fields=False):
    distance=ndi.distance_transform_edt(np.pad(phase,1))[1:-1,1:-1]
    allowed=distance>=threshold
    labels,n=ndi.label(allowed,FOOTPRINT)
    aa=np.unique(labels[a]);bb=np.unique(labels[b]);common=np.intersect1d(aa[aa>0],bb[bb>0])
    connected=bool(len(common))
    if return_fields:
        return connected,distance,labels,int(common[0]) if connected else 0
    return connected


def certificate(lower,upper,a,b,domain,kind,ref):
    """Conservative [violation,missing] bounds; kind=True is forbidden.

    Gated required violations are not monotone: presence and connectivity
    bounds must be combined separately. Unknown cases return class=-1.
    """
    lo=lower&domain;hi=upper&domain
    p_lo=all(np.count_nonzero(lo&r)>=.5*np.count_nonzero(r) for r in [a,b])
    p_hi=all(np.count_nonzero(hi&r)>=.5*np.count_nonzero(r) for r in [a,b])
    threshold=1.5 if kind else .5*ref
    c_lo=connection(lo,a,b,threshold)
    c_hi=connection(hi,a,b,threshold)
    assert not c_lo or c_hi
    if kind:
        v_lo=p_lo and c_lo;v_hi=p_hi and c_hi
    else:
        v_lo=p_lo and not c_hi;v_hi=p_hi and not c_lo
    u_lo=not p_hi;u_hi=not p_lo
    label=2 if u_lo else int(v_lo) if not u_hi and v_lo==v_hi else -1
    return dict(lower=np.array([v_lo,u_lo]),upper=np.array([v_hi,u_hi]),label=label,
        present_lower=p_lo,present_upper=p_hi,connected_lower=c_lo,connected_upper=c_hi)


def native_label(phase,a,b,domain,kind,ref):
    return certificate(phase,phase,a,b,domain,kind,ref)['label']


def priority(lower,upper,counts,a,b,domain,kind,ref,method,random_order):
    """Policy API deliberately accepts neither native image nor target label."""
    unknown=upper&~lower
    candidates=block_sum(unknown&domain)>0
    fraction=counts/64
    entropy=-(fraction*np.log(np.maximum(fraction,1e-12))+(1-fraction)*np.log(np.maximum(1-fraction,1e-12)))
    # Identical strong presence prioritisation for all controls.
    port_score=np.zeros_like(counts,dtype=float)
    for port in [a,b]:
        n=port.sum();low=np.count_nonzero(lower&domain&port);high=np.count_nonzero(upper&domain&port)
        if low<.5*n<=high:
            port_score+=block_sum(unknown&domain&port)/max(n,1)
    if method=='entropy':score=entropy
    elif method=='random':score=random_order
    elif method=='center':
        y,x=np.indices(counts.shape);score=1/(1+(y-(len(counts)-1)/2)**2+(x-(counts.shape[1]-1)/2)**2)
    elif method=='witness':
        threshold=1.5 if kind else .5*ref
        connected,distance,labels,component=connection(upper&domain,a,b,threshold,True)
        score=.001*entropy
        if connected and not connection(lower&domain,a,b,threshold):
            allowed=labels==component
            start=np.argwhere(a&allowed)
            costs=np.where(allowed,1/np.maximum(distance,.5),np.inf)
            mcp=MCP_Geometric(costs,fully_connected=True)
            cost,_=mcp.find_costs([tuple(v) for v in start])
            end_points=np.argwhere(b&allowed)
            endpoint=tuple(end_points[np.argmin(cost[b&allowed])])
            path=mcp.traceback(endpoint)
            path_mask=np.zeros_like(lower);yy,xx=np.array(path).T;path_mask[yy,xx]=True
            # Revealing a path's clearance neighbourhood can settle a width
            # decision even when its centre pixels themselves were known.
            around=ndi.distance_transform_edt(~path_mask)<=np.ceil(threshold)+1
            influence=unknown&domain&around
            weight=influence/(1+np.abs(distance-threshold))
            score+=block_sum(weight)
    else:raise ValueError(method)
    score=score+10000*port_score
    score=np.where(candidates,score,-np.inf)
    return score


class ArrayOracle:
    """Ideal stored-image readout simulator. All native access is logged."""
    def __init__(self,native,counts):
        self.__native=np.asarray(native,bool)
        assert np.array_equal(block_sum(self.__native),counts)
        self.requested=[]

    def reveal(self,tiles):
        result=[]
        for y,x in tiles:
            key=(int(y),int(x));assert key not in self.requested
            self.requested.append(key)
            result.append(self.__native[y*8:(y+1)*8,x*8:(x+1)*8].copy())
        return result


def rollout(oracle,counts,a,b,domain,kind,ref,method,seed=17,budgets=(0,8,32,128)):
    lower,upper=initial_bounds(counts);revealed=np.zeros_like(counts,bool)
    rng=np.random.default_rng(seed);random_order=rng.random(counts.shape)
    result={};queries=0
    for budget in budgets:
        cert=certificate(lower,upper,a,b,domain,kind,ref)
        while queries<budget and cert['label']<0:
            score=priority(lower,upper,counts,a,b,domain,kind,ref,method,random_order)
            score[revealed]=-np.inf
            available=np.flatnonzero(np.isfinite(score.ravel()))
            if not len(available):
                raise AssertionError('Uncertain certificate without observable unknown pixels')
            # Stable ties are determined by coordinate order, never outcomes.
            ranking=available[np.argsort(-score.ravel()[available],kind='stable')]
            chosen=ranking[:min(8,budget-queries)]
            tiles=[tuple(v) for v in np.array(np.unravel_index(chosen,counts.shape)).T]
            for (y,x),pixels in zip(tiles,oracle.reveal(tiles)):
                lower[y*8:(y+1)*8,x*8:(x+1)*8]=pixels
                upper[y*8:(y+1)*8,x*8:(x+1)*8]=pixels
                revealed[y,x]=True
            queries+=len(tiles)
            cert=certificate(lower,upper,a,b,domain,kind,ref)
        result[str(budget)]=dict(label=cert['label'],queries=queries,
            revealed=np.packbits(revealed.ravel()),interval_lower=cert['lower'],interval_upper=cert['upper'])
    return result
