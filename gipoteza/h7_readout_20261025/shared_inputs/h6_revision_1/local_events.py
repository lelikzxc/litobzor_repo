"""Target-only local event construction and native-width geometric supervision."""
import hashlib
import itertools
import math

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import reconstruction, skeletonize

FOOTPRINT = np.ones((3, 3), bool)


def key(*parts):
    return hashlib.sha256('/'.join(map(str, parts)).encode()).hexdigest()


def widest_radius(phase, a, b):
    """Exact max-min path clearance between regions, in native pixels (CC8)."""
    if not (phase & a).any() or not (phase & b).any():
        return 0.
    distance = ndi.distance_transform_edt(np.pad(phase,1,constant_values=False))[1:-1,1:-1].astype(np.float32)
    seed = np.zeros_like(distance)
    seed[a] = distance[a]
    reached = reconstruction(seed, distance, method='dilation', footprint=FOOTPRINT)
    return float(reached[b].max(initial=0))


def crop_box(y, x, shape, side=256):
    yy = min(max(int(y)-side//2, 0), shape[0]-side)
    xx = min(max(int(x)-side//2, 0), shape[1]-side)
    return yy, xx, side, side


def disk(shape, centre, radius=6):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy-centre[0])**2+(xx-centre[1])**2 <= radius**2


def local_spec(target, target_hash, required_cap=24, forbidden_cap=32):
    # PIL mode-1 arrays can contain True bytes 255. Zhang's Cython lookup
    # requires canonical 0/1 bool storage and otherwise can segfault.
    target = np.not_equal(target, False)
    labels, count = ndi.label(target, FOOTPRINT)
    depth = ndi.distance_transform_edt(target)
    exterior_distance, nearest = ndi.distance_transform_edt(~target, return_indices=True)
    owner = labels[tuple(nearest)]
    del nearest
    skeleton = skeletonize(target)
    degree = ndi.convolve(skeleton.astype(np.uint8), np.ones((3,3), np.uint8))-skeleton
    away = ndi.distance_transform_edt(~(skeleton & (degree != 2)))
    yy, xx = np.where(skeleton & (degree == 2) & (depth >= 8) & (away >= 48))
    bins = {}
    for y, x in zip(yy.tolist(), xx.tolist()):
        cell = (y//64, x//64)
        rank = ((y % 64 - 32)**2+(x % 64 - 32)**2, y, x)
        if cell not in bins or rank < bins[cell][0]: bins[cell] = (rank, y, x)

    def walk(start, first):
        prev, cur = start, first; length = math.dist(start, first)
        while length < 48:
            y, x = cur
            if degree[y,x] != 2: return None
            neighbours = [(y+dy,x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                          if (dy or dx) and skeleton[y+dy,x+dx] and (y+dy,x+dx) != prev]
            if len(neighbours) != 1: return None
            following = neighbours[0]; length += math.dist(cur, following); prev, cur = cur, following
        return cur

    required = []
    for _, y, x in sorted(bins.values(), key=lambda z:key(target_hash,'req',z[1],z[2])):
        if min(y,x,target.shape[0]-y,target.shape[1]-x) < 128: continue
        neighbours = [(y+dy,x+dx) for dy in (-1,0,1) for dx in (-1,0,1)
                      if (dy or dx) and skeleton[y+dy,x+dx]]
        if len(neighbours) != 2: continue
        ends = [walk((y,x), q) for q in neighbours]
        if any(q is None for q in ends): continue
        if min(depth[q] for q in ends) < 8: continue
        if any((y-e['centre'][0])**2+(x-e['centre'][1])**2 < 64**2 for e in required): continue
        box = crop_box(y,x,target.shape); y0,x0,h,w = box; sl=np.s_[y0:y0+h,x0:x0+w]
        a,b = [disk((h,w),(q[0]-y0,q[1]-x0)) for q in ends]
        ci=int(labels[y,x]); domain=(owner[sl]==ci)&(exterior_distance[sl]<=16)
        q_ref=widest_radius(target[sl]&domain,a,b)
        if q_ref < 4: continue
        required.append(dict(id=f'req:{y}:{x}',kind='required',centre=[y,x],box=box,
                             points=[list(q)for q in ends],components=[ci],q_ref=q_ref,
                             a=a,b=b,domain=domain))
        if len(required)>=required_cap:break

    ridge = np.zeros(target.shape,bool)
    pair_a=np.zeros(target.shape,np.int32);pair_b=np.zeros(target.shape,np.int32)
    for dy,dx in [(0,1),(1,0)]:
        region=np.s_[:target.shape[0]-dy,:target.shape[1]-dx]
        other=np.s_[dy:,dx:]
        different=(owner[region]!=owner[other])&(owner[region]>0)&(owner[other]>0)
        ridge[region]|=different
        pair_a[region]=np.where(different,np.minimum(owner[region],owner[other]),pair_a[region])
        pair_b[region]=np.where(different,np.maximum(owner[region],owner[other]),pair_b[region])
    yy,xx=np.where(ridge & ~target & (exterior_distance>=4)&(exterior_distance<=80))
    bins={}
    for y,x in zip(yy.tolist(),xx.tolist()):
        ca,cb=int(pair_a[y,x]),int(pair_b[y,x]); cell=(ca,cb,y//64,x//64)
        rank=((y%64-32)**2+(x%64-32)**2,y,x)
        if cell not in bins or rank<bins[cell][0]:bins[cell]=(rank,y,x,ca,cb)
    candidates=sorted(bins.values(),key=lambda z:key(target_hash,'forb',z[1],z[2]))
    forbidden=[]
    for _,y,x,ca,cb in candidates:
        if min(y,x,target.shape[0]-y,target.shape[1]-x)<128:continue
        if any((y-e['centre'][0])**2+(x-e['centre'][1])**2<64**2 for e in forbidden):continue
        box=crop_box(y,x,target.shape);y0,x0,h,w=box;sl=np.s_[y0:y0+h,x0:x0+w]
        a=(labels[sl]==ca)&(depth[sl]>=4);b=(labels[sl]==cb)&(depth[sl]>=4)
        if min(a.sum(),b.sum())<16:continue
        domain=(owner[sl]==ca)|(owner[sl]==cb)
        forbidden.append(dict(id=f'forb:{y}:{x}',kind='forbidden',centre=[y,x],box=box,
                              components=[ca,cb],q_ref=0.,a=a,b=b,domain=domain))
        if len(forbidden)>=forbidden_cap:break
    return required+forbidden,dict(target_components=count,required=len(required),forbidden=len(forbidden))


def observe(produced, event):
    y,x,h,w=event['box'];phase=produced[y:y+h,x:x+w]&event['domain']
    a,b=event['a'],event['b']
    fractions=[float((phase&region).sum()/region.sum())for region in [a,b]]
    missing=min(fractions)<.5
    q=widest_radius(phase,a,b)
    # A required-width violation includes severe narrowing; it is not renamed a
    # purely topological split. Missing is always retained as a separate label.
    threshold=.5*event['q_ref'] if event['kind']=='required' else 1.5
    margin=q-threshold if event['kind']=='required' else threshold-q
    violation=not missing and (margin<0 if event['kind']=='required' else margin<=0)
    return dict(q=q,margin=margin,missing=bool(missing),violation=bool(violation),presence=fractions)


def public_event(e):
    return {k:v for k,v in e.items() if k not in ['a','b','domain']}
