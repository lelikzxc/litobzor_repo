#!/usr/bin/env python3
"""Fetch CC BY 4.0 REFICS metal triples via HTTP ranges, never full multi-GB ZIPs.
Only Python standard library is needed for downloading. Image QA uses Pillow/NumPy.
Run: python3 download_refics_subset.py --per-node 32
"""
import argparse, collections, hashlib, io, json, pathlib, re, struct, time, urllib.request, zipfile
BASE = 'https://trust-hub.org/downloads/resource/data/refics/'
ROOT = pathlib.Path(__file__).resolve().parent
LIMIT = 250_000_000
DOWNLOADS = []
TOTAL = 0

def get_range(url, start, end, cache):
    global TOTAL
    key = f'{start}_{end}'
    target = cache / (key + '.bin')
    meta = cache / (key + '.json')
    if target.exists() and meta.exists():
        return target.read_bytes(), json.loads(meta.read_text())
    request = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding':'identity'})
    with urllib.request.urlopen(request, timeout=45) as response:
        if response.status != 206:
            raise RuntimeError(f'Refusing non-range response: {response.status} {url}')
        length = int(response.headers['Content-Length'])
        if TOTAL + length > LIMIT:
            raise RuntimeError('Transfer budget exceeded')
        data = response.read(length + 1)
        if len(data) != length: raise RuntimeError('Truncated HTTP range')
        record = {'url':url, 'range':response.headers['Content-Range'], 'bytes':len(data),
                  'etag':response.headers.get('ETag'), 'last_modified':response.headers.get('Last-Modified'),
                  'sha256':hashlib.sha256(data).hexdigest()}
    TOTAL += len(data)
    target.write_bytes(data);meta.write_text(json.dumps(record,indent=2)+'\n')
    DOWNLOADS.append(record)
    return data, record

class RemoteZip(io.RawIOBase):
    def __init__(self, url, node):
        self.url=url;self.pos=0;self.buffers=[];self.cache=ROOT/'range_cache'/node
        self.cache.mkdir(parents=True,exist_ok=True)
        data, record=get_range(url,0,0,self.cache)
        self.size=int(record['range'].split('/')[-1])
    def seekable(self): return True
    def readable(self): return True
    def tell(self):return self.pos
    def seek(self, offset, whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        return self.pos
    def prefetch(self,start,end):
        data,_=get_range(self.url,start,end,self.cache)
        self.buffers=[(start,end,data)]
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        if not n:return b''
        if n > LIMIT:raise RuntimeError('Refusing full archive read')
        end=min(self.size,self.pos+n)-1
        for first,last,buffer in self.buffers:
            if first<=self.pos and end<=last:
                data=buffer[self.pos-first:end-first+1]
                self.pos=end+1
                return data
        data,_=get_range(self.url,self.pos,end,self.cache)
        self.pos=end+1
        return data

def parse_name(name):
    stem=pathlib.PurePosixPath(name).stem
    parts=stem.split('_')
    if len(parts)<4 or not parts[0].isdigit():return None
    if parts[2]!='M1':return None
    kind='layout' if parts[-1]=='gt' else 'mask' if len(parts)==4 else 'sem' if len(parts)==10 else None
    if kind is None:return None
    return {'uid':parts[0], 'node_nm':int(parts[1]), 'layer':parts[2], 'magnification':parts[3], 'kind':kind, 'parts':parts}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--per-node',type=int,default=32);ap.add_argument('--inspect',action='store_true');args=ap.parse_args()
    rows=[];inventories=[]
    for node in ['32','90']:
        url=BASE+node+'_M1.zip'
        remote=RemoteZip(url,node)
        with zipfile.ZipFile(remote) as archive:
            entries=archive.infolist();names=set(archive.namelist())
            counts=collections.Counter();mags=collections.Counter();uids=set();sem=[]
            for entry in entries:
                parsed=parse_name(entry.filename)
                if parsed:
                    counts[parsed['kind']]+=1
                    if parsed['kind']=='sem':
                        sem.append((entry,parsed));mags[parsed['magnification']]+=1;uids.add(parsed['uid'])
            inv={'node_nm':int(node),'url':url,'archive_entries':len(entries),'kind_counts':dict(counts),'sem_per_mag':dict(mags),'unique_sem_uids':len(uids),'first_entries':[e.filename for e in entries[:12]]}
            inventories.append(inv);print(json.dumps(inv),flush=True)
            if args.inspect:continue
            # Every selected base UID appears only once. Spread across magnification groups.
            groups=collections.defaultdict(list)
            for entry,parsed in sem:groups[parsed['magnification']].append((entry,parsed))
            for mag in groups:groups[mag].sort(key=lambda x:(int(x[1]['uid']),x[0].filename))
            selected=[];seen=set();mkeys=sorted(groups,key=float)
            while len(selected)<args.per_node:
                progressed=False
                for mag in mkeys:
                    while groups[mag] and groups[mag][0][1]['uid'] in seen:groups[mag].pop(0)
                    if not groups[mag]:continue
                    entry,parsed=groups[mag].pop(0);seen.add(parsed['uid']);selected.append((entry,parsed));progressed=True
                    if len(selected)>=args.per_node:break
                if not progressed:break
            for entry, parsed in selected:
                base='_'.join(parsed['parts'][:4]);folder=str(pathlib.PurePosixPath(entry.filename).parent)
                image_names={'sem':entry.filename,'mask':folder+'/'+base+'.png','layout':folder+'/'+base+'_gt.png'}
                # Some releases put GT in distinct paths; require an exact unambiguous basename.
                for kind,name in image_names.copy().items():
                    if name not in names:
                        matches=[n for n in names if pathlib.PurePosixPath(n).name==pathlib.PurePosixPath(name).name]
                        if len(matches)!=1:raise RuntimeError(f'Missing/ambiguous paired {kind}: {name}; matches={matches[:5]}')
                        image_names[kind]=matches[0]
                paired_infos=[archive.getinfo(name) for name in image_names.values()]
                first=min(e.header_offset for e in paired_infos)
                last=max(e.header_offset+30+len(e.filename.encode())+len(e.extra)+e.compress_size+1024 for e in paired_infos)
                remote.prefetch(first,last)
                row={k:v for k,v in parsed.items() if k not in ['parts','kind']}
                row.update({'dataset':'REFICS','modality':'synthetic_SEM','source_layout_id':f'AES_M1_{node}nm','layout_group':f'REFICS_{node}_M1_UID{parsed["uid"]}','license':'CC-BY-4.0','archive_url':url,'files':{}})
                row['generation_parameters']=dict(zip(['shot','foreground_mean','foreground_std','background_mean','background_std','dwell_time'],parsed['parts'][4:]))
                row['generation_parameters']['dwell_time_us']=float(row['generation_parameters']['dwell_time'].replace('$','.'))
                source_layout=ROOT/'tools_and_doc'/'layout_files'/f'design_M1_{node}nm.png'
                if source_layout.exists():
                    source_bytes=source_layout.read_bytes()
                    row.update(source_layout_file=str(source_layout.relative_to(ROOT)),source_layout_sha256=hashlib.sha256(source_bytes).hexdigest(),source_layout_pixels=list(struct.unpack('>II',source_bytes[16:24])))
                row['relative_magnification_factor_in_released_tool']=float(parsed['magnification'])/100
                row['physical_pixel_size_nm']=None
                for kind,name in image_names.items():
                    data=archive.read(name);dest=ROOT/'subset'/node/kind/pathlib.PurePosixPath(name).name
                    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                    row['files'][kind]={'path':str(dest.relative_to(ROOT)),'member':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                rows.append(row)
                print(f'Fetched {node}nm UID={parsed["uid"]} mag={parsed["magnification"]}',flush=True)
    (ROOT/'refics_archive_inventory.json').write_text(json.dumps(inventories,indent=2)+'\n')
    if rows:(ROOT/'refics_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
    (ROOT/'download_session.json').write_text(json.dumps({'new_http_bytes':TOTAL,'max_http_bytes':LIMIT,'requests':DOWNLOADS},indent=2)+'\n')
    print('New bytes',TOTAL,'triples',len(rows),flush=True)
if __name__=='__main__':main()
