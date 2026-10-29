"""Read public dataset ZIP directory using validated HTTP byte ranges."""
from pathlib import Path
from urllib.request import Request,urlopen
import io,json,zipfile,hashlib

HERE=Path(__file__).resolve().parent
URL='https://edmond.mpg.de/api/access/datafile/223856'
SIZE=6396523974

class RemoteZip(io.RawIOBase):
    def __init__(self):self.pos=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def read(self,n=-1):
        if n<0:n=SIZE-self.pos
        n=min(n,SIZE-self.pos)
        if n<=0:return b''
        assert n<=25_000_000, f'Unexpected large read {n}'
        request=Request(URL,headers={'Range':f'bytes={self.pos}-{self.pos+n-1}'})
        with urlopen(request,timeout=90) as response:
            content_range=response.headers.get('Content-Range','')
            assert response.status==206 and content_range.startswith(f'bytes {self.pos}-'), (response.status,content_range)
            raw=response.read(n+1)
        assert len(raw)==n,(len(raw),n)
        self.pos+=len(raw);return raw

if __name__=='__main__':
    import sys
    with zipfile.ZipFile(RemoteZip()) as z:
        if len(sys.argv)==1:
            listing=[dict(name=r.filename,size=r.file_size,compressed=r.compress_size,offset=r.header_offset) for r in z.infolist()]
            (HERE/'sem_zip_listing.json').write_text(json.dumps(listing,indent=2)+'\n')
            for r in listing[:40]:print(json.dumps(r),flush=True)
            print('members',len(listing),flush=True)
        else:
            desired=json.loads(Path(sys.argv[1]).read_text());out=HERE/'sem_full';out.mkdir(exist_ok=True);manifest=[]
            for name in desired:
                p=Path(name)
                assert not p.is_absolute() and '..' not in p.parts
                dest=out/p;dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():
                    raw=z.read(name);tmp=dest.with_suffix(dest.suffix+'.part');tmp.write_bytes(raw);tmp.replace(dest)
                raw=dest.read_bytes();row=dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());manifest.append(row);print(json.dumps(row),flush=True)
            (out/'DOWNLOAD.json').write_text(json.dumps(dict(url=URL,persistent_id='doi:10.17617/3.HY5SYN',source_license='CC BY 4.0',files=manifest),indent=2)+'\n')
