import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const folder = path.dirname(new URL(import.meta.url).pathname);
const p = await PresentationFile.importPptx(await FileBlob.load(path.join(folder, 'source.pptx')));
const snapshot = await p.inspect({kind:'slide,textbox,image,layout',maxChars:24000});
await fs.writeFile(path.join(folder, 'source_inspect.ndjson'),snapshot.ndjson);
console.log(snapshot.ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.getItem(i);
 const image=await p.export({slide,format:'png',scale:1.5});
 await fs.writeFile(path.join(folder,`before-${i+1}.png`),new Uint8Array(await image.arrayBuffer()));
 const layout=await slide.export({format:'layout'});
 await fs.writeFile(path.join(folder,`before-${i+1}.layout.json`),await layout.text());
}
console.log('SOURCE_RENDERED',p.slides.items.length);
