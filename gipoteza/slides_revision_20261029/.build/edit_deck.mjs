import fs from 'node:fs/promises';
import path from 'node:path';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const dir=path.dirname(new URL(import.meta.url).pathname);
const p=await PresentationFile.importPptx(await FileBlob.load(path.join(dir,'source.pptx')));
const records=(await fs.readFile(path.join(dir,'source_inspect.ndjson'),'utf8')).split('\n').filter(Boolean).map(x=>JSON.parse(x));
const get=(slide,name)=>{
 const found=records.filter(x=>x.kind==='textbox'&&x.slide===slide&&x.name===name);
 if(found.length!==1)throw new Error(`Ambiguous target ${slide} ${name}`);
 return p.resolve(found[0].id);
};
const relevance=[
 'Актуальность темы обусловлена необходимостью повышения точности автоматизированного контроля качества в микроэлектронном производстве. Локальные дефекты и нарушения геометрии структур на кремниевых пластинах могут приводить к снижению выхода годных изделий.',
 'Методы глубокого обучения позволяют автоматизировать выделение дефектных областей и определение типа нарушения. При этом высокое попиксельное качество сегментации не гарантирует сохранения связности тонких структур [1]. Небольшие ошибки маски могут менять геометрическое состояние и вычисляемый физический отклик [2].',
 'Поэтому необходимо исследовать метод обучения, учитывающий геометрически значимые ошибки сегментации при классификации дефектов.',
];
const goal='Разработка и исследование метода сегментации и классификации дефектов на изображениях кремниевых пластин с использованием глубокого обучения, учитывающего влияние геометрически значимых ошибок сегментации на определение класса.';
const tasks=[
 'Выбрать базовые методы сегментации и классификации дефектов для сравнения на основе анализа литературы.',
 'Подготовить набор данных и разметку, сформировать обучающую, валидационную и тестовую выборки.',
 'Разработать и обучить модель сегментации с геометрической регуляризацией для последующей классификации геометрических состояний структур.',
 'Исследовать влияние детализации изображений и локальных ошибок сегментации на связность структур и качество классификации.',
 'Сравнить разработанный метод с базовыми решениями по IoU сегментации, Macro-F1 классификации и вычислительным затратам.',
];
const body=get(2,'PlaceHolder 2');
body.text=relevance.map((run,i)=>({runs:[{run,textStyle:{typeface:'Arial',fontSize:'14pt',color:'#000000'}}],
 spaceBefore:i===0?1191:0,spaceAfter:i===2?0:800}));
body.position={left:100.08,top:120.98,width:759.72,height:312};
body.text.style={typeface:'Arial',fontSize:56/3,color:'#000000',autoFit:'none',wrap:'square'};
const aim=get(3,'PlaceHolder 2');
const oldGoal=records.find(x=>x.kind==='textbox'&&x.slide===3&&x.name==='PlaceHolder 2').text;
aim.text.replace(oldGoal,goal);
const taskShape=get(3,'Текст 3');
const oldTasks=records.find(x=>x.kind==='textbox'&&x.slide===3&&x.name==='Текст 3').text;
taskShape.text.replace(oldTasks,tasks.join('\n'));
taskShape.position={left:100.08,top:219.97,width:759.72,height:276};
const slide2=p.slides.getItem(1);
const citations=slide2.shapes.add({geometry:'textbox',name:'Источники актуальности',position:{left:100.08,top:451,width:759.72,height:50},fill:'none',line:{fill:'none',width:0}});
citations.text=[
 [{run:'[1] Shit et al. clDice, CVPR, 2021.',link:{uri:'https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf',isExternal:true}}],
 [{run:'[2] Krygier et al. Nature Communications, 2021.',link:{uri:'https://www.nature.com/articles/s41467-021-25493-8',isExternal:true}}],
];
citations.text.style={typeface:'Arial',fontSize:56/3,color:'#595959',autoFit:'none',wrap:'square',insets:{top:0,right:0,bottom:0,left:9.45}};
slide2.speakerNotes.textFrame.setText([
 '[1] Shit S. et al. clDice – A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation. CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf',
 'Источник обосновывает ограниченность попиксельных метрик и важность сохранения связности сетеподобных структур. Это общий CV-механизм, а не прямой эксперимент на дефектах металлизации ИС.',
 '[2] Krygier M. C. et al. Quantifying the unknown impact of segmentation uncertainty on image-based simulations. Nature Communications 12, 5414 (2021). https://www.nature.com/articles/s41467-021-25493-8',
 'Источник показывает влияние вариаций сегментации на вычисляемые физические величины. Электрический отклик в текущем исследовании рассчитывается по геометрической модели.',
].join('\n\n'));
await fs.writeFile(path.join(dir,'revised_text.json'),JSON.stringify({relevance,goal,tasks},null,2));
const candidate=path.join(dir,'draft.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const snap=await p.inspect({kind:'slide,textbox,notes',maxChars:22000});
await fs.writeFile(path.join(dir,'after_inspect.ndjson'),snap.ndjson);
console.log(snap.ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.getItem(i);
 const img=await p.export({slide,format:'png',scale:1.5});
 await fs.writeFile(path.join(dir,`after-${i+1}.png`),new Uint8Array(await img.arrayBuffer()));
 const layout=await slide.export({format:'layout'});
 await fs.writeFile(path.join(dir,`after-${i+1}.layout.json`),await layout.text());
}
console.log('DRAFT_READY',candidate);
