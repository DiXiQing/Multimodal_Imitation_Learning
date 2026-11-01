import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Di/Downloads/Di_ゼミ発表_修正版.pptx'));
const map=new Map((await fs.readFile('tmp/english_deck/translations.tsv','utf8')).trim().split(/\r?\n/).map(s=>s.split('\t')));
const records=(await p.inspect({kind:'textbox',maxChars:1000000})).ndjson.split('\n').filter(s=>s.trim()).map(s=>JSON.parse(s));
let count=0;
for(const r of records){
 if(!r.text || !/[\u3040-\u30ff\u3400-\u9fff]/.test(r.text))continue;
 const translated=r.text.split('\n').map(s=>map.get(s)??s.replaceAll('・', ' / ')).join('\n');
 if(/[\u3040-\u30ff\u3400-\u9fff]/.test(translated))console.log('UNTRANSLATED',r.id,JSON.stringify(translated));
 const obj=p.resolve(r.id);
 obj.text.replace(r.text,translated);
 const size=r.style?.fontSize??20;
 const longest=Math.max(...translated.split('\n').map(s=>s.length));
 const width=r.position.width;
 const lines=translated.split('\n').reduce((n,s)=>n+Math.max(1,Math.ceil(s.length*size*.48/width)),0);
 const available=r.position.height-8;
 const fitted=Math.min(size,available/(lines*1.2));
 if(fitted<size)obj.text.style={fontSize:Math.max(12,fitted)};
 count++;
}
for(const slide of p.slides.items)for(const shape of slide.shapes.items){const a=shape.position;if(a.width<0||a.height<0){shape.position={left:a.left+Math.min(0,a.width),top:a.top+Math.min(0,a.height),width:Math.abs(a.width),height:Math.abs(a.height)};shape.flipVertical=a.height<0;shape.flipHorizontal=a.width<0;}}
await (await PresentationFile.exportPptx(p)).save('tmp/english_deck/candidate.pptx');
console.log('TRANSLATED',count);
await fs.writeFile('tmp/english_deck/translated-inspect.txt',(await p.inspect({kind:'textbox',maxChars:1000000})).ndjson);

