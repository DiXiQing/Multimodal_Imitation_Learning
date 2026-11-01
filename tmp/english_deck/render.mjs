import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/MineApp/Code/Multimodal_Imitation_Learning/tmp/english_deck/candidate.pptx'));
console.log('loaded');
const b=await p.export({slide:p.slides.items[3],format:'png',scale:1});
console.log('rendered');
await fs.writeFile('tmp/english_deck/translated4.png',new Uint8Array(await b.arrayBuffer()));


