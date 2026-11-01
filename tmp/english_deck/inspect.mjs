import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Di/Downloads/Di_ゼミ発表_修正版.pptx'));
console.log((await p.inspect({kind:'slide,textbox,table',maxChars:600000})).ndjson);
await fs.writeFile('tmp/english_deck/source-montage.webp',new Uint8Array(await (await p.export({format:'webp',montage:true,scale:.6})).arrayBuffer()));
