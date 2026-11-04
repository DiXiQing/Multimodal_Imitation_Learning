import fs from 'node:fs/promises';
import {Presentation,PresentationFile,FileBlob} from 'file:///C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/MineApp/Code/Multimodal_Imitation_Learning/Report/T_EN.pptx'));
console.log(await p.inspect({kind:'slide,textbox,shape',maxChars:24000}));
await fs.writeFile('C:/MineApp/Code/Multimodal_Imitation_Learning/Report/.model_update/before.png',new Uint8Array(await (await p.slides.items[0].export({format:'png',scale:1})).arrayBuffer()));


