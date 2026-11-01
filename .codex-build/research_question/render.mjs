import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from 'file:///C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const path='C:/MineApp/Code/Multimodal_Imitation_Learning/Zemi/output/Research_Question_Slide.pptx';
const p=await PresentationFile.importPptx(await FileBlob.load(path));
const png=await p.export({slide:p.slides.items[0],format:'png',scale:1});
await fs.writeFile('C:/MineApp/Code/Multimodal_Imitation_Learning/.codex-build/research_question/preview.png',new Uint8Array(await png.arrayBuffer()));
console.log('Preview saved');
