import fs from 'node:fs/promises';
import {Presentation, PresentationFile} from 'file:///C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning';
const build=root+'/.codex-build/research_question';
const skill='C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
await fs.mkdir(build,{recursive:true});
await fs.mkdir(root+'/Zemi/output',{recursive:true});
const p=Presentation.create({slideSize:{width:960,height:720}});
p.theme.defaultFont='Times New Roman';
const s=p.slides.add();s.background.fill='#FFFFFF';
function box(name,x,y,w,h,fill){return s.shapes.add({name,geometry:'roundRect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0},borderRadius:12});}
function txt(name,text,x,y,w,h,size,bold=false,color='#000000',align='left'){
const q=s.shapes.add({name,geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
q.text=text;q.text.insets=0;q.text.style={typeface:'Times New Roman',fontSize:size,bold,color,alignment:align};return q;}
txt('Slide title','Research Question',42,34,880,60,44,true);
box('Main research question',42,118,876,156,'#196581');
txt('Question','Can hand–object distance measured by an overhead camera\npredict finger aperture during grasping?',66,159,828,84,29,true,'#FFFFFF','center');
const items=[
['1','Distance–Aperture\nRelationship','How does finger aperture\nvary with hand–object\ndistance?','Distance and aperture'],
['2','Role of Temporal\nInformation','Does distance history\nimprove prediction over\nthe current distance alone?','Function / MLP 5f / LSTM Full'],
['3','Effect of\nObject Size','How does prediction\nperformance vary across\nthree object sizes?','20 mm / 30 mm / 40 mm']
];
items.forEach(([n,title,body,tag],i)=>{
const x=42+i*300;
box('Research focus '+n,x,319,276,292,'#EDF2F5');
txt('Focus number '+n,n,x+20,337,42,46,35,true,'#196581');
txt('Focus title '+n,title,x+20,394,236,66,25,true,'#123249');
txt('Focus question '+n,body,x+20,479,236,84,22,false,'#152B3A');
txt('Focus details '+n,tag,x+20,575,236,22,15,false,'#196581');
});
txt('Laboratory footer','Saga University Fukuda Laboratory',42,679,650,22,14,false,'#008060');
s.speakerNotes.text='Research question and evaluation aims for the present study. These are questions rather than established results. Predict means estimating human finger aperture from the measured hand–object distance. Function approximation uses the current distance. MLP uses five frames of distance history. LSTM uses full available distance history. Compare performance separately across 20, 30 and 40 mm objects. Prosthetic actuation remains future work.';
await (await PresentationFile.exportPptx(p)).save(build+'/candidate.pptx');
const out=await finalizePresentation({workspaceDir:root,candidatePath:build+'/candidate.pptx',finalPath:root+'/Zemi/output/Research_Question_Slide.pptx',pythonExecutable:'C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit'],explicitTotalSlideCount:1,fontPolicy:{basis:'design',families:['Times New Roman']},verifyArtifactToolImport:true,receiptPath:build+'/validation.json'});
console.log(JSON.stringify(out));
if(process.argv.includes('--render')){
 const png=await p.export({slide:s,format:'png',scale:1.5});
 await fs.writeFile(build+'/preview.png',new Uint8Array(await png.arrayBuffer()));
}
