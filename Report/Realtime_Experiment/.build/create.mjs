import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const p=Presentation.create({slideSize:{width:960,height:720}});
const slide=p.slides.add();
slide.background.fill='#FFFFFF';
const navy='#183654', teal='#1C8187', gray='#526779';
function text(name,value,x,y,w,h,size,color=navy,bold=false){
 const s=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;
 s.text.style={typeface:'Arial',fontSize:size,color,bold,verticalAlignment:'middle'};
 return s;
}
text('Title','Real-Time Grasping Experiment',48,38,864,61,42,navy,true);
text('Purpose','Compare three models on live overhead camera frames.',48,104,864,38,24);
text('Conditions','Grasping conditions',48,173,550,36,25,teal,true);
const conditions=[
 ['Cube size','20, 30, and 40 mm'],
 ['Initial hand–object distance','Approx. 250, 300, and 350 mm'],
 ['Grasp duration','Approx. 1, 2.5, and 4 s'],
];
for(const [i,[label,value]] of conditions.entries()){
 const y=221+42*i;
 text(label,label,48,y,323,32,20,navy,true);
 text(label+' values',value,370,y,329,32,20);
}
text('Trial count','27 trials',708,207,205,59,39,teal,true);
text('Trial design','One per condition',708,267,205,32,20,gray);
text('Models heading','Real-time model comparison',48,375,864,36,25,teal,true);
text('Models','Function Approximation     /     MLP (5 frames)     /     LSTM (full history)',48,421,864,38,22,navy,true);
text('Common input frames','Run all three models on the same camera stream for every trial.',48,465,864,35,21);
text('Evaluation heading','Evaluation',48,526,864,36,25,teal,true);
text('Evaluation criteria','Compare predictions with measured aperture on the same valid frames.',48,569,864,34,21);
text('Metrics','Per-trial MAE and RMSE (mm) and R². Report mean ± SD across trials.',48,608,864,35,21);
text('Footer','Saga University, Fukuda Laboratory',48,674,864,26,15,gray);
slide.speakerNotes.text='This is the planned real-time experiment. The design crosses three cube sizes, three initial hand–object distances and three grasp durations, with one trial per combination. Distance and duration are target values. Calculate trial-level metrics on the frames valid for all three models. Summarize the nine trials per size and all 27 trials using mean and standard deviation.';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning/Report/Realtime_Experiment';
await fs.writeFile(root+'/.build/preview.png',new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(root+'/.build/candidate.pptx');
