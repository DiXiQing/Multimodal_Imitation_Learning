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
function shape(name,geometry,x,y,w,h,fill,line){
 return slide.shapes.add({name,geometry,position:{left:x,top:y,width:w,height:h},fill,line});
}
text('Title','Summary and Next Steps',48,38,864,70,42,navy,true);
text('Summary heading','Summary',48,124,864,39,27,teal,true);
text('Result 1','Adding velocity reduced mean absolute error for all three models.',48,169,864,44,24);
text('Result 2','LSTM Full had the lowest MAE at 4.32 mm in five-fold cross-validation.',48,220,864,45,24);
text('Roadmap heading','Research Roadmap',48,303,864,39,27,teal,true);
shape('Roadmap line','line',71,404,690,0,'none',{fill:'#B0BDC7',width:2});
const phases=[
 ['Next','Verify predictions against\nmeasured grasp trajectories.'],
 ['Then','Recruit more participants\nand test on held-out participants.'],
 ['Future','Integrate prediction with\na powered prosthesis and\nEMG control.'],
];
for(const [i,[label,body]] of phases.entries()){
 const x=48+i*295;
 text(label,label,x,351,275,42,23,navy,true);
 shape(label+' marker','ellipse',x+15,395,18,18,i===0?teal:'#FFFFFF',{fill:teal,width:2});
 text(label+' details',body,x,426,281,83,21);
}
shape('Scope separator','line',48,552,864,0,'none',{fill:'#D7E2E9',width:1});
text('Scope heading','Current Limitations',48,566,864,31,19,teal,true);
text('Scope details','One participant and planar grasps of 20, 30 and 40 mm cubes (600 trials).',48,600,864,36,20,gray);
text('Footer','Saga University, Fukuda Laboratory',48,674,864,26,15,gray);
slide.speakerNotes.text='Results are from the joint five-fold experiment using SG smoothing and recent five-frame velocity estimation. The lowest trial-level MAE is 4.3219 mm. Velocity comparison uses the same 43,051 valid test frames across 600 trials. Follow the stated sequence of system verification, evaluation on additional participants, and future prosthetic-hand integration.';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning/Report/Summary_Next_Steps';
await fs.writeFile(root+'/.build/preview.png',new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(root+'/.build/candidate.pptx');

