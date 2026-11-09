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
text('Title','Summary and Future Work',48,34,864,68,42,navy,true);
text('Summary heading','Summary',48,114,864,38,27,teal,true);
text('Result 1','Adding velocity reduced mean absolute error for all three models.',48,159,864,40,24);
text('Result 2','LSTM Full had the lowest MAE at 4.32 mm in five-fold cross-validation.',48,203,864,40,24);
text('Scope heading','Current Limitations',48,261,864,32,20,teal,true);
text('Scope details','One participant and planar grasps of 20, 30 and 40 mm cubes (600 trials).',48,296,864,36,20,gray);
text('Roadmap heading','Future Work',48,363,864,48,32,navy,true);
shape('Roadmap line','line',149,486,663,0,'none',{fill:'#AABDC7',width:2});
const phases=[
 ['Current','Joint aperture prediction for three cube sizes'],
 ['Next','Verify predictions against measured grasp trajectories'],
 ['Then','Recruit more participants and test generalization'],
 ['Future','Integrate prediction with a powered prosthesis and EMG control'],
];
for(const [i,[label,body]] of phases.entries()){
 const x=48+i*221;
 const heading=text(label,label,x,429,202,38,22,navy,true);
 heading.text.style={alignment:'center',verticalAlignment:'middle'};
 shape(label+' marker','ellipse',x+92,477,18,18,i===0?teal:'#FFFFFF',{fill:teal,width:2});
 const details=text(label+' details',body,x,513,202,105,20);
 details.text.style={alignment:'center',verticalAlignment:'top'};
}
text('Footer','Saga University, Fukuda Laboratory',48,674,864,26,15,gray);
slide.speakerNotes.text='Results are from the joint five-fold experiment using SG smoothing and recent five-frame velocity estimation. The lowest trial-level MAE is 4.3219 mm. Velocity comparison uses the same 43,051 valid test frames across 600 trials. Follow the stated sequence of system verification, evaluation on additional participants, and future prosthetic-hand integration.';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning/Report/Summary_Next_Steps';
await fs.writeFile(root+'/.build/preview_revised.png',new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(root+'/.build/candidate_revised.pptx');

