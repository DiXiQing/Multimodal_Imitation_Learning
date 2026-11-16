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
text('Title','Real-Time Grasping Experiment',48,33,864,62,42,navy,true);
const frame = await fs.readFile('Data/1101_D/30mm/20261103_143601/frames/frame_001306.jpg');
slide.images.add({blob:new Uint8Array(frame),contentType:'image/jpeg',alt:'Overhead view of a hand reaching toward a 30 mm cube',fit:'contain',position:{left:55,top:121,width:476,height:335}});
text('Image caption','Example grasp from the overhead camera',48,461,493,32,17,gray);
text('Trial count','27 trials',565,117,347,67,48,teal,true);
text('Repetitions','One trial per combination',565,181,347,29,20,gray);
for (const [i,[label,value]] of [
 ['Cube size','20 / 30 / 40 mm'],
 ['Initial hand–object distance','≈ 250 / 300 / 350 mm'],
 ['Grasp duration','≈ 1 / 2.5 / 4 s'],
].entries()){
 const y=222+i*80;
 text(label,label,565,y,347,28,18,gray,true);
 text(label+' values',value,565,y+28,347,37,27,navy,true);
}
text('Models label','Live model comparison',48,520,864,30,23,teal,true);
text('Models','Function Approximation     /     MLP (5 frames)     /     LSTM (full history)',48,552,864,31,21,navy,true);
text('Evaluation','Same valid frames. Per-trial MAE and RMSE (mm) and R², reported as mean ± SD.',48,604,864,36,19);
text('Footer','Saga University, Fukuda Laboratory',48,674,864,26,15,gray);
slide.speakerNotes.text='All three models receive the same live camera stream in each of the 27 planned trials. Calculate error against measured finger aperture on the frames valid for all three models. Distance and grasp duration values are targets. Image source: study recording Data/1101_D/30mm/20261103_143601/frames/frame_001306.jpg. The image illustrates the overhead view.';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning/Report/Realtime_Experiment';
await fs.writeFile(root+'/.build/preview_visual.png',new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(root+'/.build/candidate_visual.pptx');
