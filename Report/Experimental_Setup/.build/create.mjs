import fs from 'node:fs/promises';
import {Presentation,PresentationFile} from 'file:///C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning';
const work=root+'/Report/Experimental_Setup';
const p=Presentation.create({slideSize:{width:1280,height:720}});
const s=p.slides.add();s.background.fill='#FFFFFF';
function text(name,value,x,y,w,h,size,color='#102E47',bold=false){
 const shape=s.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 shape.text=value;
 shape.text.style={fontFamily:'Arial',fontSize:size,color,bold,verticalAlignment:'middle'};
 return shape;
}
function line(name,x,y,w,h,color='#19848C',width=2){
 return s.shapes.add({geometry:'line',name,position:{left:x,top:y,width:w,height:h},line:{fill:color,width}});
}
text('title','Experimental Setup',64,42,1100,70,44,'#102E47',true);
s.images.add({dataUrl:'data:image/jpeg;base64,'+(await fs.readFile(work+'/.build/image4.jpeg')).toString('base64'),alt:'Actual overhead camera and cube recording setup',position:{left:64,top:144,width:420,height:496.36},fit:'contain'});
// Labels identify the camera and cube in the actual setup photograph.
text('camera label','Overhead camera',64,112,210,30,19,'#102E47',true);
line('camera pointer',145,139,15,25,'#102E47',2);
text('cube label','Target cube',307,544,154,36,19,'#102E47',true);
line('cube pointer',281,542,25,20,'#102E47',2);
text('section','Recording conditions',550,144,620,46,28,'#19848C',true);
const rows=[
 ['Camera recording','30 fps'],
 ['Cube side lengths','20, 30 and 40 mm'],
 ['Initial hand-object distance','250–350 mm'],
 ['Trial duration','1–4 s']
];
for(let i=0;i<rows.length;i++){
 const y=210+i*87;
 text('label'+i,rows[i][0],550,y,610,28,20,'#667984');
 text('value'+i,rows[i][1],550,y+29,620,45,32,'#102E47',true);
}
text('scope','Planar grasping recorded from above',550,584,630,40,23,'#19848C');
text('footer','Saga University, Fukuda Laboratory',64,675,850,24,15,'#778894');
s.speakerNotes.text='Setup photograph extracted from slide 8 of Zemi/output/Di_Zemi_English_Final.pptx. Recording conditions use confirmed experiment settings from the user: 20/30/40 mm cubes, initial hand–object distance 250–350 mm and trial duration 1–4 s. Camera recording is nominally 30 fps. Camera model, calibrated camera height and resolution are omitted pending verification.';
await fs.writeFile(work+'/.build/preview.png',new Uint8Array(await(await s.export({format:'png',scale:1})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(work+'/.build/candidate.pptx');
console.log('Created one editable slide');


