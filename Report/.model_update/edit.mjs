import fs from 'node:fs/promises';
import {PresentationFile,FileBlob} from 'file:///C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning';
const p=await PresentationFile.importPptx(await FileBlob.load(root+'/Report/T_EN.pptx'));
const slide=p.slides.items[0];
const records=(await p.inspect({kind:'textbox,shape',maxChars:40000})).records;
const updates={
'boxText_162':'x_t, s',
'text_163':'2 inputs',
'boxText_166':'Cubic B-spline\nd̂ = f(x, s)',
'text_167':'10 basis functions + size interaction',
'text_173':'x_t−4','text_175':'x_t−3','text_177':'x_t−2','text_179':'x_t−1','text_181':'x_t',
'text_182':'6 inputs',
'text_189':'32 (ReLU)',
'text_196':'32 (ReLU)',
'text_203':'128 MC runs, median',
'text_207':'x_0, s',
'text_211':'x_1, s',
'text_216':'x_t, s',
'text_226':'2 inputs at each step',
'text_227':'Joint training on 3 sizes. Trial-level 5-fold CV. MLP/LSTM use MAE, Adam and 250 epochs.',
};
for(const r of records){
 const target=slide.shapes.items.find(s=>s.name===r.name); if(!target) continue;
 let value=updates[r.name];
 if(r.text?.startsWith('Pass the 32-dimensional')) value='1 layer, 32 units. Carry h and c forward';
 if(r.text==='Dense layer') value='Linear';
 if(value!==undefined){target.text=value; if(['text_167','text_182','text_203','text_227'].includes(r.name))target.text.style={fontSize:13};}
}
for (const s of slide.shapes.items) { if(['arrow_206','arrow_210','arrow_215'].includes(s.name)) s.rotation=180; }
function text(name,value,x,y,w,h,size){
 const s=slide.shapes.add({geometry:'textbox',name,position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:'Meiryo UI',fontSize:size,color:'#0E2841',alignment:'left'};
}
text('normalization','x_t = l_t/l_0     s = cube size (2, 3 or 4 cm)     d = finger aperture (mm)',48,94,864,24,14);
text('mlp_dropout','Dropout 0.1 after first ReLU',585,394,300,24,13);
text('lstm_dropout','Dropout 0.1 before output during training',420,604,470,22,12);
text('training_details','Adam lr 0.001, weight decay 0.0001. Function uses penalized least squares (λ = 0.0001).',48,661,864,22,12);
const cellIds=[172,174,176,178,180];
for (const [i,id] of cellIds.entries()) {
  for (const name of ['cell_'+id,'text_'+(id+1)]) {
    const shape=slide.shapes.items.find(s=>s.name===name);
    shape.position={...shape.position,top:300+i*19,height:19};
  }
}
slide.shapes.add({geometry:'rect',name:'size_input_cell',position:{left:220.8,top:395,width:81.6,height:19},fill:'#FFFFFF',line:{fill:'#0E2841',width:1}});
const sizeInput=slide.shapes.add({geometry:'textbox',name:'size_input_label',position:{left:220.8,top:395,width:81.6,height:19},fill:'none',line:{fill:'none',width:0}});
sizeInput.text='s';
sizeInput.text.style={typeface:'Meiryo UI',fontSize:13.33,alignment:'center',verticalAlignment:'middle',insets:{top:0,bottom:0,left:0,right:0}};
await fs.writeFile(root+'/Report/.model_update/after.png',new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await(await PresentationFile.exportPptx(p)).save(root+'/Report/.model_update/candidate.pptx');
console.log(records.filter(r=>r.text && (r.text.includes('l_')||r.text==='Dense layer')).map(r=>[r.name,r.text]));



