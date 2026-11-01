import fs from 'node:fs/promises';
import {Presentation, PresentationFile} from 'file:///C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
import {finalizePresentation} from 'file:///C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning';
const build=root+'/.codex-build/grasp_background';
const skill='C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
await fs.mkdir(build,{recursive:true});
await fs.mkdir(root+'/Report/background_slide',{recursive:true});
const p=Presentation.create({slideSize:{width:960,height:720}});
p.theme.defaultFont='Times New Roman'; const s=p.slides.add(); s.background.fill='#FFFFFF';
function txt(text,x,y,w,h,size=21,bold=false,color='#142F48',align='left'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=text; q.text.insets=0; q.text.style={typeface:'Times New Roman',fontSize:size,bold,color,alignment:align};return q;
}
txt('Typical Aperture Changes in Human Grasping',44,39,880,48,35,true,'#000000');
const img=await fs.readFile('C:/Users/Di/AppData/Local/Temp/codex-clipboard-19d6c225-301e-44d7-8815-e164cd190f9b.png');
s.images.add({blob:img.buffer.slice(img.byteOffset,img.byteOffset+img.byteLength),contentType:'image/png',alt:'Four recorded phases of a human grasp: closed, opening, wide open, closing to grasp',position:{left:44,top:108,width:872,height:127},fit:'contain'});
['1. Initially Closed','2. Opening','3. Wide Open','4. Closing to Grasp'].forEach((t,i)=>txt(t,44+i*220,247,212,26,18,true,'#142F48','center'));
txt('Actual grasping sequence (overhead camera view)',44,282,872,25,17,false,'#6A7885');
txt('Finger aperture during grasping',48,332,408,29,21,false,'#142F48','center');
const x=Array.from({length:21},(_,i)=>i/20);
const small=[.06,.07,.10,.16,.25,.37,.5,.62,.72,.79,.83,.81,.72,.56,.38,.23,.17,.16,.16,.16,.16];
const large=small.map((v,i)=>i<3?v:Number((v*1.13+.035).toFixed(4)));
s.charts.add('scatter',{position:{left:49,top:370,width:405,height:227},series:[{name:'Smaller object',xValues:x,values:small,line:{fill:'#7D8E9D',width:2.5},marker:{symbol:'none'}},{name:'Larger object',xValues:x,values:large,line:{fill:'#1B8191',width:2.5},marker:{symbol:'none'}}],scatterOptions:{style:'smooth'},hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:'Times New Roman',fontSize:14}},xAxis:{min:0,max:1,majorUnit:.25,title:{text:'Movement progress',textStyle:{typeface:'Times New Roman',fontSize:15}},textStyle:{typeface:'Times New Roman',fontSize:12},majorGridlines:null},yAxis:{min:0,max:1.15,tickLabelPosition:'none',title:{text:'Finger aperture',textStyle:{typeface:'Times New Roman',fontSize:15}},majorGridlines:{fill:'#E2E7EB',width:.6}},chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF'});
txt('Schematic only; curves do not represent measured data',49,608,409,32,13,false,'#6A7885','center');
const pts=[['1. Opening and closing','The fingers open during reaching and close\nbefore contacting the object. [5]'],['2. Adjustment to object size','Larger objects generally lead to a larger\nmaximum grip aperture. [6]'],['3. Adjustment to movement conditions','Grip aperture adapts to movement speed\nand accuracy requirements. [6, 7]']];
pts.forEach(([a,b],i)=>{const y=337+i*95;txt(a,491,y,425,28,22,true);txt(b,491,y+34,425,52,19,false,'#1E2933');});
txt('[5] Jeannerod M. J Mot Behav 16:235–254, 1984.    [6] Smeets JBJ, Brenner E. Motor Control 3:237–271, 1999.',44,658,872,20,12,false,'#566778');
txt('[7] Wing AM, Turton A, Fraser C. J Mot Behav 18:245–260, 1986.',44,680,872,19,12,false,'#566778');
s.speakerNotes.text='Background slide, not a report of the present experiments. The top photographs are user-supplied illustrations of an actual grasp. The editable chart is a qualitative schematic, with arbitrary aperture units and movement progress; its values and peak timing are not empirical estimates.\n[5] Jeannerod M (1984). The Timing of Natural Prehension Movements. Journal of Motor Behavior 16(3):235–254. https://doi.org/10.1080/00222895.1984.10735319 https://pubmed.ncbi.nlm.nih.gov/15151851/\n[6] Smeets JBJ, Brenner E (1999). A New View on Grasping. Motor Control 3(3):237–271. https://doi.org/10.1123/mcj.3.3.237 This paper integrates previous experimental findings with a theoretical alternative.\n[7] Wing AM, Turton A, Fraser C (1986). Grasp Size and Accuracy of Approach in Reaching. Journal of Motor Behavior 18(3):245–260. https://doi.org/10.1080/00222895.1986.10735380\nNo universal numerical claims about maximum-aperture timing or speed-related percentage increases are made. This slide does not claim that all grasping is unconscious.';
console.log('rendering'); await (await PresentationFile.exportPptx(p)).save(build+'/candidate.pptx'); console.log('exported'); 
await (await PresentationFile.exportPptx(p)).save(build+'/candidate.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:build+'/candidate.pptx',finalPath:root+'/Report/background_slide/Human_Grasping_Background.pptx',pythonExecutable:'C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit'],explicitTotalSlideCount:1,requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Times New Roman']},verifyArtifactToolImport:true,receiptPath:build+'/validation.json'});
console.log(JSON.stringify(result));






