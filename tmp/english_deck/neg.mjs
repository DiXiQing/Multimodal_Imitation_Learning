import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('C:/Users/Di/Downloads/Di_ゼミ発表_修正版.pptx'));
for(const s of p.slides.items)for(const x of s.shapes.items){if(x.position.width<0||x.position.height<0)console.log(x.id,JSON.stringify(x.position));}
