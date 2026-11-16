import fs from 'node:fs/promises';
import {finalizePresentation} from 'file:///C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning/Report/Realtime_Experiment';
const skill='C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations';
try {
 await finalizePresentation({workspaceDir:root,candidatePath:root+'/.build/candidate_visual.pptx',
 finalPath:root+'/Output/Real_Time_Grasping_Experiment_Visual.pptx',explicitTotalSlideCount:1,
 pythonExecutable:'C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 verifyArtifactToolImport:true,receiptPath:root+'/.build/validation_visual.json'});
} catch (error) {
 if(error.code!=='EPERM'||error.syscall!=='link') throw error;
 await fs.copyFile(root+'/.build/candidate_visual.pptx',error.dest,fs.constants.COPYFILE_EXCL);
 console.log('Copied validated output after Windows hard-link restriction');
}
await fs.copyFile(root+'/.build/preview_visual.png',root+'/Output/Real_Time_Grasping_Experiment_Visual.png');



