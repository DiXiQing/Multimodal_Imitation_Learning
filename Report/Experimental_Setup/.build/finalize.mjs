import fs from 'node:fs/promises';
import {finalizePresentation} from 'file:///C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning/Report/Experimental_Setup';
const skill='C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations';
try {
 await finalizePresentation({workspaceDir:root,candidatePath:root+'/.build/candidate.pptx',
 finalPath:root+'/Output/Experimental_Setup.pptx',explicitTotalSlideCount:1,
 pythonExecutable:'C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
 integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
 verifyArtifactToolImport:true,receiptPath:root+'/.build/validation.json'});
} catch (error) {
 if(error.code!=='EPERM'||error.syscall!=='link') throw error;
 await fs.copyFile(error.path,error.dest,fs.constants.COPYFILE_EXCL);
 console.log('Copied validated output after Windows hard-link restriction');
}
await fs.copyFile(root+'/.build/preview.png',root+'/Output/Experimental_Setup.png');
