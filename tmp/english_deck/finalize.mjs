import fs from 'node:fs/promises';
import {finalizePresentation} from 'file:///C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning';
const skill='C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations';
const r=await finalizePresentation({workspaceDir:root,candidatePath:root+'/tmp/english_deck/complete.pptx',finalPath:root+'/Zemi/output/Di_Zemi_English_Final.pptx',pythonExecutable:'C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-bullet-geometry','--validate-heading-fit'],explicitTotalSlideCount:18,requiredNativeTableOwnerSlides:[],requiredNativeChartOwnerSlides:[],verifyArtifactToolImport:true,receiptPath:root+'/tmp/english_deck/validation-final.json'});
console.log(JSON.stringify(r));




