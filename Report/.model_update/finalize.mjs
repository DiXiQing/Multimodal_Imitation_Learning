import {finalizePresentation} from 'file:///C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations/container_tools/artifact_tool_utils.mjs';
const root='C:/MineApp/Code/Multimodal_Imitation_Learning/Report';
const skill='C:/Users/Di/.codex/plugins/cache/openai-primary-runtime/presentations/26.1030.10813/skills/presentations';
console.log(await finalizePresentation({
workspaceDir:root,candidatePath:root+'/.model_update/candidate.pptx',finalPath:root+'/Updated/T_EN_latest_models.pptx',
pythonExecutable:'C:/Users/Di/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',
integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',
layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',
explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:root+'/.model_update/validation.json'
}));

