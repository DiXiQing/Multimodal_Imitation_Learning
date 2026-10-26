"""Copy an unfiltered stratified random trial split; never modifies sources."""
from pathlib import Path
import csv
import hashlib
import json
import random
import shutil

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'Data/D4'
SEED=20261026
SOURCES={
    'fast': 'TRIAL_Color_black_Size_4cm_20261025_143653_569025',
    'normal': 'TRIAL_Color_black_Size_4cm_20261025_145833_827478',
    'slow': 'TRIAL_Color_black_Size_4cm_20261025_150714_665350',
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    if DEST.exists() and any(DEST.iterdir()):
        raise ValueError(f'Destination is not empty: {DEST}')
    plan=[];cohorts={}
    for index,(condition,recording) in enumerate(SOURCES.items()):
        source=ROOT/'Data/TopView'/recording/'grasp_segments'
        folders=sorted(p for p in source.iterdir() if p.is_dir() and (p/'camera.csv').is_file() and (p/'segment.json').is_file())
        if len(folders)<28:raise ValueError(f'{condition}: fewer than 28 complete trial folders')
        random.Random(SEED+index).shuffle(folders)
        cohorts[condition]=dict(source=str(source),available=len(folders),seed=SEED+index,train=21,test=7,unused=len(folders)-28)
        for n,folder in enumerate(folders[:28]):
            split='Train' if n<21 else 'Test'
            target=DEST/split/folder.name
            assert target.resolve().is_relative_to(DEST.resolve())
            if target.exists():raise ValueError(f'Destination exists: {target}')
            plan.append((condition,split,folder,target))
    names=[p[2].name for p in plan]
    assert len(names)==len(set(names))==84
    csv_hashes=[digest(p[2]/'camera.csv') for p in plan]
    if len(csv_hashes)!=len(set(csv_hashes)):raise ValueError('Duplicate source CSV content')
    DEST.mkdir(exist_ok=True)
    records=[]
    for condition,split,source,target in plan:
        shutil.copytree(source,target,copy_function=shutil.copy2)
        original={str(p.relative_to(source)):p for p in source.rglob('*') if p.is_file()}
        copied={str(p.relative_to(target)):p for p in target.rglob('*') if p.is_file()}
        if set(original)!=set(copied):raise ValueError(f'File list mismatch: {target}')
        total=0
        for key,path in original.items():
            if digest(path)!=digest(copied[key]):raise ValueError(f'Content mismatch: {target/key}')
            total+=path.stat().st_size
        records.append(dict(condition=condition,split=split,trial=source.name,source=str(source),destination=str(target),
                            camera_csv_sha256=digest(source/'camera.csv'),files=len(original),bytes=total))
        if len(records)%7==0:print(f'Copied and hash-verified {len(records)}/84 trials',flush=True)
    with (DEST/'random_split_manifest.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    method=dict(seed=SEED,conditions=cohorts,selection='Sorted original cropped cohort, independent seeded shuffle per condition; first 21 Train, next 7 Test. No R2 or curve-consistency filtering.',
                source_unchanged=True,cropping='Existing segments copied without recropping, normalization or signal edits.',
                verification='All copied files SHA256 match sources; trial names and camera CSV hashes unique across Train and Test.',
                note='Test is held out from D4 fitting. Do not use it for checkpoint selection. These recordings have been used in earlier D3 experiments; this is not a newly recorded independent session.')
    (DEST/'split_method.json').write_text(json.dumps(method,indent=2),encoding='utf-8')
    for split,count in [('Train',63),('Test',21)]:
        assert len(list((DEST/split).glob('*/camera.csv')))==count
    print('DONE: D4 Train=63, Test=21; all copied files verified.',flush=True)

if __name__=='__main__':main()
