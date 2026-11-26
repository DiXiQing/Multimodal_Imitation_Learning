from pathlib import Path
import subprocess,sys,json,concurrent.futures,time
ROOT=Path(__file__).resolve().parents[1];report=ROOT/'Report/Segmentation_20261126';report.mkdir(exist_ok=True,parents=True)
script=ROOT/'tool/segment_black_20261102_143906.py'
def work(source):
 log=report/(source.name+'.log')
 detection=subprocess.run([sys.executable,str(script),str(source),'--detect-only'],capture_output=True,text=True,encoding='utf-8',errors='replace')
 log.write_text(detection.stdout+detection.stderr,encoding='utf-8')
 if detection.returncode:return dict(source=source.name,status='detection_failed',log=str(log))
 if 'candidates 0 review 0' in detection.stdout:return dict(source=source.name,status='no_candidates',segments=0,review=0)
 with log.open('a',encoding='utf-8') as f:
  run=subprocess.run([sys.executable,str(script),str(source)],stdout=f,stderr=subprocess.STDOUT)
 validation=source/'grasp_segments/validation.json'
 if run.returncode==0 and validation.exists():
  entries=json.loads((source/'grasp_segments/summary.json').read_text());return dict(source=source.name,status='validated',segments=len(entries),review=sum(bool(e['review_flags']) for e in entries))
 return dict(source=source.name,status='export_failed',log=str(log))
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 futures=[pool.submit(work,p) for p in sorted((ROOT/'Data/TopView').glob('*20261126*')) if p.is_dir()]
 for f in concurrent.futures.as_completed(futures):
  result=f.result();results.append(result);(report/'batch_status.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(result,flush=True)
print('BATCH FINISHED',flush=True)
