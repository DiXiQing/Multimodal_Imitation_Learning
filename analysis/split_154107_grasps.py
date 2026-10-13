from pathlib import Path

root = Path(__file__).resolve().parents[1]
template = (root/'analysis/split_150004_grasps.py').read_text(encoding='utf-8')
template = template.replace("'Data/BackupData/D/TRIAL_Size_3.5cm_20261013_150004_279664'", "'Data/临时数据/D13/3/TRIAL_Size_3cm_20261013_154107_842077'")
template = template.replace('ranges = [(235,305), (388,459), (569,635), (782,847), (1028,1093), (1205,1274)]',
                            "candidates = json.loads((source/'candidate_ranges.json').read_text())\nranges = [(r['start'], r['end']) for r in candidates]")
template = template.replace('150004_segment','154107_segment')
template = template.replace("'6 grasps / 3.5 cm'", "'21 grasps / 3 cm'")
template = template.replace('six_grasps_smooth_overlay.png','21_grasps_smooth_overlay.png')
template = template.replace('all 6 videos','all 21 videos')
template = template.replace("color=plt.cm.tab10(number-1)","color=plt.cm.tab20((number-1)%20),lw=1.1")
template = template.replace("frameon=False,title=", "frameon=False,ncol=3,title=")
template = template.replace("source_sha256=hashes)","source_sha256=hashes,flags=['Short stable hold: crop ends before release'] if start==401 else [])")
exec(compile(template,str(root/'analysis/split_150004_grasps.py'),'exec'))
