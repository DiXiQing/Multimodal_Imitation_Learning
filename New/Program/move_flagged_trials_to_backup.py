"""Resume the user-requested separation of 12 reviewed trials, preserving audit records."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

root = Path(__file__).resolve().parents[1]/'Data'/'2'
root = root.resolve()
backup = (root/'backup').resolve()
review = root/'quality_review'
assert backup.parent == root
with (review/'review_decisions.csv').open(encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    fields = list(reader.fieldnames)
    rows = list(reader)
selected = [r for r in rows if r['numerical_review_flag'] == '1']
assert len(selected) == 12
plan = []
for r in selected:
    src = (root/r['trial']).resolve()
    dest = (backup/r['trial']).resolve()
    assert src.parent == root and dest.parent == backup
    assert src.is_dir() != dest.is_dir(), f'Ambiguous source/destination: {src}'
    location = src if src.is_dir() else dest
    assert hashlib.sha256((location/'camera.csv').read_bytes()).hexdigest() == r['csv_sha256']
    plan.append((src, dest, r))
for name in ['review_decisions.csv', 'review_summary.json', 'retained_removed_overlay.png', 'review_report.md']:
    src = review/name
    target = review/(src.stem+'_before_move'+src.suffix)
    if src.exists() and not target.exists():
        shutil.copy2(src, target)
manifest = []
for src, dest, r in plan:
    location = src if src.is_dir() else dest
    hashes = {str(p.relative_to(location)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in location.rglob('*') if p.is_file()}
    if src.is_dir():
        src.rename(dest)
    after = {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in dest.rglob('*') if p.is_file()}
    assert hashes == after
    manifest.append(dict(original=str(src), backup=str(dest), trigger=r['review_trigger'],
                         file_count=len(hashes), file_hashes=hashes))
    r.update(decision='user_requested_backup', moved_to_backup='1',
             conclusion='Temporarily set aside at user request based on review flags; not confirmed invalid.')
with (review/'review_decisions.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
(backup/'move_manifest.json').write_text(json.dumps(dict(
    moved_at=datetime.now(timezone(timedelta(hours=9))).isoformat(),
    status='User-requested temporary separation; not confirmed bad data', trials=manifest), indent=2), encoding='utf-8')
s = json.loads((review/'review_summary.json').read_text())
s.update(retained=41, moved=12, flagged_but_retained=0,
         move_basis='Explicit user request to temporarily separate all 12 numerically flagged trials, not confirmed defects.')
(review/'review_summary.json').write_text(json.dumps(s, indent=2), encoding='utf-8')
(backup/'README.txt').write_text(
    '12 numerically flagged trials temporarily moved here at user request. Not confirmed invalid. '
    'Original names and file contents preserved. See move_manifest.json. Parent retains 41 trials. '
    'Previous training results unchanged.\n', encoding='utf-8')
with (review/'review_report.md').open('a', encoding='utf-8') as f:
    f.write('\n\n## User-requested separation\n\n12 flagged trials moved to ../backup at explicit user request; '
            '41 retained. This is not confirmation of measurement defects. Moved files were hash-verified. '
            'Original review records preserved as *_before_move. Previous training results unchanged.\n')
assert sum(p.is_dir() and (p/'camera.csv').is_file() for p in root.iterdir()) == 41
assert sum(p.is_dir() and (p/'camera.csv').is_file() for p in backup.iterdir()) == 12
print('Moved 12, retained 41. Moved file contents verified unchanged.')
