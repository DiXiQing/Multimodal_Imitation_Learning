import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

tree = ast.parse((Path(__file__).resolve().parents[1] / 'Record_mediapipe.py').read_text(encoding='utf-8-sig'))
fn = next(n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'imu_record_task')
state = SimpleNamespace(running=True, calibrating=True, lock=threading.Lock())
calls = []

async def fail():
    calls.append(1)
    raise ConnectionError('mock connection failure')

async def sleep(seconds):
    pass

ns = dict(state=state, IMU_MAX_CONNECT_ATTEMPTS=3, IMU_RETRY_DELAY_SECONDS=2.0,
          imu_record_once=fail, asyncio=SimpleNamespace(sleep=sleep))
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'retry-test', 'exec'), ns)
coroutine = ns['imu_record_task']()
try:
    coroutine.send(None)
except StopIteration:
    pass
assert len(calls) == 3 and state.running is False and state.calibrating is True
print('PASS: three failed attempts stop recorder before calibration completes')
