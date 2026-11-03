import json
from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace

import deepseek_final_pair as D


def test_async_validation_uses_frozen_bytes(tmp_path):
    gate=threading.Event()
    with ThreadPoolExecutor(max_workers=1) as executor:
        host=D.Host(tmp_path/'host',6,threading.BoundedSemaphore(1),executor)
        host.start_at=time.monotonic();host.deadline=host.start_at+30
        root=tmp_path/'worker';candidate=root/'candidate';candidate.mkdir(parents=True)
        (root/'versions').mkdir();(candidate/'estimator.py').write_text('original')
        def execute(code,**kwargs):
            assert gate.wait(2)
            assert (root/'versions/0/estimator.py').read_text()=='original'
            return {'returncode':0,'stdout':'ENGINEERING_RESULT='+json.dumps({'selection_score':.5}),'stderr':''}
        session=SimpleNamespace(root=root,candidate=candidate,worker_id='test',tests=0,version_count=0,
            freeze=lambda destination:D.B.freeze(candidate.resolve(),destination.resolve(),6),execute=execute)
        reply=host.test(session,'frozen mutation check')
        assert reply['queued']
        (candidate/'estimator.py').write_text('changed')
        gate.set();executor.shutdown(wait=True)
        assert host.entries[0]['status']=='measured'
        host.select()
        assert (host.root/'selected/estimator.py').read_text()=='original'


def test_export_filter_keeps_runtime_dependencies(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'estimator.py').write_text('code');(source/'templates.npz').write_bytes(b'data')
    (source/'MUJOCO_LOG.TXT').write_text('log');(source/'__pycache__').mkdir()
    (source/'__pycache__/estimator.pyc').write_bytes(b'cache')
    D.copy_exports(source,tmp_path/'dest')
    assert {p.name for p in (tmp_path/'dest').iterdir()}=={'estimator.py','templates.npz'}


def test_original_public_validation_controls_task06_selection(tmp_path):
    with ThreadPoolExecutor(max_workers=1) as executor:
        host=D.Host(tmp_path/'host',6,threading.BoundedSemaphore(1),executor)
        for i in (0,1):
            source=tmp_path/f'source{i}';source.mkdir();(source/'estimator.py').write_text(str(i))
        first={'bundle':tmp_path/'source0','index':0,'worker':'noisy_proxy_best','score':.9}
        second={'bundle':tmp_path/'source1','index':1,'worker':'public_accurate_best','score':.8,'public_metric_score':.7}
        host.best=first;host.public_checker=SimpleNamespace(best=second)
        selected=host.select()
        assert selected['worker']=='public_accurate_best'
        assert selected['selection_basis']=='original_public_aggregate'
        assert (host.root/'selected/estimator.py').read_text()=='1'


def test_stop_falls_back_when_process_group_signal_is_denied(monkeypatch):
    from task08_swarm_client import SharedClient
    from types import SimpleNamespace
    import task08_swarm_client as client
    state={'returncode':None,'terminated':False}
    def denied(*args):
        raise PermissionError('process group denied')
    def terminate():
        state['terminated']=True;state['returncode']=-15
    process=SimpleNamespace(pid=123,poll=lambda:state['returncode'],terminate=terminate,
        wait=lambda timeout:state['returncode'])
    monkeypatch.setattr(client.os,'killpg',denied)
    SharedClient._stop(process)
    assert state['terminated']
