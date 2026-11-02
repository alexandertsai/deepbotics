"""Integration checks for the macOS port, without model calls."""
import json
from pathlib import Path
import socket
import sys

import pytest

import mac_evaluation as M
from replicate_deepseek import make_configuration
from services.controller.backend import make_backend
from services.controller.frontend import ResearchFrontend
from services.controller.sandbox import BundleSandbox
from services.controller.supervisor import Supervisor

pytestmark = pytest.mark.skipif(sys.platform != 'darwin', reason='macOS port')


def test_bundle_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv('RAX_ARTIFACT_ROOT', str(tmp_path))
    config = make_configuration(tmp_path / 'config', 2, 1, 1)
    secret = tmp_path / 'private-simulator-state'
    secret.write_text('must remain private')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'controller.py').write_text('''import socket
import numpy, scipy, PIL
from pathlib import Path
def main(ctx):
    try:
        Path(''' + repr(str(secret)) + ''').read_text()
        raise AssertionError('private file readable')
    except PermissionError:
        pass
    try:
        socket.create_connection(('127.0.0.1', 9), timeout=1)
        raise AssertionError('network available')
    except PermissionError:
        pass
    Path('output/result.txt').write_text('isolated')
    print('isolated')
''')
    class Gateway:
        terminal = False
        simulator_exited = False
        def handle(self, token, request):
            return {'jsonrpc':'2.0','id':request['id'],'result':{}}
    runner = BundleSandbox(config.formal)
    runner.preflight()
    result = runner.run(bundle, mode='rehearsal', entrypoint='controller.py', gateway=Gateway(),
        output=tmp_path / 'execution', workspace=tmp_path / 'workspace')
    assert result['returncode'] == 0, (tmp_path / 'execution/stderr.log').read_text()
    assert (tmp_path / 'execution/files/result.txt').read_text() == 'isolated'


def test_native_workflow(tmp_path, monkeypatch):
    monkeypatch.setenv('RAX_ARTIFACT_ROOT', str(tmp_path))
    config = make_configuration(tmp_path / 'workflow', 3, 2, 2)
    supervisor = Supervisor(config, lambda: make_backend(config))
    workspace = config.root / 'agent-workspace'
    frontend = ResearchFrontend(supervisor, workspace)
    counter = 0
    def call(name, **arguments):
        nonlocal counter
        counter += 1
        response = frontend.handle({'jsonrpc':'2.0','id':counter,'method':'tools/call',
                                   'params':{'name':name,'arguments':arguments}})
        assert 'error' not in response, response
        result = response['result']
        assert not result.get('isError'), result
        return json.loads(result['content'][0]['text'])
    try:
        call('start_episode')
        obs = call('dexjoco_observe')
        assert len(obs['state']) == 23
        assert len(obs['depth']) == 2
        assert 'camera_intrinsics' in obs and 'camera_poses' in obs
        assert not any(k in obs for k in ['object_poses','table_height','seed','layout_id'])
        call('dexjoco_hold', steps=2)
        call('finish')
        project = workspace / 'code/hold'
        project.mkdir(parents=True)
        (project / 'controller.py').write_text("def main(ctx):\n    ctx.call('dexjoco_hold', steps=2)\n")
        bundle = call('register', source='code/hold')['bundle']
        rehearsal = call('rehearse', bundle=bundle)
        assert rehearsal['returncode'] == 0, rehearsal
        assert rehearsal['task_complete'] is False
        call('submit', bundle=bundle)
        formal = supervisor.run_submission()
        assert formal['status'] == 'completed', formal
        assert formal['completed_episodes'] == 2 and formal['error_count'] == 0
    finally:
        frontend.close()
