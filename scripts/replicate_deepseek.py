"""Run RAX's native Claude Code workflow with a bounded Fireworks API relay.

Custom adapter: API metering and macOS launch configuration only. RAX owns tools,
episodes, workspaces, registration, rehearsals, submission, and formal evaluation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time

import httpx2
import mac_evaluation as M
from claude_mac_evaluation import MODEL, usage_from_response
from services.controller.config import Configuration, Limits


def make_configuration(root, episodes, formal_episodes, workers):
    limits = Limits(6600, 8192, 4, 256, 1024, 1024**3)
    return Configuration(root=root, task='fold_glasses', exploration_seeds=tuple(range(episodes)),
        formal_seed=0, eval_seed=0, sim_gpu='mac', development=limits, training=limits, formal=limits,
        demonstration_context='terminal_state', expert_demonstrations=0, formal_episodes=formal_episodes,
        formal_eval_seed=1, formal_workers=workers, observation_profile='rgbd', exploration_envs=1,
        subagents=0, task_rules='public', agent_cli='claude', agent_provider='custom',
        agent_effort='max', external_perception=False)


class Relay:
    def __init__(self, root, keys, args):
        self.root, self.keys, self.args = root, keys, args
        self.token = secrets.token_hex(32)
        self.lock = threading.Lock()
        self.calls, self.cost, self.tokens = 0, 0.0, 0
        self.reserved = 0.0
        self.exhausted = threading.Event()
        self.records = []
        for path in sorted((root / 'api').glob('metrics-*.json')):
            record = json.loads(path.read_text())
            self.records.append(record)
            self.calls = max(self.calls, record['call'] + 1)
            self.cost += record.get('estimated_cost_usd', 0)
            self.tokens += record.get('reported_tokens', 0)
        for path in sorted((root / 'api').glob('request-*.json')):
            index = int(path.stem.split('-')[1])
            self.calls = max(self.calls, index + 1)
            if not (root / 'api' / f'metrics-{index:04d}.json').exists():
                request = json.loads(path.read_text())
                self.cost += path.stat().st_size * .30 / 1e6 + request['max_tokens'] * 1.20 / 1e6

    def redact(self, value):
        return M.redact(value, {**self.keys, 'relay': self.token})

    def summary(self):
        with self.lock:
            return {'requests': self.calls, 'reported_tokens': self.tokens,
                    'estimated_cost_usd': self.cost, 'budget_exhausted': self.exhausted.is_set(),
                    'errors': sum(bool(r.get('error')) for r in self.records)}

    def server(self):
        relay = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def reply(self, status, body, content_type='application/json'):
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def fail(self, code, message):
                self.reply(code, json.dumps({'type': 'error', 'error': {
                    'type': 'invalid_request_error', 'message': message}}).encode())

            def do_POST(self):
                if (self.headers.get('Authorization') != 'Bearer ' + relay.token
                        and self.headers.get('x-api-key') != relay.token):
                    return self.fail(401, 'Invalid local relay credential')
                if self.path.split('?')[0] != '/v1/messages':
                    return self.fail(404, 'Only the Messages API is available')
                size = int(self.headers.get('Content-Length', 0))
                if not 0 < size <= 64 * 1024**2:
                    return self.fail(400, 'Invalid request size')
                body = json.loads(self.rfile.read(size))
                body['model'] = MODEL
                body['max_tokens'] = min(int(body.get('max_tokens', 32768)), relay.args.max_output_tokens)
                body['output_config'] = {'effort': 'max'}
                body['thinking'] = {'type': 'adaptive'}
                encoded = json.dumps(body).encode()
                reservation = len(encoded) * .30 / 1e6 + body['max_tokens'] * 1.20 / 1e6
                with relay.lock:
                    if (relay.calls >= relay.args.max_calls or relay.tokens >= relay.args.max_tokens
                            or relay.cost + relay.reserved + reservation > relay.args.max_cost):
                        relay.exhausted.set()
                        return self.fail(400, 'Local experiment API budget exhausted; preserve workspace and stop')
                    index = relay.calls
                    relay.calls += 1
                    relay.reserved += reservation
                M.write_json(relay.root / 'api' / f'request-{index:04d}.json', relay.redact(body))
                started = time.monotonic()
                record = {'call': index, 'model': MODEL, 'effort': 'max', 'started_at': time.time()}
                try:
                    with httpx2.Client(timeout=600) as client:
                        response = client.post('https://api.fireworks.ai/inference/v1/messages', content=encoded,
                            headers={'Authorization': 'Bearer ' + relay.keys['FIREWORKS_API_KEY'],
                                     'Content-Type': 'application/json', 'anthropic-version': '2023-06-01'})
                    data = response.content
                    content_type = response.headers.get('content-type', 'application/json')
                    record['status_code'] = response.status_code
                    if response.is_success:
                        usage = usage_from_response(data, 'text/event-stream' in content_type)
                        record['usage'] = usage
                        uncached = int(usage.get('input_tokens') or 0) + int(usage.get('cache_creation_input_tokens') or 0)
                        cached = int(usage.get('cache_read_input_tokens') or 0)
                        output = int(usage.get('output_tokens') or 0)
                        record['estimated_cost_usd'] = (uncached*.30 + cached*.006 + output*1.20) / 1e6
                        record['reported_tokens'] = uncached + cached + output
                    else:
                        record['error'] = data.decode(errors='replace')
                    safe = relay.redact({'text': data.decode(errors='replace')})['text'].encode()
                    (relay.root / 'api' / f'response-{index:04d}.txt').write_bytes(safe)
                    self.reply(response.status_code, safe, content_type)
                except Exception as exc:
                    record['error'] = relay.redact({'error': f'{type(exc).__name__}: {exc}'})['error']
                    self.fail(502, record['error'])
                finally:
                    record['latency_seconds'] = time.monotonic() - started
                    record = relay.redact(record)
                    with relay.lock:
                        relay.reserved -= reservation
                        relay.cost += record.get('estimated_cost_usd', 0)
                        relay.tokens += record.get('reported_tokens', 0)
                        relay.records.append(record)
                    M.write_json(relay.root / 'api' / f'metrics-{index:04d}.json', record)
                    M.write_json(relay.root / 'api-summary.json', relay.summary())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--formal-episodes', type=int, default=150)
    parser.add_argument('--formal-workers', type=int, default=2)
    parser.add_argument('--max-calls', type=int, default=1000)
    parser.add_argument('--max-tokens', type=int, default=50_000_000)
    parser.add_argument('--max-output-tokens', type=int, default=32768)
    parser.add_argument('--max-cost', type=float, default=5.0)
    parser.add_argument('--max-seconds', type=int, default=14400)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    os.environ['RAX_ARTIFACT_ROOT'] = str(root.parent)
    root.mkdir(parents=True, exist_ok=args.resume, mode=0o700)
    keys = M.load_keys()
    if not keys.get('FIREWORKS_API_KEY'):
        raise SystemExit('No locally configured Fireworks API key')
    config = make_configuration(root, args.episodes, args.formal_episodes, args.formal_workers)
    if not args.resume:
        M.write_json(root / 'operator-input.json', json.loads(json.dumps(asdict(config), default=str)))
    relay = Relay(root, keys, args)
    server = relay.server()
    process = None
    started = time.monotonic()
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        with tempfile.TemporaryDirectory(prefix='rax-relay-', dir=Path.home()/'.config/kimirax') as temp:
            credential = Path(temp) / 'relay-token'
            credential.write_text(relay.token)
            credential.chmod(0o600)
            command = [str(M.ROOT / '.venv/bin/python'), '-m', 'harness.codex_cli.auto_research',
                '--config', str(root / 'operator-input.json'), '--non-interactive',
                '--claude-model', MODEL, '--claude-key-file', str(credential),
                '--claude-base-url', f'http://127.0.0.1:{server.server_port}']
            if args.resume:
                login = root / 'agent-login.json'
                value = json.loads(login.read_text())
                value.update(key_file=str(credential), base_url=f'http://127.0.0.1:{server.server_port}')
                M.write_json(login, value)
                command.append('--resume')
            environment = dict(os.environ, PYTHONPATH=str(M.ROOT / 'rax'), PYTHONUNBUFFERED='1')
            M.write_json(root / 'replication.json', {
                'adapter': 'custom_macos_port_and_metered_api_relay', 'model': MODEL,
                'harness': 'RAX native Claude Code auto_research', 'target': 'Opus DexJoCo protocol',
                'deviations': ['DeepSeek via Fireworks replaces Opus', 'max effort maps xhigh to max on Fireworks',
                    'optional Gemini perception disabled: no key', 'macOS GLFW and Seatbelt replace EGL and bubblewrap',
                    f'{args.formal_workers} formal workers rather than four',
                    'broad action target bounds added', 'API spend, token, call and session wall-time caps'],
                'caps': vars(args) | {'output': str(root)}, 'started_at': time.time()})
            with (root / 'agent.log').open('ab') as log:
                process = subprocess.Popen(command, cwd=M.ROOT / 'rax', env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    if relay.exhausted.is_set() or time.monotonic() - started > args.max_seconds:
                        process.terminate()
                        break
                    time.sleep(1)
                process.wait(timeout=180)
    except KeyboardInterrupt:
        if process and process.poll() is None:
            process.terminate()
            process.wait(timeout=180)
    finally:
        if process and process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        server.shutdown()
        M.write_json(root / 'run-summary.json', relay.summary() | {
            'returncode': process.returncode if process else None,
            'wall_seconds': time.monotonic()-started,
            'formal_result': json.loads((root/'formal.json').read_text()) if (root/'formal.json').exists() else None})
    print(json.dumps(json.loads((root/'run-summary.json').read_text()), indent=2))


if __name__ == '__main__':
    main()
