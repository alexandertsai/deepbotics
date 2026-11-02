import json
import threading
import tempfile
from types import SimpleNamespace
from pathlib import Path
import socket

import httpx2
import pytest

import claude_mac_evaluation as C


def test_cli_turn_cap_is_preserved_as_budget_stop():
    row = {"type": "result", "subtype": "error_max_turns", "is_error": True, "num_turns": 7,
           "errors": ["Reached maximum number of turns (6)"]}
    result = C.cli_result("diagnostic\n" + json.dumps(row) + "\n")
    assert result["subtype"] == "error_max_turns" and result["num_turns"] == 7


def test_stream_usage_includes_cached_input_without_double_counting():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 10, "output_tokens": 0, "cache_read_input_tokens": 50}}},
        {"type": "message_delta", "usage": {"input_tokens": 10, "output_tokens": 30, "cache_read_input_tokens": 50}},
    ]
    data = ("\n\n".join("data: " + json.dumps(e) for e in events) + "\n\ndata: [DONE]\n\n").encode()
    assert C.token_total(C.usage_from_response(data, True)) == 90


def test_relay_enforces_model_budget_and_never_forwards_local_auth(tmp_path, monkeypatch):
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer dummy-upstream-key"
        return httpx2.Response(200, json={"usage": {"input_tokens": 20, "output_tokens": 10}, "content": []})

    original = httpx2.Client
    monkeypatch.setattr(C.httpx2, "Client", lambda **kw: original(**kw, transport=httpx2.MockTransport(respond)))
    episode = SimpleNamespace(lock=threading.RLock(), relay_token="dummy-local-token", keys={"FIREWORKS_API_KEY": "dummy-upstream-key"},
        calls=0, args=SimpleNamespace(max_calls=1, max_total_tokens=100, max_output_tokens=4096, timeout=10),
        reported_tokens=0, directory=tmp_path, api_records=[], budget_reached=False)
    server = C.relay_server(episode)
    try:
        with original() as client:
            url = f"http://127.0.0.1:{server.server_port}/v1/messages"
            request = {"model": "wrong-model", "max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 10000}}
            assert client.post(url, json=request).status_code == 401
            headers = {"x-api-key": episode.relay_token}
            assert client.post(url, json=request, headers=headers).status_code == 200
            assert client.post(url, json=request, headers=headers).status_code == 400
        assert len(calls) == 1
        assert calls[0]["model"] == C.MODEL and calls[0]["max_tokens"] == 4096
        assert calls[0]["thinking"]["budget_tokens"] == 1024
        assert episode.reported_tokens == 30 and episode.budget_reached
        for path in tmp_path.rglob("*"):
            if path.is_file():
                assert "dummy-upstream-key" not in path.read_text() and "dummy-local-token" not in path.read_text()
    finally:
        server.shutdown()
        server.server_close()


def test_rax_stdio_bridge_reaches_mcp_without_model_calls(tmp_path):
    import subprocess
    from services.dexjoco import tools as T

    episode = C.Episode.__new__(C.Episode)
    episode.definitions = lambda: [{"name": T.OBSERVE, "inputSchema": {"type": "object", "properties": {}}}]
    short = tempfile.TemporaryDirectory(prefix="rax-test-", dir="/tmp")
    path = Path(short.name) / "mcp.sock"
    server = C.mcp_server(episode, path)
    try:
        messages = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                    {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}]
        result = subprocess.run([str(C.M.ROOT / ".venv/bin/python"), "-I", str(C.M.ROOT / "rax/harness/mcp_bridge.py"), str(path)],
                                input="".join(json.dumps(m) + "\n" for m in messages), capture_output=True, text=True, timeout=5)
        responses = [json.loads(line) for line in result.stdout.splitlines()]
        assert result.returncode == 0
        assert responses[1]["result"]["tools"][0]["name"] == T.OBSERVE
    finally:
        server.shutdown()
        server.server_close()
        short.cleanup()
