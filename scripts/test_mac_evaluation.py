import json
from types import SimpleNamespace

import numpy as np
import pytest

import mac_evaluation as M
from services.simulator.io import RejectedBeforeAction


@pytest.fixture
def robot_state():
    return [-0.44, 0.0, 1.517, 0, 1, 0, 0, *([0.0] * 12), 0.263, 0, 0, 0]


def test_valid_hold_row_and_official_joint_limits(robot_state):
    row = M.hold_row(robot_state, False)
    M.validate_bounded_actions([row], robot_state, remaining=1)


@pytest.mark.parametrize("change", ["nan", "infinity", "wrong_width", "joint", "translation", "rotation", "workspace", "bool"])
def test_reject_unsafe_actions_before_dispatch(robot_state, change):
    row = M.hold_row(robot_state, False)
    if change == "nan":
        row[0] = float("nan")
    elif change == "infinity":
        row[0] = float("inf")
    elif change == "wrong_width":
        row.pop()
    elif change == "joint":
        row[6] = 10
    elif change == "translation":
        row[0] += 0.03
    elif change == "rotation":
        row[3:6] = [0, 0, 0]
    elif change == "workspace":
        row[2] = 2
    elif change == "bool":
        row[0] = True
    with pytest.raises((ValueError, RejectedBeforeAction)):
        M.validate_bounded_actions([row], robot_state, remaining=20)


def test_reject_over_budget_whole_chunk(robot_state):
    row = M.hold_row(robot_state, False)
    with pytest.raises(ValueError):
        M.validate_bounded_actions([row] * 2, robot_state, remaining=1)
    with pytest.raises(ValueError):
        M.validate_bounded_actions([row] * 21, robot_state, remaining=120)


def test_observation_allowlist_excludes_privileged_fields(tmp_path, robot_state):
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[::2] = 255
    obs = {"front": image, "wrist": image, "state": robot_state,
           "instruction": "Fold the glasses and place them into the case.", "step_id": 0,
           "private_initial_state": [999], "objects": {"glass": [999]}, "reward": 1}
    safe, images = M.save_observation(obs, tmp_path / "images", "observation")
    message = M.user_message(safe, images, tmp_path, 120, 6)
    text = json.dumps(message)
    assert set(safe) == {"state", "instruction", "step_id"}
    assert "private_initial_state" not in text and "objects" not in text and "reward" not in text
    assert len([p for p in message["content"] if p["type"] == "image_url"]) == 2
    with pytest.raises(ValueError):
        M.save_observation({**obs, "state": [*robot_state, 999]}, tmp_path / "images", "bad")


def test_credentials_not_written_in_error_or_response():
    secret = "dummy-test-credential"
    payload = {"message": f"Bearer {secret}", "response": {"content": secret}}
    assert secret not in json.dumps(M.redact(payload, {"MOONSHOT_API_KEY": secret}))


def test_credential_permissions_checked(tmp_path, monkeypatch):
    path = tmp_path / "credentials.json"
    path.write_text('{}')
    path.chmod(0o644)
    monkeypatch.setattr(M, "KEY_PATH", path)
    with pytest.raises(ValueError):
        M.load_keys()


def test_mac_rendering_does_not_probe_nvidia(monkeypatch):
    from services.dexjoco import process
    monkeypatch.setattr(process.sys, "platform", "darwin")
    monkeypatch.setattr(process, "egl_device_index", lambda _: pytest.fail("NVIDIA probe on Mac"))
    env = process.simulator_environment("mac", {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "CUDA_VISIBLE_DEVICES": "0"})
    assert env["MUJOCO_GL"] == "glfw"
    assert "PYOPENGL_PLATFORM" not in env and "CUDA_VISIBLE_DEVICES" not in env


def test_linux_rendering_preserved(monkeypatch):
    from services.dexjoco import process
    monkeypatch.setattr(process.sys, "platform", "linux")
    monkeypatch.setattr(process, "egl_device_index", lambda _: "2")
    env = process.simulator_environment("GPU-example", {})
    assert env["MUJOCO_GL"] == env["PYOPENGL_PLATFORM"] == "egl"
    assert env["MUJOCO_EGL_DEVICE_ID"] == "2"


@pytest.mark.parametrize("provider", ["kimi", "deepseek"])
@pytest.mark.parametrize("api_error", [False, True])
def test_hosted_sdk_contract_offline(tmp_path, monkeypatch, robot_state, provider, api_error):
    import httpx2
    import openai

    key = "test-only-secret"
    cfg = M.PROVIDERS[provider]
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        assert request.headers["authorization"] == f"Bearer {key}"
        assert body["model"] == cfg["model"]
        assert body["reasoning_effort"] == "low"
        assert body["max_completion_tokens" if provider == "kimi" else "max_tokens"] == 4096
        if api_error:
            return httpx2.Response(401, json={"error": {"message": f"Invalid credential {key}", "type": "authentication_error"}})
        return httpx2.Response(200, json={
            "id": "mock-response", "object": "chat.completion", "created": 1, "model": cfg["model"],
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None, "reasoning_content": "Test response",
                "tool_calls": [{"id": "mock-call", "type": "function", "function": {
                    "name": M.T.HOLD, "arguments": '{"steps": 5}'}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140}})

    original = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", lambda **kwargs: original(
        **kwargs, http_client=httpx2.Client(transport=httpx2.MockTransport(respond))))
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    image[::2] = 255

    class FakeSim:
        step_id = 0
        output_dir = tmp_path / "mock-sim"

        def request(self, op, **kwargs):
            if op == "observe":
                return {"front": image, "wrist": image, "state": robot_state, "instruction": "Fold the glasses", "step_id": self.step_id}
            assert op == "hold"
            self.step_id += kwargs["steps"]
            return {"frames": [self.request("observe")], "steps": [{"step_id": self.step_id}], "terminated": False}

        def finish(self, reason):
            M.write_json(self.output_dir / "sim/evaluation_outcome.json", {"native_score": 0, "reason": reason})

        def stop(self):
            pass

    monkeypatch.setattr(M, "new_sim", lambda *args: FakeSim())
    args = SimpleNamespace(layout=0, max_calls=1, max_steps=20, max_output_tokens=4096, max_total_tokens=80000, timeout=10)
    output = tmp_path / "episode"
    result = M.episode(provider, output, args, {cfg["key_env"]: key})
    assert len(requests) == result["model_calls"] == 1
    assert len(result["api_latency_seconds"]) == 1
    if api_error:
        assert result["status"] == "api_error"
        assert result["success"] is None and result["control_steps"] == 0
        assert result["errors"][0]["status_code"] == 401
    else:
        assert result["status"] == "finished_short_episode"
        assert result["reported_total_tokens"] == 140 and result["control_steps"] == 5
        assert (output / "actions-00.json").is_file()
        assert (output / "response-00.json").is_file()
    for path in output.rglob("*.json"):
        assert key not in path.read_text()
