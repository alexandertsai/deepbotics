#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Requires an Apple Silicon Mac' >&2; exit 1; }
command -v uv >/dev/null || { echo 'Install uv: https://docs.astral.sh/uv/getting-started/installation/' >&2; exit 1; }

ensure_source() {
    local directory="$1" url="$2" branch="$3" revision="$4"
    if [[ ! -d "$directory/.git" ]]; then
        git clone --depth 1 --branch "$branch" "$url" "$directory"
        if [[ "$(git -C "$directory" rev-parse HEAD)" != "$revision" ]]; then
            git -C "$directory" fetch --depth 1 origin "$revision"
            git -C "$directory" checkout --detach "$revision"
        fi
    fi
    [[ "$(git -C "$directory" rev-parse HEAD)" == "$revision" ]] || { echo "$directory differs from the pinned revision" >&2; exit 1; }
}

ensure_source rax https://github.com/robot-agent-exploration/rax.git dexjoco/native 4b577aa2ba0039ad01707ede236be9913274180e
ensure_source dexjoco https://github.com/brave-eai/dexjoco.git main 8d23b0fab23b17a58c4b55f3942e17013aaf8267
if ! git -C rax apply --reverse --check ../patches/rax-macos.patch 2>/dev/null; then
    git -C rax apply --check ../patches/rax-macos.patch
    git -C rax apply ../patches/rax-macos.patch
fi
if [[ ! -x .venv/bin/python ]]; then
    uv venv --python 3.11 .venv
fi
uv pip sync --python .venv/bin/python requirements-macos.lock
.venv/bin/python -c 'import platform,sys; assert platform.machine()=="arm64" and sys.version_info[:2]==(3,11); import mujoco,gymnasium,numpy; print("Apple Silicon environment ready")'
