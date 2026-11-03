# deepbotics

Robotics evaluation scripts for DexJoCo on Apple Silicon and RLE-Bench.

## Source checkouts

This repository tracks the evaluation scripts, environment lockfile, and local
RAX patch. Upstream repositories are separate local checkouts excluded by
`.gitignore`, along with virtual environments, credentials, and generated outputs.

`bash scripts/setup_macos.sh` clones the pinned RAX and DexJoCo sources and applies
`patches/rax-macos.patch`. The source revisions used in this workspace are:

| Directory | Upstream | Revision |
| --- | --- | --- |
| `rax` | https://github.com/robot-agent-exploration/rax | `4b577aa2ba0039ad01707ede236be9913274180e` |
| `dexjoco` | https://github.com/brave-eai/dexjoco | `8d23b0fab23b17a58c4b55f3942e17013aaf8267` |
| `vendor/rle-bench` | https://github.com/RLE-Bench/RLE-Bench | `03d36e50ced2105397ee7badfbbc66d7b76aedea` |
| `bench2dex` | https://github.com/Bench2Dex/Bench2Dex | `54ce60e48baaf140c5a1731285aefe95e7390652` |
| `bench2dex-leaderboard` | https://github.com/Bench2Dex/Bench2Dex-Leaderboard | `dfe3b3b50025dda811c4f9e8886e130cf8259a4f` |

For the RLE-Bench pilots, also prepare the upstream checkout before running the
task-specific `prepare` commands below:

```bash
mkdir -p vendor
git clone https://github.com/RLE-Bench/RLE-Bench.git vendor/rle-bench
git -C vendor/rle-bench checkout 03d36e50ced2105397ee7badfbbc66d7b76aedea
```

The Bench2Dex checkouts are reference sources and are optional for these scripts.

## DexJoCo fold_glasses on Apple Silicon

The simulator is installed and verified in `.venv`, created with `uv venv --python 3.11 .venv`.
Claude Code is now available as the agent harness through Fireworks. The private
Fireworks key is stored outside this checkout. Direct Moonshot and DeepSeek API keys
remain absent; the original direct-provider preflight below is retained as historical evidence.

## Task 08 pilot

`scripts/task08_pilot.py` prepares a short RLE-Bench Task 08 comparison between
GPT-6 Astra through the local Codex login and DeepSeek V4.1 Flash 1017 through
the existing Fireworks key. Docker Desktop must be running. The upstream checkout
is staged under `vendor/rle-bench`; the existing DexJoCo environment is unchanged.

```bash
.venv/bin/python scripts/task08_pilot.py preflight
.venv/bin/python scripts/task08_pilot.py prepare
.venv/bin/python scripts/task08_pilot.py run --output outputs/task08-pilot-01
```

Each condition receives the same reference robot and controller, 15 minutes of
development, and at most three development probes. A probe tests all three arms
on the static and acceleration/braking scenarios at development seed 11, plus
three shelf targets at both payloads. Failed probes also consume the allowance.
Both frozen submissions are then evaluated once by the unchanged full official
verifier; final verification has a separate timeout and may take substantially
longer than development. Final results are never fed back to either agent.

This is an improvement pilot from a supplied reference, not a reproduction of
the published from-scratch leaderboard. The DeepSeek condition adapts RAX's
evidence and change-discipline instructions, MCP bridge, and content-addressed
artifact storage. It does not implement the complete upstream RAX backend or
exploration/compile/formal lifecycle. The comparison measures the two configured
systems; a DeepSeek baseline without the RAX adaptation would be needed to
attribute any improvement to RAX.

DeepSeek is limited to 12 API requests with at most 8192 output tokens each.
The reused relay also has a $2 estimated-spend guard based on historical rates;
this estimate is not a provider-enforced billing cap. Astra uses the signed-in
Codex account and its applicable usage limits. No credentials are placed in the
agent prompts or artifact workspaces. Agent tools expose public assets, candidate
edits, development probes, and submission; verifier source stays outside that
interface. Controllers execute in the verifier's existing isolated Linux worker.

`outputs/task08-setup/` records setup checks. Each run writes `protocol.json`,
per-condition transcripts, version manifests, development probes, frozen
artifacts, and `summary.json`. An unchanged starter or an agent startup failure
must not be interpreted as a successful model improvement. Host boundary checks:

```bash
.venv/bin/python -m unittest discover -s scripts -p test_task08_pilot.py -v
```

## Four-worker Task 08 pilot

```bash
.venv/bin/python scripts/task08_swarm.py --output outputs/task08-swarm-01
```

This requires the saved Task 08 reference baseline and single-agent run. Four
DeepSeek workers use separate candidate files and the same RAX method instructions,
MCP bridge, memory tools, and frozen artifact storage as the earlier pilot. Their
shared allowance is 15 minutes, 12 provider requests, a 250,000 reported-token
stop threshold, and three new development probes. Each worker may make at most
three requests. Provider requests run in FIFO order, and one simulator runs at a
time with four CPUs and 4 GB; the host CLI processes add separate overhead.

Workers share measured probe results and separately labelled notes. Identical
artifacts reuse a cached probe. The coordinator selects the highest measured
partial development score and retains the starter on ties or regressions.
Hidden final results do not participate in selection. If the selected artifact
is byte-identical to the already scored reference and the verifier image matches,
the existing final result is reused after selection; otherwise it is scored once.

The saved single-agent result is an exploratory comparison. Prompt inputs,
cached starter feedback, and incumbent selection differ, so a score difference
cannot be attributed solely to four workers. `protocol.json` records these
differences; `summary.json` records actual worker activity and API usage.

The first four-worker trial exhausted three requests per worker on inspection
and produced no changes. A larger run uses configurable shared allowances:

```bash
.venv/bin/python scripts/task08_swarm.py --output outputs/task08-swarm-next \
  --seconds 1800 --max-calls 48 --calls-per-worker 12 \
  --max-tokens 3000000 --max-output-tokens 16384 --max-cost 3 \
  --tests 8 --tests-per-worker 2
```

Use a fresh output directory. Each prompt includes both starter files, and
workers are asked to implement and test within their first four requests.
Budget notices preserve the cacheable context prefix. After workers stop,
the coordinator can rehearse their frozen final candidates using remaining
shared time and probe allowances. Final verification runs afterward.

## Task 06 and Task 08 comparison

The final retry used fourteen DeepSeek agents per task, prepared starting candidates
from our own DeepSeek work and public inputs, and a rebuilt harness with asynchronous
validation and shared test data. Each task had a 15-minute development budget; official
scoring ran afterward. Astra references use one raw native Codex agent each.

| Task | DeepSeek initial | DeepSeek final | Astra | Winner |
| --- | ---: | ---: | ---: | --- |
| 06 RGB-D pose estimation | 13.30 | 32.08 | 89.53 | Astra |
| 08 mobile-base design | 38.39 | 58.19 | 63.71 | Astra |

Scores, token counts, cost estimates and artifacts are recorded in
[Task 06 comparison](benchmarks/task06-comparison.json) and
[Task 08 comparison](benchmarks/task08-comparison.json).
The initial comparisons remain in
[Task 06 initial](benchmarks/task06-comparison-initial.json) and
[Task 08 initial](benchmarks/task08-comparison-initial.json).
The saved single-agent references are
[Task 06 SOTA](benchmarks/task06-sota.json) and [Task 08 SOTA](benchmarks/task08-sota.json).

## Task 09 public-starter pilot

Fourteen DeepSeek agents with a custom harness scored **61.10**, beating one
raw native Codex Astra agent at **53.15**. The saved project "SOTA" reference is
[`benchmarks/task09-sota.json`](benchmarks/task09-sota.json). Scores, timing,
token usage and inference cost estimates are recorded in
[`benchmarks/task09-comparison.json`](benchmarks/task09-comparison.json).

```bash
.venv/bin/python scripts/task09_pilot.py run --output outputs/task09-pilot-next
```

Task 09 uses the public bare models for all three GELLO variants with zero-torque
trim stubs. The unchanged starter is scored explicitly. Each agent receives
15 minutes and three public nominal probes, followed by the unchanged hidden
verifier. It uses the same DeepSeek API limits and Astra login as Task 08.
The completed first pilot is in `outputs/task09-pilot-01/summary.json`.

Run the four-worker Task 09 variant with a shared 30-minute development budget:

```bash
.venv/bin/python scripts/task09_swarm.py --output outputs/task09-swarm-next
```

Defaults are 48 provider requests, 12 per worker, a 3-million reported-token stop
threshold, eight new probes (two per worker), and a $3 estimated API guard.
An unchanged-starter public probe runs before the development clock. Selection
uses negative mean nominal/adapted hold RMS error across all three variants,
subject to successful probes and the public mass budget. This proxy omits private
validity and robustness criteria. The selected frozen candidate receives one
final private evaluation. The summary records API estimates, development time,
starter probe time, and final verification time. Local simulation and subscription
costs are excluded from API estimates.

For eight workers with eight concurrent model requests and the same shared budget:

```bash
.venv/bin/python scripts/task09_swarm.py --output outputs/task09-eight-next \
  --workers 8 --api-concurrency 8 --calls-per-worker 6 --tests-per-worker 1
```

Simulation remains serial. Concurrent requests reserve estimated tokens and cost
before admission, then settle against reported usage. The simulator container's
four-CPU/4-GB limit excludes the host CLI processes. Concurrency, worker count and
per-worker allowances differ together, so this is an exploratory comparison.

For fifteen workers using three active worker slots and paced API requests:

```bash
.venv/bin/python scripts/task09_swarm.py --output outputs/task09-fifteen-next \
  --workers 15 --active-workers 3 --api-concurrency 3 --paced \
  --seconds 2700 --max-cost 5 --max-tokens 6000000 \
  --max-output-tokens 8192 \
  --max-calls 270 --calls-per-worker 18 --logical-calls-per-worker 6 \
  --tests 15 --tests-per-worker 1
```

The request limits count provider attempts, including retries; each worker gets
six logical model requests, with up to three provider attempts per request.
The initial shared pacing ceilings are 120k total prompt, 120k uncached prompt,
and 30k generated tokens per minute. The pacer reserves estimated tokens, reads
provider limit headers, and pauses on throttling. The dollar guard is an API
estimate, including retry accounting, rather than a provider invoice limit.

The prepared iterative follow-up uses three probes and twelve logical requests
per worker, a public proxy including passive residual and mean servo headroom,
and bounded shared status. Its prepared command is saved in
`outputs/task09-followup-plan/config.json`; it has not been launched automatically.
The command enables `--paced --adaptive-pacing`: rates rise by 10% after two clean
minutes and at least three successes, remain at most 80% of reported limits, and
back off toward initial targets on 429s with cooldown and retries. Unknown input
limits use explicit experimental ceilings. The $5 estimate guard stays fixed.
Raw limit headers and rate adjustments are logged separately.

Astra pilot runs save `astra/token-usage.json` with input, cached-input, output,
and reasoning counts from Codex completion events or retained session counters.
Cached input and reasoning output are subsets, not extra tokens. Interrupted
session counts are labelled partial. The report includes a separate hypothetical
standard API cost using dated rates and context-tier scenarios; it does not infer
a subscription charge. Historical ephemeral runs without usage events remain
unknown. Rebuild a usage report with:

```bash
.venv/bin/python scripts/codex_usage.py outputs/task09-pilot-next/astra
```

## Claude Code harness through Fireworks

`scripts/claude_mac_evaluation.py` runs the installed Claude Code CLI as the real agent
loop, with RAX's existing `DexjocoBackend`, RGB-only MCP packets, action validators,
official simulator process, private operator recording, and `harness/mcp_bridge.py`.
It also uses RAX's custom-provider credential routing to pin every Claude model alias
to the selected Fireworks model. This is a **minimal custom macOS launcher**, rather
than a migration of RAX's full exploration/compile/formal workflow.

The agent has exactly four RAX robot MCP tools. Built-in file, shell, web, and subagent
tools are disabled. Claude Code uses `--bare`, a separate home/config directory, and
an empty agent workspace; the user's personal Claude login is not used. No model input
contains object state, operator reports, simulator source, or credentials.

Fireworks exposes an official Anthropic-compatible endpoint at
`https://api.fireworks.ai/inference/v1/messages`. A local relay enforces the chosen
model, six-call budget, 4096-output-token request cap and low reasoning effort, and
records actual provider requests, responses, latency and usage. The agent only receives
a temporary local relay credential; the Fireworks key remains in the host process.
Responses retain Fireworks' Anthropic protocol, including image/tool blocks and SSE;
no protocol translation is applied. The relay buffers each response to record it.

The selected DeepSeek model is `accounts/fireworks/models/deepseek-v4p1-flash-1017`.
Fireworks documents vision, tool calling, and serverless support for it. Its
`deepseek-v4-pro-0813` variant has no image input and is only available on demand, so it
does not fit this camera-based test under the no-provisioning constraint. Kimi uses
Fireworks' serverless vision model `accounts/fireworks/models/kimi-k3`.

Both short episodes have now run with seed 0, identical initial camera hashes/robot
state, and identical budgets. Every provider request returned HTTP 200.

| Model | API calls | Executed steps | Rejected motion calls | Task completed | Input/output tokens | Mean API latency |
| --- | --- | --- | --- | --- | --- | --- |
| DeepSeek V4.1 Flash 1017 | 6 | 0 | 3 | No | 16,309 / 5,673 | 6.00 s |
| Kimi K3 | 6 | 30 | 3 | No | 29,538 / 5,540 | 14.59 s |

Input token totals include cache hits. Kimi executed two valid motion chunks.
DeepSeek's three motion requests were rejected for hand limits or excessive translation.
Both stopped at the test budgets; neither is a completed full benchmark episode.
`outputs/mac-evaluation/claude-fireworks-comparison.json` holds the paired results.

```bash
cd /Users/yihrae/Documents/Projects/kimirax
.venv/bin/python scripts/enter_api_keys.py --provider fireworks
.venv/bin/python scripts/claude_mac_evaluation.py --model deepseek \
  --layout 0 --max-calls 6 --max-steps 120 --max-output-tokens 4096 \
  --max-total-tokens 80000 --timeout 180
.venv/bin/python scripts/claude_mac_evaluation.py --model kimi \
  --layout 0 --max-calls 6 --max-steps 120 --max-output-tokens 4096 \
  --max-total-tokens 80000 --timeout 180
.venv/bin/python -m pytest -q scripts/test_mac_evaluation.py scripts/test_claude_mac_evaluation.py
```

Saved DeepSeek evidence is in `outputs/mac-evaluation/claude-fireworks-deepseek/`;
Kimi evidence is in `outputs/mac-evaluation/claude-fireworks-kimi/`. Each run contains
`launch.json`, `claude-transcript.jsonl`, `claude-stderr.txt`, `api/request-*.json`,
`api/response-*.txt`, `api/metrics-*.json`, `tools/*.json`, `images/`, `rax/`, and
`result.json`. Tool records include requested actions, validation errors, and RAX's
executed step/action transition when an action is accepted. RAX also retains video and
intermediate camera frames under `native/workspace/runtime/frames/`.

When Claude attempts a seventh provider call, the local relay returns an explicit
budget error without contacting Fireworks. Claude Code then exits with an error code;
`budget_reached: true` identifies this expected evaluator stop. All paid calls and
provider failures are recorded separately. An `unrecognized_model` CLI diagnostic is
expected for Fireworks model IDs and does not imply a failed provider request.
Claude Code's own `error_max_turns` stop also uses exit code 1; it is classified as
`status: budget_exhausted` with `stop_reason: claude_turn_budget` in the controller result.

Official references:

- https://docs.fireworks.ai/nexus/harnesses/claude-code.md
- https://docs.fireworks.ai/tools-sdks/anthropic-compatibility
- https://fireworks.ai/models/deepseek-ai/deepseek-v4p1-flash-1017
- https://fireworks.ai/models/deepseek-ai/deepseek-v4-pro-0813
- https://fireworks.ai/models/fireworks/kimi-k3
- https://code.claude.com/docs/en/cli-reference

## Original direct-provider preflight results

Validated artifacts are in `outputs/mac-evaluation/validated-preflight/`.

| Check | Result |
| --- | --- |
| Official scene loading | Passed |
| Front and wrist RGB cameras | Passed, 640 x 640 x 3 each; inspected visually |
| Reset repeatability | Passed, identical robot states and PNG hashes across fresh processes |
| Official hold | Passed, 30 steps per repeat |
| Bounded RAX action chunk | Passed, 3 steps per repeat |
| Kimi hosted episode | Blocked: `MOONSHOT_API_KEY` missing; 0 model calls |
| DeepSeek hosted episode | Blocked: `DEEPSEEK_API_KEY` missing; 0 model calls |
| Identical model initial conditions | Passed, same seed, robot state and camera hashes |
| Local adapter tests | 19 passed, including offline SDK success/error contracts for both providers |
| RAX backend unit tests | 19 passed |

`comparison.json` records blocked status and null model success. The simulator's early-close
`native_score: 0` is not a completed model evaluation or an official benchmark failure.
No hosted responses, latency measurements or token usage exist yet because no API requests
were sent. The hold/action checks did not achieve task success, as expected.

The workspace was initially empty. No `outputs/mac-evaluation/` directory or exploratory
scripts existed here or under `/Users/yihrae/Documents/Projects` when inspected. The earlier
`initial-smoke/` and `hosted-preflight/` directories are retained; `validated-preflight/`
includes the final bounded-action checks.

## Run

From this directory:

```bash
cd /Users/yihrae/Documents/Projects/kimirax
bash scripts/setup_macos.sh
.venv/bin/python scripts/mac_evaluation.py smoke
```

Create API keys through the official platforms:

- Kimi: https://platform.kimi.ai/console/api-keys
- DeepSeek: https://platform.deepseek.com/

Enter keys in a local terminal using hidden prompts:

```bash
.venv/bin/python scripts/enter_api_keys.py
.venv/bin/python scripts/mac_evaluation.py credentials
```

The entry script stores keys outside this checkout in
`~/.config/kimirax/api_keys.json` with directory permissions 700 and file permissions 600.
Blank input keeps an existing value. The credential check prints presence only.
Never paste keys into chat, command-line arguments, source files or shell history.
Environment variables `MOONSHOT_API_KEY` and `DEEPSEEK_API_KEY` are also supported;
`KIMI_API_KEY` is accepted as a fallback. Environment values take precedence over the file.
The runner rejects a credential file accessible to other users and redacts known keys
from saved responses and errors. `.gitignore` excludes credentials and generated outputs.

Run one short episode per model:

```bash
.venv/bin/python scripts/mac_evaluation.py evaluate --provider both --layout 0 \
  --max-calls 6 --max-steps 120 --max-output-tokens 4096 \
  --max-total-tokens 80000 --timeout 180
```

Every evaluation first repeats the simulator smoke test without API calls. Both models
then start in fresh simulator processes at official seed 0, episode index 0, using
`configs/rand_obj/fold_glasses.yaml`. Change `--provider` to `kimi` or `deepseek` for an
individual rerun. Output directories use timestamps; `--output PATH` requires a new path
and never overwrites a previous run. Exit status 2 means a missing key, API error or
runner error. Exit status 1 means smoke verification failed.

Run validation:

```bash
.venv/bin/python -m pytest -q scripts/test_mac_evaluation.py
DEXJOCO_SOURCE="$PWD/dexjoco" PYTHONPATH="$PWD/rax" \
  .venv/bin/python -m pytest -q rax/local_tests/unit/test_dexjoco_backend.py
```

## Integration and macOS changes

This is a **minimal custom hosted API adapter**, implemented in `scripts/mac_evaluation.py`.
It uses RAX's existing `DexjocoProcess`, identity-checked simulator RPC, action/state
layouts, request validators, official wrapper, hold implementation, recording and scoring.
Its exposed tools are `dexjoco_step`, `dexjoco_hold`, `dexjoco_observe` and `dexjoco_status`.
It does not run RAX's exploration/compile/formal agent workflow: that workflow and its
Kimi/DeepSeek CLI launchers require Linux bubblewrap and NVIDIA-oriented host setup.
No model-generated code or shell commands are executed by this adapter.

`patches/rax-macos.patch` changes only RAX's rendering environment:

- Darwin selects GLFW and removes EGL/CUDA device variables, avoiding NVIDIA discovery.
- Direct simulator-server startup defaults to GLFW on Darwin.
- Linux retains the original EGL selection.

The `uv` setup uses native macOS arm64 wheels instead of RAX's Linux x86_64 Miniforge
installer, conda environments, NVIDIA EGL setup, CPU Torch index and bubblewrap workflow.
The required simulator package versions remain official: MuJoCo 3.4.0, Gymnasium 1.0.0,
NumPy 1.26.4, and DexJoCo's OpenPI client. `requirements-macos.lock` records the complete
installed environment. A macOS graphical login session is required for GLFW's OpenGL
context even though image rendering is offscreen.

Pinned sources:

- RAX `dexjoco/native`: `4b577aa2ba0039ad01707ede236be9913274180e`
- DexJoCo `main`: `8d23b0fab23b17a58c4b55f3942e17013aaf8267`, also RAX's pinned revision

DexJoCo source, assets, controller gains, timing, dynamics, reset and success logic are
unchanged. The official success function requires both temple hinges above 1.1 radians
and the glasses inside the official case bounds for 50 consecutive steps. The runner
reads RAX's recorded official outcome; it does not implement a substitute success check.

## API configuration

Official documentation was checked on November 1, 2026:

| Provider | Documented model ID | Base URL | Configuration |
| --- | --- | --- | --- |
| Kimi | `kimi-k3.1-pro` | `https://api.moonshot.ai/v1` | `reasoning_effort=low`, `max_completion_tokens=4096` |
| DeepSeek | `deepseek-flash` | `https://api.deepseek.com` | thinking enabled, `reasoning_effort=low`, `max_tokens=4096` |

Kimi's current quickstart names K3.1 Pro and documents native vision; it always reasons.
DeepSeek's vision guide names `deepseek-flash`; its pricing page identifies the served
version as DeepSeek-V4.1-Flash-1017. The Pro variant does not accept images. Actual
authenticated availability and account balance remain unverified until keys are supplied.

References:

- https://platform.kimi.ai/docs/get-your-api-key
- https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart
- https://platform.kimi.ai/docs/guide/use-kimi-vision-model
- https://api-docs.deepseek.com/guides/vision/
- https://api-docs.deepseek.com/guides/thinking_mode/
- https://api-docs.deepseek.com/api/create-chat-completion/
- https://api-docs.deepseek.com/quick_start/pricing/
- https://github.com/robot-agent-exploration/rax/tree/dexjoco/native
- https://github.com/brave-eai/dexjoco

## Bounds, model inputs and artifacts

Both models receive the same instruction, tool schemas, front/wrist RGB images,
23-value robot proprioception and tool feedback. The adapter constructs these inputs
from an explicit allowlist. Object poses, hinge sensors, simulator state tails, private
goal reports and other operator files are never included. Full assistant tool-call and
reasoning fields are preserved in conversation history as required by the API protocols.

Action rows must contain 22 finite numbers. A chunk has at most 20 rows and must fit the
remaining step budget. Hand targets must respect official XML joint limits. Flange targets
are bounded to x/y [-0.85, 0.85] m and z [0.55, 1.8] m, with rotation-vector norm at most pi.
Target changes are limited to 0.025 m translation, 0.25 rad geodesic orientation and
0.25 rad per hand joint per step. Validation checks the entire chunk before dispatch;
it rejects invalid actions instead of silently clipping them. Only one tool call is
allowed per response. Invalid calls consume a model call but no physics steps.

Physics pauses during API inference, following RAX's synchronous evaluation semantics.
The official 30 Hz control pacing and 0.02 s physics increment are preserved. This is a
short integration test, not the full 150-episode RAX benchmark. API sampling is not
deterministic. Calls have no automatic retries. The reported 80,000-token threshold is
checked between calls and may be exceeded by the final request; each request has its own
hard output-token cap, and the call count is bounded. Budgets include reasoning output.

Each output directory contains:

- `manifest.json`: source revisions, provider configuration and input policy.
- `smoke/`: reset/hold/action images, executed actions and smoke results.
- `<provider>/images/`: initial images and observations, every tenth action frame and final frames.
- `<provider>/request-*.json`: actual model messages and tools, without authentication headers.
- `<provider>/response-*.json`: full responses, usage and measured request latency.
- `<provider>/actions-*.json`: executed actions with simulator step IDs.
- `<provider>/tool-result-*.json`: accepted/rejected tool feedback and errors.
- `<provider>/result.json`: status, success, budgets, latency, usage and errors.
- `<provider>/rax/`: RAX simulator logs, video and official outcome; private operator reports stay here.
- `comparison.json`: both results and initial-condition hash comparisons.

Missing-key runs contain initial images and blocked results, with no fabricated responses
or actions. Hosted calls may use an existing account balance; no credit purchase or cloud
hardware provisioning is performed by these scripts.
# DeepSeek replication of the Opus protocol

`scripts/replicate_deepseek.py` uses RAX's native Claude Code exploration,
registration, rehearsal, submission and frozen formal evaluation. The custom
adapter supplies macOS process/sandbox support and a metered Fireworks relay.
It does not supply a task controller or privileged simulator observations.

Prepare the environment and verify the official goal-image demonstrations:

```bash
bash scripts/setup_macos.sh
DEXJOCO_SOURCE="$PWD/dexjoco" DEXJOCO_PYTHON="$PWD/.venv/bin/python" \
  .venv/bin/python rax/scripts/setup_dexjoco_demos.py \
  --gpu mac --task fold_glasses --candidates 8
.venv/bin/python -m pytest scripts/test_rax_macos_workflow.py -q
```

Start a fresh run with the locally stored Fireworks credential:

```bash
.venv/bin/python scripts/replicate_deepseek.py \
  --output outputs/mac-evaluation/deepseek-opus-protocol \
  --episodes 32 --formal-episodes 150 --formal-workers 2 --max-cost 5
```

The run uses `accounts/fireworks/models/deepseek-v4p1-flash-1017`, maximum
reasoning effort, 640x640 front/wrist RGB-D, calibration and backprojection,
three verified terminal goal images, public task instructions, one persistent
workspace, one environment and no subagents. Exploration seeds are 1000–1031;
formal evaluation uses seeds 0–2, 50 consecutive resets each. Task physics and
official success checks are unchanged. Each episode permits 1200 control steps.

Differences from the Opus paper are recorded in `replication.json`: DeepSeek on
Fireworks, optional Gemini perception off because no key is configured, GLFW
rendering and Seatbelt isolation on macOS, two formal workers, broad validation
bounds on action targets, and API/session spending limits. Default limits are
$5 estimated API usage, 1000 requests, 50 million reported tokens, 32768 output
tokens per request and four hours for the whole run. No credits are purchased.
These limits can end an experiment before all exploration episodes are used.

Inspect `api-summary.json`, `agent-lifecycle.json`, `state.json`, `agent.log`,
`published/runs/` and `published/formal_batch.json` in the output directory.
API requests, responses, latency, token counts and estimated costs are in `api/`.
RAX saves camera frames, actions, tool errors, rehearsals and official outcomes.
Private simulator diagnostic files are outside the agent-visible published tree.
The model receives only goal images, camera observations, geometry calibration,
robot proprioception and the public task information.

For an interrupted session, repeat the same command with `--resume`. API usage
limits apply cumulatively to the saved metrics. Completed formal tests are never
silently retried. To run a separate experiment, choose a new output directory.
