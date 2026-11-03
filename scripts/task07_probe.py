"""Short public-seed bin-clearing episodes through the public policy sandbox."""

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, "/workspace")


def main():
    from harness import runtime, sandbox, spec
    seeds = tuple(int(value) for value in os.environ.get("TASK07_PUBLIC_SEEDS", "35").split(","))
    budget = float(os.environ.get("TASK07_PUBLIC_SECONDS", "10"))
    if any(seed not in spec.DESIGN_SEEDS for seed in seeds) or not 0 < budget <= 120:
        raise ValueError("Probe accepts published seeds and a bounded public episode duration")
    rows = []
    for seed in seeds:
        started = time.monotonic()
        try:
            with sandbox.PolicyProcess("/submission") as policy:
                policy.check_loaded()
                log = runtime.run_episode(policy, seed, budget_t=budget, render=True,
                    policy_seed=0, wall_budget_s=spec.EPISODE_WALL_BUDGET_S)
                row = log.metrics(budget)
                if policy.dead or log.policy_faults:
                    row["policy_stderr"] = policy.stderr_tail()[-3000:]
            rows.append({"public_seed": seed, "episode_seconds": budget,
                "evaluation_seconds": time.monotonic() - started, **row})
        except Exception as error:
            rows.append({"public_seed": seed, "episode_seconds": budget,
                "evaluation_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}"})
    Path("/logs/probe.json").write_text(json.dumps({"kind": "short_public_episode_probe_not_official_score",
        "episode_seconds": budget, "public_seeds": seeds, "episodes": rows}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
