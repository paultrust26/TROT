"""Run TROT and its baselines on SWE-smith tasks.

Every method uses the same agent loop, prompt, tools and test harness. Each
task's repository is cloned from GitHub at the task's commit, the bug is
injected with the task's mutation patch, and the agent must fix it. A task is
resolved when its FAIL_TO_PASS tests pass.

Usage:
    export OPENAI_API_KEY=...
    export OPENAI_BASE_URL=...     # optional, for a provider with an OpenAI compatible API
    python scripts/prepare_tasks.py --out data/tasks.jsonl
    python run_trot.py --config configs/default.yaml

Results are appended to the output JSONL, one line per (task, method), with the
full trajectory. Runs that already appear in the output are skipped, so an
interrupted run can be resumed with the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from trot.agent import (  # noqa: E402
    AgentDietRunner,
    HistoryCompressor,
    HOPGate,
    LiveAgentRunner,
    SlidingWindowRunner,
    SpinDetector,
    TACORunner,
    TestbedManager,
    TROTv3Runner,
    TruncationRunner,
)

GATE_MODELS_DIR = HERE / "trot" / "models"

# TROT's ablation switches. Each maps to the TROTv3Runner arguments it changes.
ABLATIONS = {
    "no_gate": {"gate": None},
    "no_critical_bypass": {"disable_critical_bypass": True},
    "no_submit_gate": {"disable_submit_gate": True},
    "no_history_compress": {"history_compressor": None, "disable_history_compress": True},
    "no_reflection": {"max_reflection_calls": 0},
}


def trot_kwargs(params: dict) -> dict:
    """Translate the ``methods.trot`` config section into TROTv3Runner arguments.

    Args:
        params: The ``methods.trot`` section; missing keys use the paper's settings.

    Returns:
        Keyword arguments for TROTv3Runner, with any listed ablations applied last.
    """
    gate = params.get("gate") or {}
    reflection = params.get("reflection") or {}
    history = params.get("history") or {}
    nudges = params.get("nudges") or {}
    spin = params.get("spin_detector") or {}

    kwargs = {
        "gate": (
            HOPGate(models_dir=GATE_MODELS_DIR, risk_threshold=gate.get("risk_threshold", 0.15))
            if gate.get("enabled", True)
            else None
        ),
        "history_compressor": (
            HistoryCompressor(
                recency_window=history.get("recency_window", 5),
                min_tokens_to_compress=history.get("min_tokens_to_compress", 200),
                age_decay_threshold=history.get("age_decay_threshold", 15),
            )
            if history.get("enabled", True)
            else None
        ),
        "disable_history_compress": not history.get("enabled", True),
        "spin_detector": SpinDetector(
            repeat_threshold=spin.get("repeat_threshold", 3),
            max_nudges=spin.get("max_nudges", 2),
        ),
        "a": reflection.get("a", 2),
        "b": reflection.get("b", 1),
        "theta": reflection.get("theta", 3000),
        "max_reflection_calls": reflection.get("max_calls", 5),
        "nudge_start_step": nudges.get("start_step", 15),
        "nudge_interval": nudges.get("interval", 8),
        "max_nudges": nudges.get("max_nudges", 3),
        "post_edit_protection": params.get("post_edit_protection", 2),
        "tight_critical_bypass": params.get("tight_critical_bypass", False),
        "disable_submit_gate": not params.get("submit_gate", True),
    }
    for name in params.get("ablations") or []:
        if name not in ABLATIONS:
            raise ValueError(f"unknown TROT ablation '{name}', choose from {sorted(ABLATIONS)}")
        kwargs.update(ABLATIONS[name])
    return kwargs


def build_runner(method: str, cfg: dict, api_key: str, api_base: str):
    """Construct a method runner with the settings used in the paper.

    Args:
        method: One of baseline, truncation, sliding_window, taco, agent_diet, trot.
        cfg: Parsed configuration (models, step budget and per-method parameters).
        api_key: API key for the model provider.
        api_base: Base URL of the provider; empty for the client default.

    Returns:
        A runner exposing ``run_task``.
    """
    model = cfg["model"]
    reflect_model = cfg.get("reflect_model") or model
    max_steps = cfg.get("max_steps", 50)
    params = (cfg.get("methods") or {}).get(method) or {}
    common = {"api_key": api_key, "api_base": api_base, "model": model, "max_steps": max_steps}

    if method == "baseline":
        runner = LiveAgentRunner(**common, gate=None)
    elif method == "truncation":
        runner = TruncationRunner(**common, max_chars=params.get("max_chars", 3000))
    elif method == "sliding_window":
        runner = SlidingWindowRunner(**common, window_size=params.get("window_size", 8))
    elif method == "taco":
        runner = TACORunner(**common)
    elif method == "agent_diet":
        runner = AgentDietRunner(
            **common,
            reflect_model=reflect_model,
            a=params.get("a", 2),
            b=params.get("b", 1),
            theta=params.get("theta", 500),
        )
    elif method == "trot":
        runner = TROTv3Runner(**common, reflect_model=reflect_model, **trot_kwargs(params))
    else:
        raise ValueError(f"unknown method '{method}'")

    runner._max_output_tokens = cfg.get("max_output_tokens", 4096)
    return runner


def run_one(task: dict, method: str, cfg: dict, api_key: str, api_base: str, testbeds: TestbedManager) -> dict:
    """Prepare a fresh testbed for one task, run one method on it and return the result record."""
    task_id = task["instance_id"]
    base_path = testbeds.setup_testbed(task_id, mutation_patch=task.get("mutation_patch") or None)
    if not base_path:
        return {"task_id": task_id, "method": method, "error": "testbed setup failed"}
    work_path = f"{base_path}_{method}"
    if os.path.exists(work_path):
        shutil.rmtree(work_path)
    shutil.copytree(base_path, work_path)

    runner = build_runner(method, cfg, api_key, api_base)
    start = time.time()
    try:
        result = runner.run_task(
            task_id=task_id,
            problem_statement=task["problem_statement"],
            testbed_path=work_path,
            condition=method,
            fail_to_pass=task.get("fail_to_pass") or None,
            pass_to_pass=task.get("pass_to_pass") or None,
        )
    finally:
        shutil.rmtree(work_path, ignore_errors=True)

    tokens = (
        result.total_input_tokens
        + result.total_output_tokens
        + result.reflection_input_tokens
        + result.reflection_output_tokens
    )
    return {
        "task_id": task_id,
        "method": method,
        "model": cfg["model"],
        "resolved": result.resolved,
        "num_steps": result.num_steps,
        "total_tokens": tokens,
        "input_tokens": result.total_input_tokens,
        "output_tokens": result.total_output_tokens,
        "reflection_calls": result.reflection_calls,
        "duration_s": round(time.time() - start, 1),
        "error": result.error,
        "tier_distribution": result.tier_distribution,
        "trajectory": [
            {
                "step": s.step_idx,
                "tool": s.tool_name,
                "args": s.tool_args,
                "thought": s.agent_thought or "",
                "raw_observation": s.raw_observation or "",
                "served_observation": s.served_observation or "",
                "obs_tokens_raw": s.raw_obs_tokens,
                "obs_tokens_served": s.served_obs_tokens,
                "tier": s.tier_served,
            }
            for s in result.steps
        ],
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "default.yaml")
    parser.add_argument("--methods", nargs="+", help="Methods to run (overrides the config)")
    parser.add_argument("--limit", type=int, help="Run only the first N tasks")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text()) or {}
    base = args.config.resolve().parent.parent
    tasks_path = base / cfg.get("tasks", "data/tasks.jsonl")
    out_path = base / cfg.get("output", "results/results.jsonl")
    testbed_dir = base / cfg.get("testbed_dir", "testbeds")
    methods = args.methods or list((cfg.get("methods") or {}).keys())

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        parser.error("set OPENAI_API_KEY to your provider's API key")
    api_base = os.environ.get("OPENAI_BASE_URL", "")
    if not cfg.get("model"):
        parser.error("set `model` in the config")

    tasks = load_jsonl(tasks_path)
    if not tasks:
        parser.error(f"no tasks found at {tasks_path}; run scripts/prepare_tasks.py first")
    if args.limit:
        tasks = tasks[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["method"]) for r in load_jsonl(out_path) if "task_id" in r and "method" in r}
    testbeds = TestbedManager(base_dir=str(testbed_dir))

    for task in tasks:
        for method in methods:
            if (task["instance_id"], method) in done:
                continue
            record = run_one(task, method, cfg, api_key, api_base, testbeds)
            with out_path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            status = "resolved" if record.get("resolved") else "failed"
            print(f"{task['instance_id']} {method}: {status}, {record.get('total_tokens', 0):,} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
