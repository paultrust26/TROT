"""Smoke tests: every method builds from the default config and the gate loads its models.

No model API is called.
"""

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import run_trot  # noqa: E402
from trot.agent import HOPGate  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


@pytest.mark.parametrize("method", list(CONFIG["methods"]))
def test_every_method_builds(method):
    runner = run_trot.build_runner(method, CONFIG, "test-key", "")
    assert hasattr(runner, "run_task")
    assert runner._max_output_tokens == CONFIG["max_output_tokens"]


def test_trot_ablations_apply():
    cfg = {**CONFIG, "methods": {"trot": {"ablations": ["no_gate", "no_reflection"]}}}
    runner = run_trot.build_runner("trot", cfg, "test-key", "")
    assert runner._gate is None
    assert runner._max_reflection_calls == 0


def test_config_values_reach_trot():
    trot = {
        **CONFIG["methods"]["trot"],
        "reflection": {"a": 3, "b": 2, "theta": 1234, "max_calls": 7},
        "history": {"enabled": True, "recency_window": 9, "min_tokens_to_compress": 50, "age_decay_threshold": 20},
        "nudges": {"start_step": 10, "interval": 4, "max_nudges": 1},
        "gate": {"enabled": False},
        "submit_gate": False,
    }
    runner = run_trot.build_runner("trot", {**CONFIG, "methods": {"trot": trot}}, "test-key", "")
    assert (runner._a, runner._b, runner._theta, runner._max_reflection_calls) == (3, 2, 1234, 7)
    assert runner._compressor.recency_window == 9 and runner._compressor.age_decay_threshold == 20
    assert (runner._nudge_start, runner._nudge_interval) == (10, 4)
    assert runner._gate is None and runner._disable_submit_gate


def test_unknown_ablation_is_rejected():
    cfg = {**CONFIG, "methods": {"trot": {"ablations": ["not_a_switch"]}}}
    with pytest.raises(ValueError):
        run_trot.build_runner("trot", cfg, "test-key", "")


def test_gate_loads_all_models():
    gate = HOPGate(models_dir=run_trot.GATE_MODELS_DIR)
    models = gate._wrapper._interceptor.allocator.models
    assert set(models) == {"read", "bash", "test", "search"}
    assert all(m.is_loaded() for m in models.values())


def test_gate_serves_an_observation():
    gate = HOPGate(models_dir=run_trot.GATE_MODELS_DIR)
    served, tier, raw_tokens, served_tokens = gate.process(
        "bash", {"command": "ls"}, "OBSERVATION:\n" + "file.py\n" * 200, "list files", "Fix the bug"
    )
    assert isinstance(served, str) and tier
    assert served_tokens <= raw_tokens


def test_task_ids_file():
    ids = json.loads((ROOT / "data" / "task_ids.json").read_text())
    assert len(ids) == 89 and len(set(ids)) == 89
