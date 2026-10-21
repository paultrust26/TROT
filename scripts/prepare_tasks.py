"""Build the evaluation task file from the public SWE-smith dataset.

Reads the 89 held out task IDs in ``data/task_ids.json`` and looks each one up
in the ``SWE-bench/SWE-smith`` dataset on the Hugging Face Hub, writing one task
per line with the fields ``run_trot.py`` expects.

Usage:
    python scripts/prepare_tasks.py --out data/tasks.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def as_list(value) -> list[str]:
    """Return a test list from SWE-smith, whose test fields are lists or JSON strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip().startswith("["):
        return json.loads(value)
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ids", type=Path, default=ROOT / "data" / "task_ids.json")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "tasks.jsonl")
    args = parser.parse_args()

    wanted = set(json.loads(args.ids.read_text()))
    found: dict[str, dict] = {}
    for row in load_dataset("SWE-bench/SWE-smith", split="train", streaming=True):
        iid = row.get("instance_id", "")
        if iid in wanted:
            found[iid] = {
                "instance_id": iid,
                "problem_statement": row.get("problem_statement", ""),
                "mutation_patch": row.get("patch", ""),
                "fail_to_pass": as_list(row.get("FAIL_TO_PASS")),
                "pass_to_pass": as_list(row.get("PASS_TO_PASS")),
            }
            if len(found) == len(wanted):
                break

    missing = sorted(wanted - found.keys())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for iid in sorted(found):
            f.write(json.dumps(found[iid]) + "\n")
    print(f"wrote {len(found)} tasks to {args.out}")
    if missing:
        print(f"warning: {len(missing)} task IDs were not found: {missing[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
