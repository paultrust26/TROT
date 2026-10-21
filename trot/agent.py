"""Agent loop, tool execution and observation policies for TROT and its baselines.

Every method shares the same agent loop, system prompt, tools and test harness,
so the only difference between methods is how observations are served to the
model. The agent calls the model through the standard ``openai`` client; pass
``api_base`` to use any provider with an OpenAI compatible API.

Methods:
  LiveAgentRunner      Baseline: full observations, no compression
  TruncationRunner     Keeps the start and the error tail of each observation
  SlidingWindowRunner  Masks observations older than ``window_size`` steps
  TACORunner           Rule based compression with a critical observation bypass
  AgentDietRunner      Auxiliary LLM reflection on observations above a threshold
  TROTv3Runner         TROT: deterministic rules, learned gate, selective reflection
"""

import json
import os
import re
import subprocess
import time
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import OpenAI
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENC.encode(text, disallowed_special=()))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command in the terminal.\n\n"
            "* Long running commands: Avoid running long-running commands that may hang.\n"
            "* Interactive commands: Do not run interactive commands.\n"
            "* Directory: If you need to change directory, use cd in the same command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Custom editing tool for viewing, creating and editing files.\n"
            "* State is persistent across command calls and discussions.\n"
            "* `view` command with no `view_range` shows the full file.\n"
            "* `view` command with `view_range` shows the specified lines.\n"
            "* `str_replace` requires unique `old_str` in the file.\n"
            "* `create` creates a new file (fails if it already exists).\n"
            "* `insert` inserts text after the specified line number.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "str_replace", "create", "insert"],
                        "description": "The command to execute.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Absolute path to file or directory.",
                    },
                    "file_text": {
                        "type": "string",
                        "description": "Required for `create`: content of the new file.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Required for `str_replace`: exact string to replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Optional for `str_replace`: replacement string.",
                    },
                    "insert_line": {
                        "type": "integer",
                        "description": "Required for `insert`: line number after which to insert.",
                    },
                    "new_str_insert": {
                        "type": "string",
                        "description": "Required for `insert`: text to insert.",
                    },
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional for `view`: [start_line, end_line].",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": "Submit your solution when you believe the issue is resolved.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]


SYSTEM_PROMPT = "You are a helpful assistant that can interact with a computer to solve tasks."


INSTANCE_TEMPLATE = """<uploaded_files>
{working_dir}
</uploaded_files>
I've uploaded a python code repository in the directory {working_dir}. Consider the following PR description:

<pr_description>
{problem_statement}
</pr_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <pr_description> are met?
I've already taken care of all changes to any of the test files described in the <pr_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Your task is to make the minimal changes to non-tests files in the {working_dir} directory to ensure the <pr_description> is satisfied.
Follow these steps to resolve the issue:
1. As a first step, it might be a good idea to find and read code relevant to the <pr_description>
2. Create a script to reproduce the error and execute it with `python <filename.py>` using the bash tool, to confirm the error
3. Edit the source code of the repo to resolve the issue
4. Rerun your reproduce script and confirm that the error is fixed!
5. Think about edgecases and make sure your fix handles them as well
Your thinking should be thorough and so it's fine if it's very long."""


@dataclass
class AgentStep:
    step_idx: int
    tool_name: str
    tool_args: dict
    agent_thought: str
    raw_observation: str
    served_observation: str
    raw_obs_tokens: int
    served_obs_tokens: int
    tier_served: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cumulative_input_tokens: int = 0
    cumulative_output_tokens: int = 0


@dataclass
class EvalResult:
    task_id: str
    condition: str
    submitted: bool
    num_steps: int
    total_input_tokens: int
    total_output_tokens: int
    total_observation_tokens_raw: int
    total_observation_tokens_served: int
    observation_savings_pct: float
    patch_produced: str
    resolved: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    test_exit_code: int = -1
    f2p_passed: int = 0
    f2p_total: int = 0
    p2p_passed: int = 0
    p2p_total: int = 0
    steps: list[AgentStep] = field(default_factory=list)
    tier_distribution: dict = field(default_factory=dict)
    error: str = ""
    duration_seconds: float = 0.0
    # AgentDiet overhead tracking
    reflection_input_tokens: int = 0
    reflection_output_tokens: int = 0
    reflection_calls: int = 0
    # History compression tracking
    history_tokens_saved: int = 0
    history_compressions_applied: int = 0
    # Spin detection tracking
    spin_events_detected: int = 0
    spin_nudges_given: int = 0
    spin_force_stopped: bool = False
    # Skill index build cost (separate from reflection)
    skill_build_input_tokens: int = 0
    skill_build_output_tokens: int = 0
    skill_build_calls: int = 0
    # Trajectory monitor
    monitor_exit_triggered: bool = False
    monitor_exit_step: int = -1
    monitor_exit_confidence: float = 0.0
    monitor_reflects_given: int = 0


class ToolExecutor:
    """Executes SWE-smith tools for real against a testbed directory."""

    def __init__(self, testbed_dir: str):
        self._testbed = testbed_dir
        self._timeout = 60

    def execute(self, tool_name: str, args: dict) -> str:
        """Execute a tool call and return the observation."""
        if tool_name == "bash":
            return self._exec_bash(args.get("command", ""))
        elif tool_name == "str_replace_editor":
            return self._exec_editor(args)
        elif tool_name == "submit":
            return self._exec_submit()
        return f"Unknown tool: {tool_name}"

    def _exec_bash(self, command: str) -> str:
        """Execute a bash command in the testbed."""
        if not command.strip():
            return "OBSERVATION:\nNo command provided."

        # Replace /testbed references with actual testbed path
        command = command.replace("/testbed", self._testbed)

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._testbed,
                env={**os.environ},
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr
            if not output.strip():
                return "Your command ran successfully and did not produce any output."
            return f"OBSERVATION:\n{output}"
        except subprocess.TimeoutExpired:
            return "OBSERVATION:\nCommand timed out after 60 seconds."
        except Exception as e:
            return f"OBSERVATION:\nError executing command: {e}"

    def _exec_editor(self, args: dict) -> str:
        """Execute str_replace_editor commands."""
        command = args.get("command", "")
        path = args.get("path", "")

        if not path:
            return "OBSERVATION:\nError: path is required."

        # Resolve path (handle both absolute /testbed paths and relative)
        if path.startswith("/testbed"):
            full_path = self._testbed + path[len("/testbed") :]
        elif not path.startswith("/"):
            full_path = os.path.join(self._testbed, path)
        else:
            full_path = path

        # Block edits to test files
        if command in ("str_replace", "create", "insert"):
            rel_path = full_path.replace(self._testbed, "").lstrip("/")
            if self._is_test_file(rel_path):
                return (
                    "OBSERVATION:\nError: You cannot modify test files. "
                    "The tests are already correct — your task is to fix the source code "
                    "so that the existing tests pass. Edit the source files instead."
                )

        if command == "view":
            return self._view_file(full_path, args.get("view_range"))
        elif command == "create":
            return self._create_file(full_path, args.get("file_text", ""))
        elif command == "str_replace":
            return self._str_replace(full_path, args.get("old_str", ""), args.get("new_str", ""))
        elif command == "insert":
            return self._insert_line(
                full_path, args.get("insert_line", 0), args.get("new_str_insert", args.get("new_str", ""))
            )
        return f"OBSERVATION:\nUnknown editor command: {command}"

    @staticmethod
    def _is_test_file(rel_path: str) -> bool:
        """Check if a path is a test file that should not be modified."""
        parts = rel_path.split("/")
        basename = parts[-1] if parts else ""
        if basename.startswith("test_") or basename.startswith("tests_"):
            return True
        if basename.endswith("_test.py") or basename.endswith("_tests.py"):
            return True
        if "tests/" in rel_path or "test/" in rel_path:
            if rel_path.endswith(".py"):
                return True
        if basename == "conftest.py":
            return True
        return False

    def _view_file(self, path: str, view_range: Optional[list] = None) -> str:
        """View a file or directory."""
        if os.path.isdir(path):
            try:
                entries = sorted(os.listdir(path))
                listing = "\n".join(entries)
                return f"OBSERVATION:\nHere's the result of running `ls` on {path}:\n{listing}"
            except Exception as e:
                return f"OBSERVATION:\nError listing directory: {e}"

        if not os.path.exists(path):
            return f"OBSERVATION:\nError: file {path} does not exist."

        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            return f"OBSERVATION:\nError reading file: {e}"

        total_lines = len(lines)

        if view_range:
            start = max(1, int(view_range[0]))
            end = min(total_lines, int(view_range[1])) if len(view_range) > 1 else total_lines
            selected = lines[start - 1 : end]
            numbered = "".join(f"{start + i:>6}\t{line}" for i, line in enumerate(selected))
            return f"OBSERVATION:\nHere's the result of running `cat -n` on {path}:\n{numbered}"
        else:
            if total_lines > 300:
                return (
                    f"OBSERVATION:\n<NOTE>This file is too large to display entirely. "
                    f"Showing abbreviated version with {total_lines} lines total.</NOTE>\n"
                    + "".join(f"{i + 1:>6}\t{line}" for i, line in enumerate(lines[:100]))
                    + f"\n... ({total_lines - 200} lines omitted) ...\n"
                    + "".join(f"{total_lines - 99 + i:>6}\t{line}" for i, line in enumerate(lines[-100:]))
                )
            numbered = "".join(f"{i + 1:>6}\t{line}" for i, line in enumerate(lines))
            return f"OBSERVATION:\nHere's the result of running `cat -n` on {path}:\n{numbered}"

    def _create_file(self, path: str, content: str) -> str:
        """Create a new file."""
        if os.path.exists(path):
            return f"OBSERVATION:\nError: file {path} already exists. Use str_replace to edit it."
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            return f"OBSERVATION:\nFile created successfully at: {path}"
        except Exception as e:
            return f"OBSERVATION:\nError creating file: {e}"

    def _str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace a string in a file."""
        if not os.path.exists(path):
            return f"OBSERVATION:\nError: file {path} does not exist."

        try:
            with open(path, "r") as f:
                content = f.read()
        except Exception as e:
            return f"OBSERVATION:\nError reading file: {e}"

        count = content.count(old_str)
        if count == 0:
            return (
                f"OBSERVATION:\nError: `old_str` not found in {path}. "
                "Make sure the string matches exactly (including whitespace and indentation)."
            )
        if count > 1:
            return (
                f"OBSERVATION:\nError: `old_str` appears {count} times in {path}. "
                "Please provide a more unique string to replace."
            )

        new_content = content.replace(old_str, new_str, 1)
        with open(path, "w") as f:
            f.write(new_content)

        # Show snippet around the edit
        new_lines = new_content.splitlines()
        # Find where the edit was made
        edit_start = content[: content.find(old_str)].count("\n")
        snippet_start = max(0, edit_start - 2)
        snippet_end = min(len(new_lines), edit_start + new_str.count("\n") + 5)
        snippet = "\n".join(
            f"{snippet_start + i + 1:>6}\t{line}" for i, line in enumerate(new_lines[snippet_start:snippet_end])
        )

        return (
            f"OBSERVATION:\nThe file {path} has been edited. "
            f"Here's the result of running `cat -n` on a snippet of {path}:\n{snippet}"
        )

    def _insert_line(self, path: str, line_num: int, text: str) -> str:
        """Insert text after a line number."""
        if not os.path.exists(path):
            return f"OBSERVATION:\nError: file {path} does not exist."

        try:
            with open(path, "r") as f:
                lines = f.readlines()
        except Exception as e:
            return f"OBSERVATION:\nError reading file: {e}"

        insert_lines = text.splitlines(keepends=True)
        if insert_lines and not insert_lines[-1].endswith("\n"):
            insert_lines[-1] += "\n"

        lines[line_num:line_num] = insert_lines
        with open(path, "w") as f:
            f.writelines(lines)

        return (
            f"OBSERVATION:\nThe file {path} has been edited. Inserted {len(insert_lines)} lines after line {line_num}."
        )

    def _exec_submit(self) -> str:
        """Handle submit — get the diff."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                cwd=self._testbed,
                timeout=10,
            )
            diff = result.stdout.strip()
            if diff:
                return f"OBSERVATION:\nChanges submitted. Diff:\n{diff}"
            return "OBSERVATION:\nNo changes detected."
        except Exception as e:
            return f"OBSERVATION:\nError getting diff: {e}"

    def get_patch(self) -> str:
        """Get the git diff (the patch the agent produced)."""
        try:
            result = subprocess.run(
                ["git", "diff"],
                capture_output=True,
                text=True,
                cwd=self._testbed,
                timeout=10,
            )
            return result.stdout.strip()
        except Exception:
            return ""

    def run_tests(
        self,
        fail_to_pass: Optional[list[str]] = None,
        pass_to_pass: Optional[list[str]] = None,
        timeout: int = 180,
    ) -> dict:
        """Run targeted tests and verify resolution using SWE-bench semantics.

        If fail_to_pass/pass_to_pass are provided, runs ONLY those specific tests
        and checks resolution: all F2P must pass AND all P2P must still pass.

        Returns:
            dict with keys: resolved (bool), exit_code (int),
            f2p_passed (int), f2p_total (int), p2p_passed (int), p2p_total (int),
            num_passed (int), num_failed (int), output (str truncated)
        """
        if fail_to_pass:
            return self._run_targeted_tests(fail_to_pass, pass_to_pass or [], timeout)

        # Fallback: run full pytest suite
        cmd = "python -m pytest --tb=short -q 2>&1"
        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._testbed,
                env={**os.environ},
            )
            output = result.stdout + result.stderr
            num_passed, num_failed = self._parse_pytest_summary(output)
            return {
                "resolved": result.returncode == 0,
                "exit_code": result.returncode,
                "f2p_passed": 0,
                "f2p_total": 0,
                "p2p_passed": 0,
                "p2p_total": 0,
                "num_passed": num_passed,
                "num_failed": num_failed,
                "output": output[-2000:],
                "command": cmd,
            }
        except subprocess.TimeoutExpired:
            return {
                "resolved": False,
                "exit_code": -1,
                "f2p_passed": 0,
                "f2p_total": 0,
                "p2p_passed": 0,
                "p2p_total": 0,
                "num_passed": 0,
                "num_failed": 0,
                "output": "Test execution timed out",
                "command": cmd,
            }
        except Exception:
            return {
                "resolved": False,
                "exit_code": -1,
                "f2p_passed": 0,
                "f2p_total": 0,
                "p2p_passed": 0,
                "p2p_total": 0,
                "num_passed": 0,
                "num_failed": 0,
                "output": "No test runner found",
                "command": "",
            }

    def _run_targeted_tests(
        self,
        fail_to_pass: list[str],
        pass_to_pass: list[str],
        timeout: int,
    ) -> dict:
        """Run specific F2P and P2P tests and determine resolution."""
        f2p_passed = 0
        p2p_passed = 0
        all_output = []

        # Run F2P tests (these should PASS after fix = resolution)
        if fail_to_pass:
            f2p_result = self._run_test_list(fail_to_pass, timeout)
            f2p_passed = f2p_result["num_passed"]
            all_output.append(f"=== FAIL_TO_PASS ({f2p_passed}/{len(fail_to_pass)}) ===\n{f2p_result['output']}")

        # Run P2P tests (these should still PASS = maintenance)
        if pass_to_pass:
            # Only run a sample of P2P if there are many (avoid long timeouts)
            p2p_sample = pass_to_pass[:50]
            p2p_result = self._run_test_list(p2p_sample, timeout)
            p2p_passed = p2p_result["num_passed"]
            all_output.append(f"=== PASS_TO_PASS ({p2p_passed}/{len(p2p_sample)}) ===\n{p2p_result['output']}")

        # Resolution: ALL F2P pass AND ALL P2P pass
        f2p_all_pass = f2p_passed == len(fail_to_pass)
        p2p_all_pass = (p2p_passed == len(pass_to_pass[:50])) if pass_to_pass else True
        resolved = f2p_all_pass and p2p_all_pass

        combined_output = "\n".join(all_output)
        return {
            "resolved": resolved,
            "exit_code": 0 if resolved else 1,
            "f2p_passed": f2p_passed,
            "f2p_total": len(fail_to_pass),
            "p2p_passed": p2p_passed,
            "p2p_total": len(pass_to_pass[:50]) if pass_to_pass else 0,
            "num_passed": f2p_passed + p2p_passed,
            "num_failed": (len(fail_to_pass) - f2p_passed)
            + (len(pass_to_pass[:50]) - p2p_passed if pass_to_pass else 0),
            "output": combined_output[-3000:],
            "command": "targeted pytest",
        }

    def _run_test_list(self, tests: list[str], timeout: int) -> dict:
        """Run a specific list of pytest test IDs."""
        # Join test IDs as arguments to pytest
        test_args = " ".join(f'"{t}"' for t in tests)
        cmd = f"python -m pytest --tb=short -q {test_args} 2>&1"

        try:
            result = subprocess.run(
                ["bash", "-c", cmd],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._testbed,
                env={**os.environ},
            )
            output = result.stdout + result.stderr
            num_passed, num_failed = self._parse_pytest_summary(output)
            return {"num_passed": num_passed, "num_failed": num_failed, "output": output[-2000:]}
        except subprocess.TimeoutExpired:
            return {"num_passed": 0, "num_failed": len(tests), "output": "timed out"}
        except Exception as e:
            return {"num_passed": 0, "num_failed": len(tests), "output": str(e)}

    @staticmethod
    def _parse_pytest_summary(output: str) -> tuple[int, int]:
        """Parse pytest summary line for pass/fail counts."""
        num_passed = 0
        num_failed = 0
        for line in output.splitlines():
            m = re.search(r"(\d+) passed", line)
            if m:
                num_passed = int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m:
                num_failed = int(m.group(1))
            m = re.search(r"(\d+) error", line)
            if m:
                num_failed += int(m.group(1))
        return num_passed, num_failed


class ObservationCompressor:
    """Compresses observations at delivery time (before they enter history).

    Applied once when the tool result is first served to the agent. Targets
    patterns that waste tokens on first delivery — not staleness (that's
    HistoryCompressor's job).

    From analysis of 100 SWE-smith trajectories, 49% of observation tokens
    are deterministically identifiable waste:

      1. Edit confirmations re-show file (22%): after str_replace, the tool
         re-displays surrounding context. Agent just wrote it — strip to stub.
      2. Submit harness boilerplate (30%): identical multi-paragraph "Thank you
         for your work" message. Replace with 1-line confirmation.
      3. Passing test verbosity (2%): full pytest output when all passed.
         Reduce to "X passed in Ys".
      4. pip install noise: download progress, version resolution.
         Keep only final "Successfully installed" line.
      5. Pytest collection/header noise: platform info, rootdir, plugins, etc.
         Strip to just the results section.
    """

    def __init__(self, enabled: bool = True, problem_statement: str = "", tier_selector_path: Optional[str] = None):
        self._enabled = enabled
        self._tokens_saved = 0
        self._compressions = 0
        self._problem_keywords: set[str] = set()
        if problem_statement:
            self.set_problem_statement(problem_statement)

        # Tier selector model: learned decision on when to compress file views
        # Replaces fixed rule (>5 defs + >1500 tokens) with P(safe to compress)
        self._tier_model = None
        if tier_selector_path:
            try:
                import pickle
                with open(tier_selector_path, "rb") as f:
                    checkpoint = pickle.load(f)
                self._tier_model = checkpoint["model"]
            except Exception:
                pass

        # State tracking for tier selector features
        self._files_read: set[str] = set()
        self._edits_count: int = 0
        self._reads_count: int = 0
        self._last_edit_step: int = -1
        self._current_step: int = 0

    def set_problem_statement(self, problem: str):
        """Extract keywords from problem statement to protect relevant observations."""
        import re as _re
        identifiers = set(_re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b", problem))
        self._problem_keywords = {w for w in identifiers if not w.isupper() or len(w) > 4}

    def _contains_target_code(self, obs: str) -> bool:
        """Check if observation contains code relevant to the problem statement."""
        if not self._problem_keywords:
            return False
        obs_sample = obs[:3000]
        matches = sum(1 for kw in self._problem_keywords if kw in obs_sample)
        return matches >= 3

    def track_tool_call(self, tool_name: str, tool_args: dict):
        """Track tool calls for tier selector state (call BEFORE compress)."""
        self._current_step += 1
        if tool_name == "str_replace_editor":
            cmd = tool_args.get("command", "")
            if cmd == "view":
                self._reads_count += 1
            elif cmd in ("str_replace", "create", "insert"):
                self._edits_count += 1
                self._last_edit_step = self._current_step
                path = tool_args.get("path", "")
                if path:
                    self._files_read.add(path)
        elif tool_name == "Read":
            self._reads_count += 1

    def _predict_safe_to_compress(self, tool_args: dict, observation: str, thought: str = "") -> bool:
        """Use tier selector model to predict if this read is safe to compress.

        Returns True if the model predicts compression is safe (P(safe) >= 0.8).
        Falls back to False (don't compress) if no model loaded.
        """
        if self._tier_model is None:
            return False

        try:
            import numpy as _np
            import re as _re

            file_path = tool_args.get("path", tool_args.get("file_path", ""))
            obs_tokens = len(observation) // 4
            line_count = observation.count("\n") + 1
            thought_lower = thought.lower() if thought else ""

            features = _np.array([[
                obs_tokens / 1000.0,
                float(obs_tokens > 1500),
                float("Error" in observation[:500] or "Traceback" in observation[:500]),
                float(bool(_re.search(r"(?:def |class |async def )", observation[:2000]))),
                line_count / 100.0,
                float(any(w in thought_lower for w in ("fix", "edit", "modify", "change", "update"))),
                float(any(w in thought_lower for w in ("check", "look", "explore", "see", "understand"))),
                float(len(thought)) / 300.0,
                float(file_path in self._files_read),
                float("test" in (file_path or "").lower()),
                float(any(kw in (file_path or "").lower() for kw in ("config", "settings", ".yaml", ".json", ".toml"))),
                self._current_step / 50.0,
                float(self._edits_count) / 5.0,
                float(self._reads_count) / 10.0,
                (self._current_step - self._last_edit_step) / 10.0 if self._last_edit_step >= 0 else self._current_step / 10.0,
            ]], dtype=_np.float32)

            p_safe = float(self._tier_model.predict_proba(features)[0][1])
            return p_safe >= 0.95  # raised from 0.8 — only compress when very confident
        except Exception:
            return False

    def compress(self, tool_name: str, tool_args: dict, observation: str, thought: str = "") -> str:
        if not self._enabled:
            return observation

        original_tokens = len(observation) // 4

        # Never compress observations containing target code (LLM-discovered rule #3)
        if original_tokens > 500 and self._contains_target_code(observation):
            return observation

        result = observation

        if tool_name == "str_replace_editor":
            result = self._compress_editor(tool_args, observation)
        elif tool_name == "bash":
            result = self._compress_bash(tool_args, observation)

        saved = original_tokens - len(result) // 4
        if saved > 0:
            self._tokens_saved += saved
            self._compressions += 1
        return result

    def _compress_editor(self, args: dict, obs: str) -> str:
        command = args.get("command", "")

        # Edit confirmations: the agent often relies on the re-shown code, so keep it.
        # Savings are only ~256 tok/fire — not worth the resolve rate risk.
        # Keep full observation so agent can verify its edit landed correctly.

        if command == "view":
            # Targeted views (view_range specified) — agent asked for specific lines, never compress
            if args.get("view_range"):
                return obs

            # Directory listings: strip noise entries (__pycache__, .egg-info, .tox, etc.)
            first_lines = obs[:50].split('\n')
            is_dir_listing = "Here's the result of running `ls`" in obs or (
                len(first_lines) > 1 and not any(c.isdigit() for c in first_lines[1]) and obs.count('\n') < 50
            )
            if is_dir_listing:
                obs = self._strip_dir_noise(obs)

            obs_tokens = len(obs) // 4

            # Tier selector model: learned decision on when to compress
            # Uses agent intent, file history and session state
            if self._tier_model and obs_tokens > 500:
                if self._predict_safe_to_compress(args, obs):
                    return self._compress_to_outline(obs, args.get("path", ""))

            # RELAXED: Large file with many definitions — outline only on RE-READS
            # (agents use file views on first read, but rarely on re-reads)
            if obs_tokens > 1500:
                import re as _re
                def_count = len(_re.findall(r"(?:^|\n)[\s\d]*\t?\s*(?:def |class |async def )", obs))
                file_path = args.get("path", "")
                is_reread = file_path in self._files_read
                if def_count > 5 and is_reread and not self._contains_target_code(obs):
                    return self._compress_to_outline(obs, args.get("path", ""))

            # File views: only strip boilerplate on large re-reads (not first view)
            if obs_tokens > 2000:
                file_path = args.get("path", "")
                if file_path in self._files_read:
                    return self._strip_file_boilerplate(obs)

        return obs

    def _compress_bash(self, args: dict, obs: str) -> str:
        command = args.get("command", "")
        first_word = command.strip().split()[0] if command.strip() else ""

        # Submit harness boilerplate
        if "Thank you for your work on this issue" in obs:
            return "[submit acknowledged]"

        # pip install: keep only the final result
        if ("pip" in command or "pip3" in command) and ("Collecting" in obs or "Successfully" in obs):
            return self._compress_pip(obs)

        # Passing tests: compress verbose output to summary (only large suites)
        if self._is_all_passed_test(obs) and len(obs) // 4 > 1500:
            return self._compress_passing_tests(obs)

        # Verbose test output with individual PASSED lines — strip per-test lines, keep summary
        if ("PASSED" in obs or "PASS" in obs) and obs.count("PASSED") > 20:
            return self._compress_verbose_test_lines(obs)

        # Pytest with failures: strip header noise, keep failures
        if "test session starts" in obs and "FAILED" in obs and len(obs) // 4 > 1000:
            return self._compress_test_failures(obs)

        # File listing (find/ls): compress large listings to count + sample
        if first_word in ("find", "ls") and obs.count("\n") > 20:
            return self._compress_file_listing(obs, command)

        # Grep results: compress large result sets to top matches + count
        if first_word in ("grep", "rg", "ag", "ack") and obs.count("\n") > 20:
            return self._compress_grep_results(obs, command)

        # Git log: compress to summary (commit hashes + messages only)
        if first_word == "git" and ("log" in command or "blame" in command) and len(obs) // 4 > 500:
            return self._compress_git_output(obs)

        return obs

    def _strip_file_boilerplate(self, obs: str) -> str:
        """Strip license headers and copyright blocks from file views."""
        lines = obs.split("\n")
        result_lines = []
        in_header = True
        header_patterns = ["# copyright", "# license", "# permission", "# the software",
                          "# -*- coding", "# this file", "# authors", "# distributed"]

        for line in lines:
            lower = line.lower().strip()
            if in_header and any(p in lower for p in header_patterns):
                continue
            elif in_header and lower.startswith("#") and len(lower) < 3:
                continue
            else:
                in_header = False
                result_lines.append(line)

        result = "\n".join(result_lines)
        if len(result) < len(obs) - 100:
            return result
        return obs

    def _compress_pip(self, obs: str) -> str:
        """Keep only the final 'Successfully installed' line from pip output."""
        lines = obs.split("\n")
        important = []
        for line in lines:
            if "Successfully installed" in line:
                important.append(line)
            elif "ERROR" in line or "error:" in line.lower():
                important.append(line)
            elif "already satisfied" in line.lower() and not important:
                important.append(line)
                break
        if important:
            return "\n".join(important)
        return obs[:200] + "\n[pip output truncated]"

    def _is_all_passed_test(self, obs: str) -> bool:
        """Check if this is a test run where everything passed."""
        has_passed = "passed" in obs.lower()
        has_failed = "failed" in obs.lower() or "FAILED" in obs or "error" in obs.lower()
        return has_passed and not has_failed

    def _compress_passing_tests(self, obs: str) -> str:
        """Compress verbose all-passing test output to a summary line."""
        import re as _re
        # Extract summary line like "42 passed in 3.21s"
        match = _re.search(r"=+\s*([\d,]+ passed[^=]*?)=+", obs)
        if match:
            return f"All tests passed: {match.group(1).strip()}"
        match = _re.search(r"(\d+ passed)", obs)
        if match:
            return f"All tests passed: {match.group(1)}"
        return obs[:100] + "\n[all tests passed — verbose output omitted]"

    def _compress_test_failures(self, obs: str) -> str:
        """Strip pytest header/footer noise, keep failure details."""
        lines = obs.split("\n")
        result = []
        in_failures = False
        in_header = True

        for line in lines:
            # Skip header (platform, plugins, rootdir, etc.)
            if in_header:
                if line.startswith("FAILED") or "FAILURES" in line or "ERROR" in line:
                    in_header = False
                    in_failures = True
                    result.append(line)
                elif "collecting" in line.lower() or "collected" in line.lower():
                    result.append(line)
                    in_header = False
                continue

            # Keep failure sections
            if "FAILURES" in line or "ERRORS" in line or line.startswith("FAILED"):
                in_failures = True
            if "short test summary" in line.lower():
                in_failures = True

            if in_failures or line.startswith("E ") or "assert" in line.lower():
                result.append(line)
            elif line.startswith("=") and ("passed" in line or "failed" in line):
                result.append(line)

        if result:
            return "\n".join(result)
        return obs

    def _compress_file_listing(self, obs: str, command: str) -> str:
        """Compress large find/ls output to count + sample."""
        lines = [l for l in obs.split("\n") if l.strip() and not l.startswith("OBSERVATION")]
        if len(lines) <= 20:
            return obs
        sample = lines[:10]
        return (
            f"OBSERVATION:\n[{len(lines)} files/directories found]\n"
            + "\n".join(sample)
            + f"\n... ({len(lines) - 10} more)"
        )

    def _compress_grep_results(self, obs: str, command: str) -> str:
        """Compress large grep output to top matches + count."""
        lines = [l for l in obs.split("\n") if l.strip() and not l.startswith("OBSERVATION")]
        if len(lines) <= 20:
            return obs
        sample = lines[:15]
        return (
            f"OBSERVATION:\n[{len(lines)} matches found]\n"
            + "\n".join(sample)
            + f"\n... ({len(lines) - 15} more matches)"
        )

    def _compress_git_output(self, obs: str) -> str:
        """Compress git log/blame to commit summaries."""
        lines = obs.split("\n")
        kept = []
        for line in lines:
            if line.startswith("commit ") or line.startswith("Author:") or line.startswith("Date:"):
                kept.append(line)
            elif line.strip() and not line.startswith(" ") and len(line.strip()) < 200:
                kept.append(line)
            elif len(kept) > 30:
                break
        if kept and len(kept) < len(lines):
            return "\n".join(kept) + f"\n[... {len(lines) - len(kept)} lines omitted]"
        return obs

    def _compress_to_outline(self, obs: str, path: str) -> str:
        """Compress large file view to function/class signatures only.

        Handles both raw content and numbered lines (cat -n format with leading
        line numbers and tab separators).
        """
        import re as _re
        lines = obs.split("\n")
        outline = []
        file_name = path.split("/")[-1] if path else "file"

        # Pattern matches: optional line-number prefix, then def/class/import
        sig_pattern = _re.compile(r"^[\s\d]*\t?\s*(def |class |async def |import |from \S+ import )")

        for line in lines:
            if sig_pattern.match(line):
                outline.append(line)

        if outline and len(outline) < len(lines) // 2:
            total_lines = sum(1 for l in lines if l.strip())
            header = f"OBSERVATION:\n[{file_name}: {total_lines} lines — showing outline ({len(outline)} definitions)]\n"
            return header + "\n".join(outline)
        return obs

    def _strip_dir_noise(self, obs: str) -> str:
        """Strip noise entries from directory listings."""
        noise_patterns = ("__pycache__", ".egg-info", ".tox", ".mypy_cache",
                         "node_modules", ".pytest_cache", ".coverage", ".eggs")
        lines = obs.split("\n")
        filtered = [l for l in lines if not any(n in l for n in noise_patterns)]
        if len(filtered) < len(lines):
            return "\n".join(filtered)
        return obs

    def _compress_verbose_test_lines(self, obs: str) -> str:
        """Strip individual per-test PASSED lines, keep failures + summary."""
        lines = obs.split("\n")
        kept = []
        passed_count = 0
        for line in lines:
            if "PASSED" in line and "::" in line and "FAILED" not in line:
                passed_count += 1
            else:
                kept.append(line)
        if passed_count > 5:
            kept.insert(0, f"[{passed_count} individual PASSED lines stripped]")
            return "\n".join(kept)
        return obs

    def get_metrics(self) -> dict:
        return {
            "obs_compress_tokens_saved": self._tokens_saved,
            "obs_compress_count": self._compressions,
        }


class HistoryCompressor:
    """Compresses stale observations in conversation history before each API call.

    Attacks the 96.9% of input tokens that come from replayed history — not
    from individual observations at delivery time. Purely deterministic: no LLM
    calls, just pattern-matching on tool calls to detect superseded content.

    Rules:
      1. Stale file read: same file read again later → truncate earlier read
      2. Stale bash output: same command re-run later → truncate earlier output
      3. Failed attempt: long traceback from a step later fixed → truncate
      4. Redundant search: same grep/find re-run → truncate earlier result

    Never compressed:
      - Last `recency_window` tool results (default 5)
      - The most recent observation of any file/command
      - System and user messages (index 0, 1)
    """

    def __init__(
        self,
        recency_window: int = 5,
        min_tokens_to_compress: int = 200,
        age_decay_threshold: int = 15,
        semantic_decay_model_path: Optional[str] = None,
    ):
        self.recency_window = recency_window
        self.min_tokens_to_compress = min_tokens_to_compress
        self.age_decay_threshold = age_decay_threshold
        self._tokens_saved_total = 0
        self._compressions_applied = 0

        # Semantic decay model: replaces fixed age threshold with P(still_relevant)
        self._semantic_model = None
        self._semantic_tfidf = None
        self._semantic_svd = None
        if semantic_decay_model_path:
            try:
                import pickle
                with open(semantic_decay_model_path, "rb") as f:
                    checkpoint = pickle.load(f)
                self._semantic_model = checkpoint["model"]
                self._semantic_tfidf = checkpoint.get("tfidf")
                self._semantic_svd = checkpoint.get("svd")
            except Exception:
                pass

    def compress(self, messages: list[dict], current_step: int) -> list[dict]:
        """Compress stale tool results in-place. Returns the same list (mutated).

        Called before each API call. Scans tool messages and replaces stale
        content with short stubs.
        """
        tool_msgs = self._index_tool_messages(messages)
        if not tool_msgs:
            return messages

        # Determine which tool messages are in recency window (protected)
        protected_indices = set()
        tool_msg_indices = [idx for idx, _, _, _ in tool_msgs]
        for idx in tool_msg_indices[-self.recency_window :]:
            protected_indices.add(idx)

        # Track latest occurrence of each file/command
        file_reads: dict[str, int] = {}  # path → latest msg_index
        bash_cmds: dict[str, int] = {}  # normalized_cmd → latest msg_index
        file_edits: set[str] = set()  # paths that were edited

        # Forward pass: find latest occurrence of each resource
        for msg_idx, tool_name, tool_args, _ in tool_msgs:
            if tool_name == "str_replace_editor":
                path = tool_args.get("path", "")
                cmd = tool_args.get("command", "")
                if cmd == "view":
                    file_reads[path] = msg_idx
                elif cmd in ("str_replace", "create", "insert"):
                    file_edits.add(path)
                    # An edit supersedes all prior reads of this file
                    file_reads[path] = msg_idx
            elif tool_name == "bash":
                command = tool_args.get("command", "")
                norm_cmd = self._normalize_bash_cmd(command)
                bash_cmds[norm_cmd] = msg_idx

        # Backward pass: mark stale entries for compression
        for msg_idx, tool_name, tool_args, content in tool_msgs:
            if msg_idx in protected_indices:
                continue

            if not content or count_tokens(content) < self.min_tokens_to_compress:
                continue

            stub = self._check_stale(
                msg_idx, tool_name, tool_args, content, file_reads, bash_cmds, file_edits, current_step, tool_msgs
            )
            if stub:
                original_tokens = count_tokens(content)
                messages[msg_idx]["content"] = stub
                saved = original_tokens - count_tokens(stub)
                self._tokens_saved_total += saved
                self._compressions_applied += 1

        return messages

    def _check_stale(
        self,
        msg_idx: int,
        tool_name: str,
        tool_args: dict,
        content: str,
        file_reads: dict[str, int],
        bash_cmds: dict[str, int],
        file_edits: set[str],
        current_step: int,
        tool_msgs: list,
    ) -> Optional[str]:
        """Check if a tool message is stale. Returns stub string or None."""

        if tool_name == "str_replace_editor":
            path = tool_args.get("path", "")
            cmd = tool_args.get("command", "")

            if cmd == "view" and path:
                latest_idx = file_reads.get(path, msg_idx)
                if latest_idx > msg_idx:
                    if path in file_edits:
                        return f"[read {path} — file was later edited and re-read]"
                    else:
                        return f"[read {path} — superseded by later read]"

            # Stale edit confirmations: the agent already moved past this edit
            if cmd in ("str_replace", "insert", "create") and path:
                if "has been edited" in content or "File created" in content:
                    short_path = path.split("/")[-1]
                    return f"[edited {short_path} — confirmed]"

        elif tool_name == "bash":
            command = tool_args.get("command", "")
            norm_cmd = self._normalize_bash_cmd(command)

            latest_idx = bash_cmds.get(norm_cmd, msg_idx)
            if latest_idx > msg_idx:
                exit_hint = ""
                if "Error" in content or "error" in content or "Traceback" in content:
                    exit_hint = ", had errors"
                elif "OBSERVATION:\nYour command ran successfully" in content:
                    exit_hint = ", succeeded"
                cmd_short = command[:80] + ("..." if len(command) > 80 else "")
                return f"[ran `{cmd_short}` — re-run later{exit_hint}]"

            # Long failed output that was later retried successfully
            if self._is_failed_output(content) and self._was_later_fixed(norm_cmd, msg_idx, bash_cmds, tool_msgs):
                cmd_short = command[:80] + ("..." if len(command) > 80 else "")
                lines = content.count("\n")
                return f"[ran `{cmd_short}` — failed ({lines} lines), later fixed]"

            # Stale passing test output: all tests passed, agent moved on
            if self._is_passing_test(content) and count_tokens(content) > 300:
                cmd_short = command[:60] + ("..." if len(command) > 60 else "")
                return f"[ran `{cmd_short}` — all tests passed]"

        # Age-based decay: unique observations that are very old get compressed
        # These were never re-run/re-read, but they replay in every API call
        if current_step > 0 and count_tokens(content) > 300:
            # Estimate step age from position in tool_msgs list
            msg_position = next((i for i, (idx, _, _, _) in enumerate(tool_msgs) if idx == msg_idx), 0)
            total_tool_msgs = len(tool_msgs)
            age = total_tool_msgs - msg_position

            # Semantic decay: use model if available, else fall back to age threshold
            if self._semantic_model is not None and age >= 5:
                if not self._is_semantically_relevant(content, tool_name, tool_args, age, file_edits, messages):
                    return self._age_decay(tool_name, tool_args, content, file_edits)
            elif age >= self.age_decay_threshold:
                return self._age_decay(tool_name, tool_args, content, file_edits)

        return None

    def _is_semantically_relevant(
        self, content: str, tool_name: str, tool_args: dict,
        age: int, file_edits: set[str], messages: list[dict]
    ) -> bool:
        """Use semantic decay model to check if observation is still relevant.

        Returns True if observation should be KEPT (still relevant).
        Returns False if it's safe to decay.
        """
        try:
            import numpy as np
            import re as _re

            obs_path = tool_args.get("path", "")

            # Get recent thoughts from messages (last 3 assistant messages)
            recent_thoughts = ""
            assistant_count = 0
            for msg in reversed(messages):
                if msg.get("role") == "assistant" and msg.get("content"):
                    recent_thoughts = msg["content"][:500] + " " + recent_thoughts
                    assistant_count += 1
                    if assistant_count >= 3:
                        break

            # Structural features
            is_file_view = float(tool_name == "str_replace_editor" and tool_args.get("command") == "view")
            file_edited_after = float(obs_path in file_edits) if obs_path else 0.0
            has_error = float("Error" in content[:500] or "Traceback" in content[:500])
            obs_tokens = len(content) // 4

            # NLP: keyword overlap
            obs_ids = set(_re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", content[:2000]))
            thought_ids = set(_re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", recent_thoughts))
            keyword_overlap = len(obs_ids & thought_ids) / max(len(obs_ids), 1)

            # TF-IDF similarity (if available)
            tfidf_sim = 0.0
            obs_emb = np.zeros(16, dtype=np.float32)
            if self._semantic_tfidf and self._semantic_svd:
                obs_emb = self._semantic_svd.transform(
                    self._semantic_tfidf.transform([content[:2000]])
                )[0].astype(np.float32)
                if recent_thoughts:
                    thought_emb = self._semantic_svd.transform(
                        self._semantic_tfidf.transform([recent_thoughts[:2000]])
                    )[0]
                    norm_o = np.linalg.norm(obs_emb)
                    norm_t = np.linalg.norm(thought_emb)
                    if norm_o > 1e-8 and norm_t > 1e-8:
                        tfidf_sim = float(np.dot(obs_emb, thought_emb) / (norm_o * norm_t))

            # Build feature vector: structural (6) + keyword (1) + tfidf_sim (1) + embedding (16) = 24
            # Must match train_semantic_decay.py feature order
            features = np.concatenate([
                [float(age), is_file_view, file_edited_after, has_error,
                 obs_tokens / 1000.0, keyword_overlap, tfidf_sim],
                obs_emb,
            ]).reshape(1, -1).astype(np.float32)

            p_relevant = float(self._semantic_model.predict_proba(features)[0][1])
            return p_relevant >= 0.3  # Keep if P(relevant) >= 0.3

        except Exception:
            # Fallback: use age threshold
            return age < self.age_decay_threshold

    def _age_decay(self, tool_name: str, tool_args: dict, content: str, file_edits: set[str]) -> Optional[str]:
        """Compress old unique observations to short stubs.

        Targets:
          - File views of files never edited → outline stub
          - Old grep/search results → filenames-only stub
          - Intermediate test output (not latest) → pass/fail stub
          - Python script output → last few lines only
        """
        if tool_name == "str_replace_editor":
            path = tool_args.get("path", "")
            cmd = tool_args.get("command", "")
            if cmd == "view" and path and path not in file_edits:
                # File was explored but never edited — agent extracted what it needed
                lines = content.split("\n")
                line_count = len(lines)
                short_path = path.split("/")[-1]
                # Keep just function signatures as a reminder
                import re as _re
                sigs = [l.strip() for l in lines if _re.match(r"\s*\d*\t?\s*(def |class )", l)]
                if sigs:
                    sig_str = "; ".join(sigs[:5])
                    return f"[read {short_path} ({line_count} lines, not edited) — contains: {sig_str}]"
                return f"[read {short_path} ({line_count} lines, not edited) — context available via re-read]"

        elif tool_name == "bash":
            command = tool_args.get("command", "")
            first_word = command.strip().split()[0] if command.strip() else ""

            # Old grep/search → filenames only
            if first_word in ("grep", "find", "rg", "ag"):
                lines = [l for l in content.split("\n") if l.strip()]
                # Extract just file paths from grep output
                import re as _re
                files = set()
                for l in lines[:50]:
                    m = _re.match(r"([^\s:]+\.\w+)", l)
                    if m:
                        files.add(m.group(1).split("/")[-1])
                if files:
                    return f"[searched: found matches in {', '.join(sorted(files)[:8])}]"
                return f"[searched: {len(lines)} results (details decayed)]"

            # Old test output → pass/fail summary
            if any(kw in command for kw in ("pytest", "test", "unittest")):
                has_fail = "FAILED" in content or "Error" in content
                has_pass = "passed" in content.lower()
                status = "FAILED" if has_fail else ("passed" if has_pass else "ran")
                cmd_short = command[:60]
                return f"[test `{cmd_short}` — {status} (details decayed, re-run for current state)]"

            # Old python script output → keep last few lines
            if first_word in ("python", "python3", "python3.10"):
                lines = content.split("\n")
                if len(lines) > 8:
                    last_lines = "\n".join(lines[-5:])
                    return f"[python output ({len(lines)} lines, decayed) — last lines:\n{last_lines}]"

        return None

    def _is_passing_test(self, content: str) -> bool:
        """Check if output is a passing test run with no failures."""
        has_passed = "passed" in content.lower()
        has_failed = "failed" in content.lower() or "FAILED" in content or "error" in content.lower()
        return has_passed and not has_failed

    def _is_failed_output(self, content: str) -> bool:
        """Check if output looks like a failure (traceback, errors)."""
        fail_signals = ["Traceback (most recent call last)", "FAILED", "Error:", "error:"]
        return any(s in content for s in fail_signals) and count_tokens(content) > 300

    def _was_later_fixed(
        self,
        norm_cmd: str,
        msg_idx: int,
        bash_cmds: dict[str, int],
        tool_msgs: list,
    ) -> bool:
        """Check if a similar command succeeded later."""
        # Look for the same command (or a test command) run later that succeeded
        for later_idx, tool_name, tool_args, content in tool_msgs:
            if later_idx <= msg_idx:
                continue
            if tool_name != "bash":
                continue
            later_norm = self._normalize_bash_cmd(tool_args.get("command", ""))
            if later_norm == norm_cmd and content:
                if "OBSERVATION:\nYour command ran successfully" in content:
                    return True
                if "passed" in content.lower() and "failed" not in content.lower():
                    return True
        return False

    def _normalize_bash_cmd(self, cmd: str) -> str:
        """Normalize a bash command for dedup comparison."""
        cmd = cmd.strip()
        # Remove common prefixes that don't change semantics
        for prefix in ["cd /testbed && ", "cd /testbed; "]:
            if cmd.startswith(prefix):
                cmd = cmd[len(prefix) :]
        return cmd

    def _index_tool_messages(self, messages: list[dict]) -> list[tuple[int, str, dict, str]]:
        """Index all tool messages with their preceding assistant context.

        Returns list of (msg_index, tool_name, tool_args, content).
        """
        result = []
        for i, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            # Find the preceding assistant message to get tool info
            tool_name = ""
            tool_args = {}
            tool_call_id = msg.get("tool_call_id", "")

            # Look back for the matching assistant message
            for j in range(i - 1, -1, -1):
                asst = messages[j]
                if asst.get("role") != "assistant":
                    continue
                for tc in asst.get("tool_calls", []):
                    if tc.get("id") == tool_call_id:
                        func = tc.get("function", {})
                        tool_name = func.get("name", "")
                        try:
                            tool_args = json.loads(func.get("arguments", "{}"))
                        except (json.JSONDecodeError, TypeError):
                            tool_args = {}
                        break
                if tool_name:
                    break

            result.append((i, tool_name, tool_args, msg.get("content", "")))
        return result

    @property
    def tokens_saved(self) -> int:
        return self._tokens_saved_total

    @property
    def compressions_applied(self) -> int:
        return self._compressions_applied

    def get_metrics(self) -> dict:
        return {
            "history_tokens_saved": self._tokens_saved_total,
            "history_compressions_applied": self._compressions_applied,
        }


class SpinDetector:
    """Detects when the agent is stuck in a loop and intervenes.

    Patterns detected:
      1. Identical command repeat: same bash command run 3+ times with same output
      2. Identical file re-read: same file viewed 3+ times without edits between
      3. Test loop: same test run 3+ times with identical failures
      4. Edit-test cycle: same edit attempted, same test fails, repeated 2+ times

    Interventions (escalating):
      - First trigger: inject a nudge message suggesting a different approach
      - Second trigger: inject a stronger nudge to submit or try something new
      - Third trigger: force submit (return early termination signal)
    """

    NUDGE_GENTLE = (
        "SYSTEM NOTE: You appear to be repeating the same action without progress. "
        "Consider a different approach: try a different file, a different fix strategy, "
        "or if you believe your changes are sufficient, use the submit tool."
    )
    NUDGE_STRONG = (
        "SYSTEM NOTE: You have repeated the same approach multiple times without progress. "
        "You MUST either try a fundamentally different strategy or submit your current changes now."
    )

    def __init__(
        self,
        repeat_threshold: int = 3,
        max_nudges: int = 2,
    ):
        self.repeat_threshold = repeat_threshold
        self.max_nudges = max_nudges
        self._bash_history: list[tuple[str, str]] = []  # (normalized_cmd, output_hash)
        self._file_views: dict[str, list[int]] = {}  # path → [step_indices]
        self._last_edit_step: int = -1
        self._last_test_step: int = -1
        self._test_results: list[str] = []  # hashes of test output
        self._nudges_given: int = 0
        self._force_stop: bool = False
        self._spin_events: int = 0

    def record_action(
        self,
        step_idx: int,
        tool_name: str,
        tool_args: dict,
        observation: str,
    ) -> Optional[str]:
        """Record an action and check for spin. Returns nudge message or None.

        If force_stop is triggered, returns "FORCE_SUBMIT" sentinel.
        """
        if tool_name == "bash":
            return self._check_bash_spin(tool_args.get("command", ""), observation, step_idx)
        elif tool_name == "str_replace_editor":
            cmd = tool_args.get("command", "")
            path = tool_args.get("path", "")
            if cmd == "view":
                return self._check_view_spin(path, step_idx)
            elif cmd in ("str_replace", "create", "insert"):
                self._last_edit_step = step_idx
                self._last_test_step = -1  # reset test tracking after edit
                # Reset file view counts for this path (edit breaks the spin)
                self._file_views.pop(path, None)
        return None

    def _check_bash_spin(self, command: str, output: str, step_idx: int = 0) -> Optional[str]:
        """Check if bash commands are spinning."""
        import hashlib

        norm_cmd = command.strip()
        out_hash = hashlib.md5(output.encode()[:2000]).hexdigest()[:12]

        self._bash_history.append((norm_cmd, out_hash))

        # Check for identical command+output repeats
        recent = self._bash_history[-self.repeat_threshold :]
        if len(recent) == self.repeat_threshold:
            cmds = set(c for c, _ in recent)
            hashes = set(h for _, h in recent)
            if len(cmds) == 1 and len(hashes) == 1:
                return self._escalate("repeated identical bash command with same output")

        # Check for test loops: same test command, same failure, 3+ times
        if any(kw in norm_cmd for kw in ["pytest", "unittest", "npm test", "cargo test"]):
            self._test_results.append(out_hash)
            recent_tests = self._test_results[-self.repeat_threshold :]
            if (
                len(recent_tests) == self.repeat_threshold
                and len(set(recent_tests)) == 1
                and ("FAILED" in output or "Error" in output or "failed" in output)
            ):
                return self._escalate("same test failing identically across multiple runs")

            # S4: Test re-run without code change
            # Agent tests again but hasn't edited since the previous test ran
            if (self._last_test_step > self._last_edit_step and
                    step_idx - self._last_edit_step > 3):
                nudge = self._escalate(
                    "running tests again without making code changes since last test — "
                    "edit the code before re-testing"
                )
                self._last_test_step = step_idx
                return nudge
            self._last_test_step = step_idx

        return None

    def _check_view_spin(self, path: str, step_idx: int) -> Optional[str]:
        """Check if file views are spinning (reading same file without editing)."""
        if not path:
            return None

        if path not in self._file_views:
            self._file_views[path] = []
        self._file_views[path].append(step_idx)

        views = self._file_views[path]
        # Only flag if no edit happened since the first repeated view
        if len(views) >= self.repeat_threshold and self._last_edit_step < views[-self.repeat_threshold]:
            return self._escalate(f"reading {path} repeatedly without making edits")

        return None

    def _escalate(self, reason: str) -> str:
        """Escalate intervention based on how many nudges we've given."""
        self._spin_events += 1

        if self._nudges_given >= self.max_nudges:
            self._force_stop = True
            return "FORCE_SUBMIT"
        elif self._nudges_given == 0:
            self._nudges_given += 1
            return self.NUDGE_GENTLE
        else:
            self._nudges_given += 1
            return self.NUDGE_STRONG

    @property
    def should_force_stop(self) -> bool:
        return self._force_stop

    def get_metrics(self) -> dict:
        return {
            "spin_events_detected": self._spin_events,
            "nudges_given": self._nudges_given,
            "force_stopped": self._force_stop,
        }


class TestbedManager:
    """Manages cloning repos at specific commits for evaluation."""

    def __init__(self, base_dir: str = "/tmp/hop_eval_testbeds"):
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def setup_testbed(self, instance_id: str, mutation_patch: Optional[str] = None) -> Optional[str]:
        """Clone repo at the right commit and apply mutation to create buggy state.

        SWE-smith workflow:
          1. Clone repo at base commit (clean/working state)
          2. Apply the mutation patch (introduces the bug)
          3. Agent's job is to reverse/fix the mutation

        Args:
            instance_id: SWE-smith format: owner__repo.commit_short.mutation__hash
            mutation_patch: The diff that introduces the bug (from HF dataset 'patch' field).

        Returns:
            Path to testbed directory, or None on failure.
        """
        parts = instance_id.split(".")
        repo_part = parts[0]  # owner__repo
        commit_short = parts[1]  # 8-char commit hash

        owner, repo = repo_part.split("__", 1)
        github_url = f"https://github.com/{owner}/{repo}.git"

        testbed_path = self._base_dir / instance_id.replace(".", "_")

        # Reuse if already set up — reset and re-apply mutation
        if testbed_path.exists() and (testbed_path / ".git").exists():
            subprocess.run(["git", "checkout", "."], cwd=str(testbed_path), capture_output=True, timeout=10)
            subprocess.run(["git", "clean", "-fd"], cwd=str(testbed_path), capture_output=True, timeout=10)
            if mutation_patch:
                self._apply_mutation(str(testbed_path), mutation_patch)
            return str(testbed_path)

        # Clone — randomised delay to avoid GitHub rate-limits when 7 threads clone in parallel
        import random as _random
        time.sleep(_random.uniform(0, 5))
        print(f"    Cloning {github_url} @ {commit_short}...")
        try:
            result = subprocess.run(
                ["git", "clone", "--depth", "100", github_url, str(testbed_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"    Clone failed: {result.stderr[:200]}")
                return None

            # Checkout the specific commit
            result = subprocess.run(
                ["git", "checkout", commit_short],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(testbed_path),
            )
            if result.returncode != 0:
                # Shallow clone might not have the commit, try fetching
                subprocess.run(
                    ["git", "fetch", "--unshallow"],
                    capture_output=True,
                    timeout=120,
                    cwd=str(testbed_path),
                )
                result = subprocess.run(
                    ["git", "checkout", commit_short],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(testbed_path),
                )
                if result.returncode != 0:
                    print(f"    Checkout failed: {result.stderr[:200]}")
                    return None

            # Install package dependencies
            self._install_deps(str(testbed_path))

            # Apply mutation to introduce the bug
            if mutation_patch:
                if not self._apply_mutation(str(testbed_path), mutation_patch):
                    print(f"    WARNING: Mutation patch failed to apply")
                    return None

            return str(testbed_path)

        except subprocess.TimeoutExpired:
            print(f"    Clone timed out")
            return None
        except Exception as e:
            print(f"    Clone error: {e}")
            return None

    def _install_deps(self, testbed_path: str):
        """Install package in editable mode with ALL test dependencies."""
        setup_py = os.path.join(testbed_path, "setup.py")
        setup_cfg = os.path.join(testbed_path, "setup.cfg")
        pyproject = os.path.join(testbed_path, "pyproject.toml")

        if not (os.path.exists(setup_py) or os.path.exists(setup_cfg) or os.path.exists(pyproject)):
            return

        print(f"    Installing deps...", end="", flush=True)
        env = {**os.environ, "PIP_QUIET": "1", "PIP_NO_INPUT": "1"}

        # Step 1: Install the package with all available extras
        extras_to_try = ["dev", "test", "tests", "testing", "all"]
        installed = False
        for extra in extras_to_try:
            result = subprocess.run(
                ["pip", "install", "-e", f".[{extra}]"],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=testbed_path,
                env=env,
            )
            if result.returncode == 0:
                installed = True
                break
        if not installed:
            subprocess.run(
                ["pip", "install", "-e", "."],
                capture_output=True,
                text=True,
                timeout=180,
                cwd=testbed_path,
                env=env,
            )

        # Step 2: Install any requirements files (often contain test deps)
        req_files = [
            "requirements-test.txt",
            "requirements-tests.txt",
            "requirements-dev.txt",
            "requirements_dev.txt",
            "test-requirements.txt",
            "dev-requirements.txt",
            "requirements/test.txt",
            "requirements/dev.txt",
            "tests/requirements.txt",
        ]
        for req_file in req_files:
            req_path = os.path.join(testbed_path, req_file)
            if os.path.exists(req_path):
                subprocess.run(
                    ["pip", "install", "-r", req_path],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=testbed_path,
                    env=env,
                )

        # Step 3: Install tox deps if tox.ini exists (common for older projects)
        tox_ini = os.path.join(testbed_path, "tox.ini")
        if os.path.exists(tox_ini):
            try:
                with open(tox_ini) as f:
                    tox_content = f.read()
                # Extract deps from [testenv] section
                import re

                deps_match = re.search(r"\[testenv\].*?deps\s*=\s*(.*?)(?:\n\[|\Z)", tox_content, re.DOTALL)
                if deps_match:
                    deps_lines = deps_match.group(1).strip().split("\n")
                    pkgs = [
                        l.strip()
                        for l in deps_lines
                        if l.strip() and not l.strip().startswith("#") and not l.strip().startswith("[")
                    ]
                    if pkgs:
                        subprocess.run(
                            ["pip", "install"] + pkgs,
                            capture_output=True,
                            text=True,
                            timeout=120,
                            cwd=testbed_path,
                            env=env,
                        )
            except Exception:
                pass

        print(f" OK")

    def _apply_mutation(self, testbed_path: str, patch: str) -> bool:
        """Apply a mutation patch to introduce the bug."""
        patch_file = os.path.join(testbed_path, "_mutation.patch")
        with open(patch_file, "w") as f:
            f.write(patch)

        # Try multiple apply strategies (same as SWE-smith)
        apply_cmds = [
            ["git", "apply", "--verbose", patch_file],
            ["git", "apply", "--verbose", "--reject", patch_file],
            ["patch", "--batch", "--fuzz=5", "-p1", "-i", patch_file],
        ]

        for cmd in apply_cmds:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=testbed_path,
            )
            if result.returncode == 0:
                os.remove(patch_file)
                return True

        os.remove(patch_file)
        return False

    def cleanup(self, testbed_path: str):
        """Remove a testbed."""
        try:
            shutil.rmtree(testbed_path, ignore_errors=True)
        except Exception:
            pass


class HOPGate:
    """Applies HOP observation gating using trained models."""

    def __init__(self, models_dir: Optional[Path] = None, risk_threshold: float = 0.15):
        self._models_dir = models_dir
        self._risk_threshold = risk_threshold
        self._wrapper = None
        self._init_wrapper()

    def _init_wrapper(self):
        try:
            from .gate.swe_agent_wrapper import SWEAgentHOPWrapper

            self._wrapper = SWEAgentHOPWrapper(
                models_dir=str(self._models_dir) if self._models_dir else None,
                risk_threshold=self._risk_threshold,
                record_trajectory=True,
                live_metrics=True,
            )
        except Exception as e:
            print(f"  WARNING: HOP wrapper init failed: {e}, using heuristic fallback")
            try:
                from .gate.swe_agent_wrapper import SWEAgentHOPWrapper

                self._wrapper = SWEAgentHOPWrapper(
                    models_dir=None,
                    risk_threshold=self._risk_threshold,
                    record_trajectory=True,
                    live_metrics=True,
                )
            except Exception:
                self._wrapper = None

    def process(
        self,
        tool_name: str,
        args: dict,
        raw_observation: str,
        agent_thought: str,
        task: str,
    ) -> tuple[str, str, int, int]:
        """Process an observation through the gate.

        Returns: (served_content, tier, raw_tokens, served_tokens)
        """
        if self._wrapper is None:
            tokens = count_tokens(raw_observation)
            return raw_observation, "full", tokens, tokens

        normalized_tool, normalized_args = self._normalize_swe_smith_tool(tool_name, args)
        result = self._wrapper.process_observation(
            tool=normalized_tool,
            args=normalized_args,
            raw_output=raw_observation,
            agent_thought=agent_thought,
            task=task,
        )
        return result.content, result.tier_served, result.raw_tokens, result.served_tokens

    def _normalize_swe_smith_tool(self, tool_name: str, args: dict) -> tuple[str, dict]:
        """Normalize SWE-smith tools to wrapper-expected format."""
        if tool_name == "str_replace_editor":
            command = args.get("command", "")
            path = args.get("path", "")
            if command == "view":
                return "open", {"path": path, "file_path": path, **args}
            else:
                return "edit", {"path": path, "file_path": path, **args}
        elif tool_name == "bash":
            return "bash", args
        elif tool_name == "submit":
            return "submit", args
        return tool_name, args

    def get_metrics(self) -> dict:
        if self._wrapper:
            return self._wrapper.get_metrics()
        return {}

    def reset(self):
        if self._wrapper:
            self._wrapper.reset()


class LiveAgentRunner:
    """Runs a live agent on a task with real tool execution."""

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 30,
        gate: Optional[HOPGate] = None,
        history_compressor: Optional[HistoryCompressor] = None,
        spin_detector: Optional[SpinDetector] = None,
    ):
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._max_steps = max_steps
        self._gate = gate
        self._compressor = history_compressor
        self._spin_detector = spin_detector

    def run_task(
        self,
        task_id: str,
        problem_statement: str,
        testbed_path: str,
        condition: str,
        fail_to_pass: Optional[list[str]] = None,
        pass_to_pass: Optional[list[str]] = None,
    ) -> EvalResult:
        """Run the agent on a task with real tool execution.

        Args:
            task_id: Task identifier.
            problem_statement: The bug/feature description.
            testbed_path: Path to the cloned repo.
            condition: "baseline" or "gated".
            fail_to_pass: Tests that should pass after fix (from HF dataset).
            pass_to_pass: Tests that should remain passing (from HF dataset).

        Returns:
            EvalResult with all metrics.
        """
        start_time = time.time()
        steps: list[AgentStep] = []
        tier_counts: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_obs_raw = 0
        total_obs_served = 0
        submitted = False
        error_msg = ""

        # Set up tool executor
        executor = ToolExecutor(testbed_path)

        # Build user message from template
        user_message = INSTANCE_TEMPLATE.format(
            working_dir="/testbed",
            problem_statement=problem_statement,
        )

        # Build initial messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        if self._gate and condition in ("gated", "local_gated"):
            self._gate.reset()

        try:
            for step_idx in range(self._max_steps):
                # Compress stale history before sending to API
                if self._compressor and step_idx > 0:
                    self._compressor.compress(messages, step_idx)

                step_start = time.time()
                response = self._call_llm(messages)
                step_latency = (time.time() - step_start) * 1000

                if response is None:
                    error_msg = "LLM call returned None"
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                usage = response.usage
                step_input_tokens = usage.prompt_tokens if usage else 0
                step_output_tokens = usage.completion_tokens if usage else 0
                total_input_tokens += step_input_tokens
                total_output_tokens += step_output_tokens

                # No tool calls = agent finished
                if not assistant_msg.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.content or "",
                        }
                    )
                    break

                # Process tool call
                tool_call = assistant_msg.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                agent_thought = assistant_msg.content or ""

                # Execute tool FOR REAL
                raw_observation = executor.execute(tool_name, tool_args)

                # Apply gating if in gated condition (covers both "gated" and "local_gated")
                if condition in ("gated", "local_gated") and self._gate and tool_name != "submit":
                    served_obs, tier, raw_tokens, served_tokens = self._gate.process(
                        tool_name=tool_name,
                        args=tool_args,
                        raw_observation=raw_observation,
                        agent_thought=agent_thought,
                        task=problem_statement,
                    )
                else:
                    served_obs = raw_observation
                    tier = "full"
                    raw_tokens = count_tokens(raw_observation)
                    served_tokens = raw_tokens

                total_obs_raw += raw_tokens
                total_obs_served += served_tokens
                tier_counts[tier] = tier_counts.get(tier, 0) + 1

                # Record step
                step = AgentStep(
                    step_idx=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    agent_thought=agent_thought,
                    raw_observation=raw_observation,
                    served_observation=served_obs,
                    raw_obs_tokens=raw_tokens,
                    served_obs_tokens=served_tokens,
                    tier_served=tier,
                    input_tokens=step_input_tokens,
                    output_tokens=step_output_tokens,
                    latency_ms=step_latency,
                    cumulative_input_tokens=total_input_tokens,
                    cumulative_output_tokens=total_output_tokens,
                )
                steps.append(step)

                # Handle submit
                if tool_name == "submit":
                    submitted = True

                # Add to conversation history
                messages.append(
                    {
                        "role": "assistant",
                        "content": agent_thought,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": served_obs,
                    }
                )

                if submitted:
                    break

                # Spin detection: check if agent is stuck in a loop
                if self._spin_detector:
                    spin_signal = self._spin_detector.record_action(
                        step_idx=step_idx,
                        tool_name=tool_name,
                        tool_args=tool_args,
                        observation=raw_observation,
                    )
                    if spin_signal == "FORCE_SUBMIT":
                        submitted = True
                        break
                    elif spin_signal:
                        messages.append(
                            {
                                "role": "user",
                                "content": spin_signal,
                            }
                        )

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:300]}"

        duration = time.time() - start_time
        obs_savings = (1 - total_obs_served / max(total_obs_raw, 1)) * 100 if total_obs_raw > 0 else 0.0

        # Get the patch produced
        patch = executor.get_patch()

        # Run tests to verify if the task was actually resolved
        test_results = executor.run_tests(
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        resolved = test_results["resolved"] and bool(patch)

        # Collect compressor and spin detector metrics
        comp_metrics = self._compressor.get_metrics() if self._compressor else {}
        spin_metrics = self._spin_detector.get_metrics() if self._spin_detector else {}

        return EvalResult(
            task_id=task_id,
            condition=condition,
            submitted=submitted,
            num_steps=len(steps),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_observation_tokens_raw=total_obs_raw,
            total_observation_tokens_served=total_obs_served,
            observation_savings_pct=round(obs_savings, 1),
            patch_produced=patch[:5000],
            resolved=resolved,
            tests_passed=test_results["num_passed"],
            tests_failed=test_results["num_failed"],
            test_exit_code=test_results["exit_code"],
            f2p_passed=test_results.get("f2p_passed", 0),
            f2p_total=test_results.get("f2p_total", 0),
            p2p_passed=test_results.get("p2p_passed", 0),
            p2p_total=test_results.get("p2p_total", 0),
            steps=steps,
            tier_distribution=tier_counts,
            error=error_msg,
            duration_seconds=round(duration, 1),
            history_tokens_saved=comp_metrics.get("history_tokens_saved", 0),
            history_compressions_applied=comp_metrics.get("history_compressions_applied", 0),
            spin_events_detected=spin_metrics.get("spin_events_detected", 0),
            spin_nudges_given=spin_metrics.get("nudges_given", 0),
            spin_force_stopped=spin_metrics.get("force_stopped", False),
        )

    def _compact_messages(self, messages: list[dict]) -> None:
        """Compact conversation history when context window is exceeded.

        Summarizes the middle of the conversation, keeping the system prompt,
        original task, and recent messages intact.
        """
        if len(messages) <= 4:
            return

        system_msg = messages[0]
        task_msg = messages[1]
        # Keep last 4 messages (2 turns) for continuity
        recent = messages[-4:]

        # FIX: Ensure recent starts with 'assistant', not 'tool'
        # Otherwise after adding user summary, we get: [user, tool, ...] which violates role ordering
        while recent and recent[0].get("role") == "tool":
            recent = recent[1:]
        # Expand to ensure we have at least 2 messages if possible
        if len(recent) < 2 and len(messages) > 4:
            # Scan backwards to find the last complete turn (assistant + tool(s))
            for i in range(len(messages) - 1, 1, -1):
                if messages[i].get("role") == "assistant":
                    recent = messages[i:]
                    break

        middle = messages[2:-len(recent)] if recent else messages[2:]

        if not middle:
            return

        # Build a summary of the middle section
        summary_parts = []
        for msg in middle:
            role = msg.get("role", "")
            if role == "assistant":
                thought = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    func = tool_calls[0].get("function", {})
                    summary_parts.append(f"Action: {func.get('name', '?')}({func.get('arguments', '')[:100]})")
                elif thought:
                    summary_parts.append(f"Thought: {thought[:150]}")
            elif role == "tool":
                content = msg.get("content", "")
                summary_parts.append(f"Result: {content[:200]}")

        summary_text = (
            "[Context compacted - earlier steps summarized]\n"
            + "\n".join(summary_parts[-20:])  # Keep last 20 action/result pairs
        )

        # Rebuild messages in-place
        messages.clear()
        messages.append(system_msg)
        messages.append(task_msg)
        messages.append({"role": "user", "content": summary_text})
        messages.extend(recent)
        print(f"    [Compaction] Reduced context: kept system + task + summary + {len(recent)} recent msgs")

    def _call_llm(self, messages: list[dict], retries: int = 30):
        """Call the LLM API with retries. Auto-compacts on context overflow."""
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=min(8192, getattr(self, '_max_output_tokens', 8192)),
                )
                return response
            except Exception as e:
                err_str = str(e)
                if "maximum context length" in err_str or "max_model_len" in err_str or ("400" in err_str and "token" in err_str):
                    print(f"    [Context overflow] Compacting history...")
                    self._compact_messages(messages)
                    continue
                if attempt < retries - 1:
                    is_conn_err = "Connection" in err_str or "ConnectError" in err_str or "refused" in err_str or "503" in err_str
                    wait = min(60, 2 ** (attempt + 1)) if not is_conn_err else min(120, 10 * (attempt + 1))
                    print(f"    API error (attempt {attempt + 1}/{retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


AGENTDIET_REFLECTION_PROMPT = """You will analyze and compress a given step in the trajectory of an LLM agent solving a software engineering task. Your job is to reduce the length of the TARGET step by removing useless, redundant, and expired information, while preserving all information that the agent may still need in subsequent steps.

INPUT FORMAT:
The trajectory is provided as a sequence of steps wrapped in <step id="N"></step> tags. Each step contains:
- <assistant>: The agent's reasoning and tool call
- <tool_result>: The output returned by the tool

You must reduce ONLY the step marked with [REDUCE THIS STEP]. The other steps are provided as context.

TYPICAL WASTE TO REMOVE:
- (Useless information) Verbose test output listing many individually PASSED tests when only the summary or failures matter. Directory listings including cache/build files (__pycache__, .egg-info, node_modules, .git). Build output, compilation warnings, make entering/leaving directory messages.
- (Redundant information) File content that was already shown in a previous step in the context. In str_replace_editor calls: the old_str in the arguments repeats code already viewed, and the tool result repeats the new_str from the arguments. Repeated grep/find results for the same query.
- (Expired information) File contents from early exploration steps that the agent has since moved past (evidenced by later steps working on different files). Search results for hypotheses that were already confirmed or rejected in later steps.

GUIDELINES:
- Replace removed text with a SHORT takeaway in brackets, e.g., [73 tests passed, 1 xfailed in 4.48s] or [directory listing: 45 files, relevant: src/, tests/]
- Keep error messages, tracebacks, and failing test output VERBATIM — these are critical
- Keep file paths, function/class names, and line numbers that may be referenced later
- Keep the original structure: preserve the <assistant> and <tool_result> tags
- Keep the tool call command/path so the agent knows what action was taken
- Do NOT add new information, analysis, or commentary
- Do NOT remove content from steps other than the target

OUTPUT FORMAT:
Return ONLY the reduced version of the target step, wrapped in <step> tags with the same structure. Nothing else."""


class AgentDietRunner:
    """Runs a live agent with AgentDiet trajectory reduction (Algorithm 1).

    Hyperparameters (from paper's optimal settings):
      a=2: reduce step s-a when agent reaches step s
      b=1: context window includes b steps before target
      theta=500: skip reflection if step length < theta tokens
      LLM_reflect: cost-efficient model for reflection (e.g. claude-haiku-4-5)
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "claude-sonnet-4-6",
        reflect_model: str = "claude-haiku-4-5",
        max_steps: int = 30,
        a: int = 2,
        b: int = 1,
        theta: int = 500,
    ):
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._reflect_model = reflect_model
        self._max_steps = max_steps
        self._a = a
        self._b = b
        self._theta = theta

    def run_task(
        self,
        task_id: str,
        problem_statement: str,
        testbed_path: str,
        condition: str = "agentdiet",
        fail_to_pass: Optional[list[str]] = None,
        pass_to_pass: Optional[list[str]] = None,
    ) -> EvalResult:
        """Run agent with AgentDiet trajectory reduction."""
        start_time = time.time()
        steps: list[AgentStep] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_obs_raw = 0
        total_obs_served = 0
        reflection_input_tokens = 0
        reflection_output_tokens = 0
        reflection_calls = 0
        submitted = False
        error_msg = ""

        executor = ToolExecutor(testbed_path)

        user_message = INSTANCE_TEMPLATE.format(
            working_dir="/testbed",
            problem_statement=problem_statement,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Track steps as (assistant_msg_dict, tool_msg_dict) pairs for reflection
        trajectory_steps: list[tuple[dict, dict]] = []

        try:
            for step_idx in range(self._max_steps):
                step_start = time.time()
                response = self._call_llm(messages)
                step_latency = (time.time() - step_start) * 1000

                if response is None:
                    error_msg = "LLM call returned None"
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                usage = response.usage
                step_input_tokens = usage.prompt_tokens if usage else 0
                step_output_tokens = usage.completion_tokens if usage else 0
                total_input_tokens += step_input_tokens
                total_output_tokens += step_output_tokens

                if not assistant_msg.tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": assistant_msg.content or "",
                        }
                    )
                    break

                tool_call = assistant_msg.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                agent_thought = assistant_msg.content or ""

                raw_observation = executor.execute(tool_name, tool_args)
                raw_tokens = count_tokens(raw_observation)
                total_obs_raw += raw_tokens
                total_obs_served += raw_tokens  # AgentDiet serves full obs initially

                step = AgentStep(
                    step_idx=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    agent_thought=agent_thought,
                    raw_observation=raw_observation,
                    served_observation=raw_observation,
                    raw_obs_tokens=raw_tokens,
                    served_obs_tokens=raw_tokens,
                    tier_served="full",
                    input_tokens=step_input_tokens,
                    output_tokens=step_output_tokens,
                    latency_ms=step_latency,
                    cumulative_input_tokens=total_input_tokens,
                    cumulative_output_tokens=total_output_tokens,
                )
                steps.append(step)

                if tool_name == "submit":
                    submitted = True

                # Add to conversation (full observation)
                asst_dict = {
                    "role": "assistant",
                    "content": agent_thought,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_args),
                            },
                        }
                    ],
                }
                tool_dict = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": raw_observation,
                }
                messages.append(asst_dict)
                messages.append(tool_dict)
                trajectory_steps.append((asst_dict, tool_dict))

                if submitted:
                    break

                # --- AgentDiet reflection (Algorithm 1, lines 10-20) ---
                # s is 1-indexed step count; we reduce step (s - a)
                s = step_idx + 1
                target_idx = s - self._a  # 0-indexed in trajectory_steps

                if target_idx > 0 and target_idx < len(trajectory_steps):
                    target_asst, target_tool = trajectory_steps[target_idx]

                    # Serialize the full step to measure its length
                    target_serialized = self._serialize_step(target_asst, target_tool)
                    l_orig = count_tokens(target_serialized)

                    if l_orig > self._theta:
                        # Context window: steps [max(0, target-b) : s] (Algorithm 1, line 13)
                        ctx_start = max(0, target_idx - self._b)
                        ctx_end = min(len(trajectory_steps), s)
                        ctx_steps = trajectory_steps[ctx_start:ctx_end]

                        # Serialize full context with target marked
                        ctx_serialized = self._serialize_context(ctx_steps, target_idx - ctx_start)

                        # Call reflection LLM (Algorithm 1, line 14)
                        ref_result = self._call_reflection(ctx_serialized)
                        if ref_result:
                            reduced_text, ref_in_tok, ref_out_tok = ref_result
                            reflection_input_tokens += ref_in_tok
                            reflection_output_tokens += ref_out_tok
                            reflection_calls += 1

                            l_reduced = count_tokens(reduced_text)

                            # Apply only if benefit exceeds theta (Algorithm 1, line 16)
                            if l_orig - l_reduced > self._theta:
                                # Parse reduced text back into assistant + tool parts
                                reduced_asst_content, reduced_tool_content = self._parse_reduced_step(
                                    reduced_text, target_asst, target_tool
                                )

                                # Update trajectory_steps
                                new_asst = {**target_asst, "content": reduced_asst_content}
                                new_tool = {**target_tool, "content": reduced_tool_content}
                                trajectory_steps[target_idx] = (new_asst, new_tool)

                                # Update messages array: messages[0]=system, [1]=user, then pairs
                                asst_msg_idx = 2 + target_idx * 2
                                tool_msg_idx = asst_msg_idx + 1
                                if asst_msg_idx < len(messages):
                                    messages[asst_msg_idx] = new_asst
                                if tool_msg_idx < len(messages):
                                    messages[tool_msg_idx] = new_tool

                                # Track savings
                                tokens_saved = l_orig - l_reduced
                                total_obs_served -= tokens_saved

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:300]}"

        duration = time.time() - start_time
        obs_savings = (1 - total_obs_served / max(total_obs_raw, 1)) * 100 if total_obs_raw > 0 else 0.0

        patch = executor.get_patch()
        test_results = executor.run_tests(
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        resolved = test_results["resolved"] and bool(patch)

        return EvalResult(
            task_id=task_id,
            condition=condition,
            submitted=submitted,
            num_steps=len(steps),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_observation_tokens_raw=total_obs_raw,
            total_observation_tokens_served=total_obs_served,
            observation_savings_pct=round(obs_savings, 1),
            patch_produced=patch[:5000],
            resolved=resolved,
            tests_passed=test_results["num_passed"],
            tests_failed=test_results["num_failed"],
            test_exit_code=test_results["exit_code"],
            f2p_passed=test_results.get("f2p_passed", 0),
            f2p_total=test_results.get("f2p_total", 0),
            p2p_passed=test_results.get("p2p_passed", 0),
            p2p_total=test_results.get("p2p_total", 0),
            steps=steps,
            tier_distribution={},
            error=error_msg,
            duration_seconds=round(duration, 1),
            reflection_input_tokens=reflection_input_tokens,
            reflection_output_tokens=reflection_output_tokens,
            reflection_calls=reflection_calls,
        )

    def _serialize_step(self, asst: dict, tool: dict) -> str:
        """Serialize a single step (assistant + tool) into XML format."""
        tool_calls = asst.get("tool_calls", [])
        call_str = ""
        if tool_calls:
            func = tool_calls[0].get("function", {})
            call_str = f'<call tool="{func.get("name", "")}">{func.get("arguments", "")}</call>'

        return (
            f"<assistant>{asst.get('content', '')}\n{call_str}</assistant>\n"
            f"<tool_result>{tool.get('content', '')}</tool_result>"
        )

    def _serialize_context(self, ctx_steps: list[tuple[dict, dict]], target_offset: int) -> str:
        """Serialize context steps for reflection, marking the target."""
        parts = []
        for i, (asst, tool) in enumerate(ctx_steps):
            marker = " [REDUCE THIS STEP]" if i == target_offset else ""
            step_content = self._serialize_step(asst, tool)
            parts.append(f'<step id="{i}"{marker}>\n{step_content}\n</step>')
        return "\n".join(parts)

    def _parse_reduced_step(self, reduced_text: str, orig_asst: dict, orig_tool: dict) -> tuple[str, str]:
        """Parse the LLM's reduced output back into (assistant_content, tool_content).

        The LLM returns the reduced step in <assistant>...</assistant> <tool_result>...</tool_result> format.
        We extract both parts. If parsing fails, we treat the whole output as a reduced tool_result
        and keep the original assistant content.
        """
        import re

        asst_match = re.search(r"<assistant>(.*?)</assistant>", reduced_text, re.DOTALL)
        tool_match = re.search(r"<tool_result>(.*?)</tool_result>", reduced_text, re.DOTALL)

        if asst_match and tool_match:
            reduced_asst = asst_match.group(1).strip()
            # Remove the <call> tag from assistant content (it's in tool_calls, not content)
            reduced_asst = re.sub(r"<call[^>]*>.*?</call>", "", reduced_asst, flags=re.DOTALL).strip()
            reduced_tool = tool_match.group(1).strip()
        elif tool_match:
            # Only tool result was reduced, keep original assistant
            reduced_asst = orig_asst.get("content", "")
            reduced_tool = tool_match.group(1).strip()
        else:
            # Parsing failed — treat entire output as reduced tool content
            reduced_asst = orig_asst.get("content", "")
            reduced_tool = reduced_text

        return reduced_asst, reduced_tool

    def _call_reflection(self, context: str) -> Optional[tuple[str, int, int]]:
        """Call the reflection LLM. Returns (reduced_content, input_tokens, output_tokens)."""
        try:
            response = self._client.chat.completions.create(
                model=self._reflect_model,
                messages=[
                    {"role": "system", "content": AGENTDIET_REFLECTION_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=4096,
            )
            usage = response.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            content = response.choices[0].message.content or ""
            return content, in_tok, out_tok
        except Exception:
            return None

    def _call_llm(self, messages: list[dict], retries: int = 30):
        """Call the LLM API with retries."""
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=8192,
                )
                return response
            except Exception as e:
                err_str = str(e)
                if "maximum context length" in err_str or "max_model_len" in err_str or ("400" in err_str and "token" in err_str):
                    print(f"    [Context overflow] Compacting history...")
                    LiveAgentRunner._compact_messages(self, messages)
                    continue
                if attempt < retries - 1:
                    is_conn_err = "Connection" in err_str or "ConnectError" in err_str or "refused" in err_str or "503" in err_str
                    wait = min(60, 2 ** (attempt + 1)) if not is_conn_err else min(120, 10 * (attempt + 1))
                    print(f"    API error (attempt {attempt + 1}/{retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


class RepoIndex:
    """Hierarchical file index built via AST parsing. Shared across tasks in same repo.

    Stores: file → {hash, lines, classes: {name: {line, sig, doc, methods}}, functions: {name: {line, sig, doc}}}
    Updated incrementally on edits (re-parse single file, ~5ms).
    """

    CACHE_DIR = Path("/tmp/repo_indexes")

    def __init__(self, testbed_path: str, repo_name: str):
        self._testbed = testbed_path
        self._repo_name = repo_name
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._index_path = self.CACHE_DIR / f"{repo_name}.json"
        self._index: dict[str, dict] = {}
        self._load_or_build()

    def _load_or_build(self):
        """Load cached index or build from scratch."""
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text())
                return
            except (json.JSONDecodeError, OSError):
                pass
        self._build_full()
        self._save()

    def _build_full(self):
        """Full AST parse of all Python files in the repo."""
        import ast
        import hashlib

        self._index = {}
        for root, dirs, files in os.walk(self._testbed):
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", ".tox", "node_modules", ".eggs", "*.egg-info")]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, self._testbed)
                self._parse_file(rel_path, full_path)

    def _parse_file(self, rel_path: str, full_path: str):
        """Parse a single Python file and update the index entry."""
        import ast
        import hashlib

        try:
            content = open(full_path).read()
        except (OSError, UnicodeDecodeError):
            return

        file_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        lines = content.count("\n") + 1

        entry = {"hash": file_hash, "lines": lines, "classes": {}, "functions": {}}

        try:
            tree = ast.parse(content)
        except SyntaxError:
            self._index[rel_path] = entry
            return

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_info = self._extract_class(node)
                entry["classes"][node.name] = cls_info
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                func_info = self._extract_function(node)
                entry["functions"][node.name] = func_info

        self._index[rel_path] = entry

    def _extract_class(self, node) -> dict:
        """Extract class info: line, signature, docstring, methods."""
        import ast

        doc = ast.get_docstring(node) or ""
        if len(doc) > 100:
            doc = doc[:100] + "..."

        methods = {}
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[item.name] = self._extract_function(item)

        return {
            "line": node.lineno,
            "doc": doc,
            "methods": methods,
        }

    def _extract_function(self, node) -> dict:
        """Extract function info: line, signature, docstring."""
        import ast

        doc = ast.get_docstring(node) or ""
        if len(doc) > 80:
            doc = doc[:80] + "..."

        sig = self._format_sig(node)

        return {
            "line": node.lineno,
            "sig": sig,
            "doc": doc,
        }

    def _format_sig(self, node) -> str:
        """Format function signature from AST."""
        import ast

        args = node.args
        parts = []

        # positional args
        for i, arg in enumerate(args.args):
            name = arg.arg
            ann = ""
            if arg.annotation:
                try:
                    ann = ": " + ast.unparse(arg.annotation)
                except Exception:
                    pass
            parts.append(f"{name}{ann}")

        # *args
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")

        # **kwargs
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        sig = f"({', '.join(parts)})"

        # return annotation
        if node.returns:
            try:
                sig += f" -> {ast.unparse(node.returns)}"
            except Exception:
                pass

        if len(sig) > 120:
            sig = sig[:120] + "..."

        return sig

    def update_file(self, rel_path: str):
        """Re-parse a single file after an edit. ~5ms."""
        full_path = os.path.join(self._testbed, rel_path)
        if os.path.exists(full_path):
            self._parse_file(rel_path, full_path)
        else:
            self._index.pop(rel_path, None)
        self._save()

    def _save(self):
        """Persist index to disk."""
        try:
            self._index_path.write_text(json.dumps(self._index))
        except OSError:
            pass

    def browse(self, path: str) -> str:
        """Return hierarchical summary for a path (file or directory).

        For a directory: lists files with line counts and top-level symbols.
        For a file: shows classes, methods, functions with signatures and docstrings.
        """
        # Normalize path
        if path.startswith("/testbed/"):
            rel_path = path[len("/testbed/"):]
        elif path.startswith("/testbed"):
            rel_path = path[len("/testbed"):]
        elif path.startswith("/"):
            rel_path = os.path.relpath(path, self._testbed)
        else:
            rel_path = path

        rel_path = rel_path.strip("/")

        # Check if it's a file in the index
        if rel_path in self._index:
            return self._format_file(rel_path)

        # Directory: find all files under this path
        prefix = rel_path + "/" if rel_path else ""
        matching = {}
        for fp, entry in self._index.items():
            if fp.startswith(prefix) or not prefix:
                matching[fp] = entry

        if not matching:
            return f"No indexed Python files found under '{rel_path}'."

        return self._format_directory(rel_path or ".", matching)

    def _format_file(self, rel_path: str) -> str:
        """Detailed view of one file: all classes, methods, functions.

        For small files (<200 lines): show full content with line numbers.
        This prevents the "false confidence" problem where the agent sees
        an outline, thinks it understands, and edits without reading.
        """
        entry = self._index[rel_path]

        # Small files: show full content (costs ~500 tok but prevents false confidence)
        if entry["lines"] <= 200:
            full_path = os.path.join(self._testbed, rel_path)
            try:
                content = open(full_path).read()
                lines = content.split("\n")
                numbered = "\n".join(f"  {i+1:>4}\t{line}" for i, line in enumerate(lines))
                return f"# {rel_path} ({entry['lines']} lines)\n\n{numbered}"
            except (OSError, UnicodeDecodeError):
                pass

        # Large files: show outline with signatures and docstrings
        parts = [f"# {rel_path} ({entry['lines']} lines)"]
        parts.append(f"<NOTE>This file is too large to display entirely. Showing abbreviated version with {entry['lines']} lines total.</NOTE>")

        for cls_name, cls_info in entry.get("classes", {}).items():
            doc_s = f'  "{cls_info["doc"]}"' if cls_info["doc"] else ""
            parts.append(f"\n  class {cls_name} (L{cls_info['line']}){doc_s}")
            for meth_name, meth_info in cls_info.get("methods", {}).items():
                doc_m = f'  # {meth_info["doc"]}' if meth_info["doc"] else ""
                parts.append(f"    .{meth_name}{meth_info['sig']} L{meth_info['line']}{doc_m}")

        for func_name, func_info in entry.get("functions", {}).items():
            doc_f = f'  # {func_info["doc"]}' if func_info["doc"] else ""
            parts.append(f"\n  def {func_name}{func_info['sig']} L{func_info['line']}{doc_f}")

        return "\n".join(parts)

    def _format_directory(self, dir_path: str, matching: dict[str, dict]) -> str:
        """Summary view of a directory: files with symbol counts."""
        parts = [f"# Index of {dir_path}/ ({len(matching)} Python files)"]

        # Group by immediate subdirectory
        subdirs: dict[str, list] = {}
        direct_files: list[tuple[str, dict]] = []

        prefix_len = len(dir_path) + 1 if dir_path != "." else 0
        for fp, entry in sorted(matching.items()):
            relative = fp[prefix_len:]
            if "/" in relative:
                subdir = relative.split("/")[0]
                if subdir not in subdirs:
                    subdirs[subdir] = []
                subdirs[subdir].append((fp, entry))
            else:
                direct_files.append((fp, entry))

        # Show subdirectories as summaries
        for subdir, files in sorted(subdirs.items()):
            total_lines = sum(e["lines"] for _, e in files)
            total_classes = sum(len(e.get("classes", {})) for _, e in files)
            total_funcs = sum(len(e.get("functions", {})) for _, e in files)
            parts.append(f"  {subdir}/ ({len(files)} files, {total_lines} lines, {total_classes} classes, {total_funcs} functions)")

        # Show direct files with their top symbols
        for fp, entry in direct_files:
            fname = os.path.basename(fp)
            n_cls = len(entry.get("classes", {}))
            n_func = len(entry.get("functions", {}))
            symbols = []
            for c in list(entry.get("classes", {}).keys())[:3]:
                symbols.append(c)
            for f in list(entry.get("functions", {}).keys())[:3]:
                symbols.append(f + "()")
            sym_s = ", ".join(symbols[:4])
            if len(symbols) > 4:
                sym_s += f", +{len(symbols)-4} more"
            parts.append(f"  {fname} ({entry['lines']}L, {n_cls}C, {n_func}F) [{sym_s}]")

        return "\n".join(parts)

    def get_token_count(self) -> int:
        """Estimate tokens in the full index dump."""
        return count_tokens(json.dumps(self._index))


BROWSE_TOOL = {
    "type": "function",
    "function": {
        "name": "browse",
        "description": "Browse the repository index to see file structure, classes, methods, and functions "
        "WITHOUT reading full file contents. Use this FIRST to understand the codebase layout, "
        "then use str_replace_editor view only when you need to see actual implementation details.\n"
        "* For a directory: shows files with line counts, classes, and function names.\n"
        "* For a file: shows all classes, methods, functions with signatures and one-line docs.\n"
        "* Much cheaper than reading full files — use this for exploration/navigation.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to browse (file or directory). Use /testbed or /testbed/subdir.",
                }
            },
            "required": ["path"],
        },
    },
}


TOOLS_WITH_BROWSE = TOOLS + [BROWSE_TOOL]


class IndexedToolExecutor(ToolExecutor):
    """ToolExecutor that also handles the 'browse' tool and updates the index on edits."""

    def __init__(self, testbed_dir: str, repo_index: RepoIndex):
        super().__init__(testbed_dir)
        self._repo_index = repo_index

    def execute(self, tool_name: str, args: dict) -> str:
        if tool_name == "browse":
            return self._exec_browse(args.get("path", "/testbed"))

        result = super().execute(tool_name, args)

        # Update index on file modifications
        if tool_name == "str_replace_editor":
            cmd = args.get("command", "")
            if cmd in ("str_replace", "create", "insert"):
                path = args.get("path", "")
                rel = self._to_rel_path(path)
                if rel and rel.endswith(".py"):
                    self._repo_index.update_file(rel)

        return result

    def _exec_browse(self, path: str) -> str:
        return f"OBSERVATION:\n{self._repo_index.browse(path)}"

    def _to_rel_path(self, path: str) -> str:
        if path.startswith("/testbed/"):
            return path[len("/testbed/"):]
        elif path.startswith("/testbed"):
            return path[len("/testbed"):]
        elif not path.startswith("/"):
            return path
        return os.path.relpath(path, self._testbed)


class TROTv3Runner:
    """TROT v3: Best of HOPRepoLLM (efficiency) + TACO (persistence) + fixed early stopping.

    Key changes vs HOPRepoLLM/TROT v2:
      1. NO DoomDetector or TrajectoryMonitor — never kill a trajectory early
      2. Critical observation bypass — errors/tests/tracebacks always served FULL
      3. Post-edit protection — 2 steps after an edit, observations served FULL
      4. Progress Nudger (replaces DoomDetector) — GUIDE only, never EXIT
      5. Adaptive patience — only nudge when genuinely stuck, not just "slow"
      6. Anchor-aware history compression — don't stub recently-referenced content

    Architecture (layers in order):
      1. RepoIndex browse tool — agent navigates structure cheaply
      2. Critical bypass check — errors/tests/post-edit → skip gate, serve FULL
      3. HOP gate — observation compression on non-critical reads
      4. ObservationCompressor — deterministic patterns (edit confirmations, etc.)
      5. HistoryCompressor — stale-step compression (recency_window=5)
      6. SpinDetector — loop breaking (repeat_threshold=3, max_nudges=2)
      7. Progress Nudger — gentle guidance when stuck (never exits)
      8. Targeted Haiku reflection — only for huge obs after all else applied
    """

    # Patterns that indicate critical observations (never compress these)
    _CRITICAL_PATTERNS = (
        "Traceback (most recent call last)",
        "Error:",
        "error:",
        "FAILED",
        "ERRORS",
        "AssertionError",
        "TypeError",
        "ValueError",
        "KeyError",
        "AttributeError",
        "ImportError",
        "ModuleNotFoundError",
        "NameError",
        "IndexError",
        "SyntaxError",
        "RuntimeError",
        "Exception:",
        "raise ",
        "did not appear verbatim",
        "No replacement was performed",
    )

    # Tighter bypass: only genuine test failures, Python tracebacks, and edit failures.
    # Avoids false-positives from "Error:" appearing in source code, git output, etc.
    _CRITICAL_PATTERNS_TIGHT = (
        "Traceback (most recent call last)",
        "did not appear verbatim",
        "No replacement was performed",
    )

    # Progress Nudger messages (GUIDE only, never EXIT)
    _NUDGE_EXPLORATION_DRIFT = (
        "SYSTEM NOTE: You have spent many steps exploring without making an edit. "
        "Based on what you've read, commit to a fix now. Edit the most likely file, "
        "then test your change."
    )
    _NUDGE_BUDGET_PRESSURE = (
        "SYSTEM NOTE: You have used most of your step budget without a passing test. "
        "If your current approach isn't working, try a simpler, more targeted fix. "
        "Re-read the failing test to understand exactly what's expected, then make a minimal change."
    )
    _NUDGE_STUCK = (
        "SYSTEM NOTE: You seem stuck. Consider: (1) re-reading the failing test to understand "
        "what's expected, (2) trying a different approach to the fix, or (3) reverting your "
        "last change and trying a simpler solution."
    )

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "claude-sonnet-4-6",
        reflect_model: str = "claude-haiku-4-5",
        max_steps: int = 50,
        gate: Optional["HOPGate"] = None,
        spin_detector: Optional["SpinDetector"] = None,
        history_compressor: Optional["HistoryCompressor"] = None,
        observation_compressor: Optional["ObservationCompressor"] = None,
        semantic_deduplicator: Optional["SemanticDeduplicator"] = None,
        # Reflection params
        a: int = 2,
        b: int = 1,
        theta: int = 3000,
        max_reflection_calls: int = 5,
        # Progress nudger params
        nudge_start_step: int = 15,
        nudge_interval: int = 8,
        max_nudges: int = 3,
        # Post-edit protection window
        post_edit_protection: int = 2,
        # Ablation flags
        disable_critical_bypass: bool = False,
        disable_submit_gate: bool = False,
        disable_history_compress: bool = False,
        tight_critical_bypass: bool = False,
    ):
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._reflect_model = reflect_model
        self._max_steps = max_steps
        self._gate = gate
        self._spin_detector = spin_detector
        self._compressor = history_compressor or HistoryCompressor(recency_window=5, min_tokens_to_compress=200)
        self._obs_compressor = observation_compressor or ObservationCompressor()
        self._deduplicator = semantic_deduplicator
        self._a = a
        self._b = b
        self._theta = theta
        self._max_reflection_calls = max_reflection_calls
        # Progress nudger
        self._nudge_start = nudge_start_step
        self._nudge_interval = nudge_interval
        self._max_nudges = max_nudges
        # Post-edit protection — serve full for N steps after any edit
        self._post_edit_protection = max(post_edit_protection, 5)
        # Ablation flags
        self._disable_critical_bypass = disable_critical_bypass
        self._disable_submit_gate = disable_submit_gate
        self._disable_history_compress = disable_history_compress
        self._tight_critical_bypass = tight_critical_bypass

    def _is_critical_observation(self, tool_name: str, tool_args: dict, observation: str) -> bool:
        """Veto rules: observations that must never be compressed.

        The agent relies on this content, so it is always served in full:
          - test output that contains failures or assertions
          - targeted file views (the agent asked for specific lines)
          - error messages and tracebacks
        """
        if not observation:
            return False

        sample = observation[:3000]

        # Veto: test output with failures or assertions
        if tool_name == "bash":
            cmd = tool_args.get("command", "")
            if any(kw in cmd for kw in ("pytest", "test", "unittest", "python -m pytest")):
                # tight: only genuine pytest failure markers, not broad string matches
                tight_test_patterns = ("FAILED", "ERROR", "Traceback", "AssertionError")
                broad_test_patterns = ("FAILED", "ERROR", "error", "Traceback",
                                       "assert", "Expected", "expected", "!=")
                patterns = tight_test_patterns if self._tight_critical_bypass else broad_test_patterns
                if any(p in sample for p in patterns):
                    return True
                return False

        # Veto: targeted view, the agent asked for specific lines
        if tool_name == "str_replace_editor" and tool_args.get("view_range"):
            return True

        # Veto: error and traceback patterns
        patterns = self._CRITICAL_PATTERNS_TIGHT if self._tight_critical_bypass else self._CRITICAL_PATTERNS
        for pattern in patterns:
            if pattern in sample:
                return True

        return False

    def _should_compress_observation(self, tool_name: str, tool_args: dict, observation: str) -> Optional[str]:
        """Compress rules: observations that are safe to compress.

        The agent rarely uses this content. Returns the compressed observation,
        or None to let the other layers handle it. Rules cover empty or short
        bash output, long passing test runs, long bash output, git log/diff/show
        output, file listings, cat of a script the agent just wrote, and re-reads
        of files the agent has already seen.
        """
        if not observation:
            return None

        # Compress: empty or short bash output, which the agent rarely uses
        if tool_name == "bash" and len(observation.strip()) < 80:
            if "error" not in observation.lower() and "Error" not in observation:
                return "Your command ran successfully and did not produce any output."

        if tool_name == "bash":
            cmd = tool_args.get("command", "")

            # Compress: all-passing verbose tests — only large suites (>15 PASSED, >1500 tok)
            # Raised from >5/>300 to avoid hiding small targeted test results the agent needs
            if any(kw in cmd for kw in ("pytest", "test", "unittest")):
                if ("FAILED" not in observation and "Error" not in observation
                    and observation.count("PASSED") > 15 and len(observation) // 4 > 1500):
                    for line in reversed(observation.split("\n")):
                        if "passed" in line.lower() and ("second" in line.lower() or "s)" in line):
                            return f"OBSERVATION:\n{line.strip()}"
                    return f"OBSERVATION:\n{observation.count('PASSED')} tests passed (output compressed)"

            # Compress git log/blame output (not diff/show — agent needs those for code review)
            if any(kw in cmd for kw in ("git log", "git blame")):
                lines = observation.split("\n")
                if len(lines) > 15 and "Error" not in observation[:500]:
                    head = "\n".join(lines[:10])
                    return f"{head}\n\n[... {len(lines) - 10} more lines of git output omitted]"

            # Compress file listing output
            # find/ls results — agent usually only needs a few paths
            first_word = cmd.strip().split()[0] if cmd.strip() else ""
            if first_word in ("find", "ls", "tree") and observation.count("\n") > 20:
                lines = observation.split("\n")
                head = "\n".join(lines[:15])
                return f"{head}\n\n[... {len(lines) - 15} more files/dirs, {len(lines)} total]"

            # Compress cat of reproduce scripts
            # Agent creates a script then cats it — it already knows what it wrote
            if first_word == "cat" and any(kw in cmd for kw in ("reproduce", "test_", "debug", "check_")):
                obs_tokens = len(observation) // 4
                if obs_tokens > 100 and "Error" not in observation[:500]:
                    lines = observation.split("\n")
                    return f"OBSERVATION:\n[Script content: {len(lines)} lines, {obs_tokens} tokens — you wrote this]"

            # Long non-test bash output: only compress if no code and no git diff
            # Raised threshold to 150 lines; protect code output and git diffs
            import re as _re_bash
            if (observation.count("\n") > 150
                and not any(kw in cmd for kw in ("pytest", "test", "unittest", "git diff", "git show"))
                and "Error" not in observation[:1000] and "Traceback" not in observation[:1000]
                and "error" not in observation[:1000]
                and not _re_bash.search(r"(?:def |class |async def )", observation[:3000])):
                lines = observation.split("\n")
                head = "\n".join(lines[:20])
                tail = "\n".join(lines[-15:])
                return f"{head}\n\n... ({len(lines) - 35} lines omitted) ...\n\n{tail}"

        # Compress re-reads of files
        # If agent views a file it already read, compress to outline
        if tool_name == "str_replace_editor" and tool_args.get("command") == "view":
            file_path = tool_args.get("path", "")
            if file_path and file_path in self._files_fully_read:
                obs_tokens = len(observation) // 4
                if obs_tokens > 500 and not tool_args.get("view_range"):
                    # Re-read of full file → compress to first/last + note
                    lines = observation.split("\n")
                    if len(lines) > 30:
                        head = "\n".join(lines[:15])
                        tail = "\n".join(lines[-10:])
                        return f"{head}\n\n[... {len(lines) - 25} lines omitted — you previously read this file ...]\n\n{tail}"

        return None

    def _should_nudge(self, step_idx: int, steps: list["AgentStep"]) -> Optional[str]:
        """Progress Nudger: returns guidance message or None. NEVER returns EXIT.

        This replaces DoomDetector. Key difference: we only guide, never kill.
        The agent always gets to use its full step budget.
        """
        if step_idx < self._nudge_start:
            return None

        # Only check at intervals
        if (step_idx - self._nudge_start) % self._nudge_interval != 0:
            return None

        # Count nudges already given (via tracking)
        nudges_given = getattr(self, "_nudges_given_count", 0)
        if nudges_given >= self._max_nudges:
            return None

        # Extract signals
        has_edited = False
        has_test_pass = False
        last_edit_step = -1
        last_productive_step = -1
        steps_viewing_only = 0

        for s in steps:
            if s.step_idx > step_idx:
                break
            if s.tool_name == "str_replace_editor":
                cmd = s.tool_args.get("command", "")
                if cmd in ("str_replace", "create", "insert"):
                    has_edited = True
                    last_edit_step = s.step_idx
                    last_productive_step = s.step_idx
                elif cmd == "view" and s.step_idx > step_idx - 8:
                    steps_viewing_only += 1
            elif s.tool_name == "bash":
                cmd = s.tool_args.get("command", "")
                obs = s.raw_observation if hasattr(s, "raw_observation") else ""
                if any(kw in cmd for kw in ("pytest", "test", "unittest")):
                    if "passed" in obs.lower() and "failed" not in obs.lower():
                        has_test_pass = True
                        last_productive_step = s.step_idx

        steps_without_productive = step_idx - last_productive_step if last_productive_step >= 0 else step_idx

        # Don't nudge if recently productive
        if steps_without_productive <= 5:
            return None

        # Exploration drift: many views, no edit
        if not has_edited and steps_viewing_only >= 5 and step_idx >= 15:
            self._nudges_given_count = nudges_given + 1
            return self._NUDGE_EXPLORATION_DRIFT

        # Budget pressure: past 70% without a passing test
        if step_idx >= int(self._max_steps * 0.7) and not has_test_pass:
            self._nudges_given_count = nudges_given + 1
            return self._NUDGE_BUDGET_PRESSURE

        # General stuck: no progress for 10+ steps
        if steps_without_productive >= 10:
            self._nudges_given_count = nudges_given + 1
            return self._NUDGE_STUCK

        return None

    def _try_extract_and_replace(
        self,
        messages: list[dict],
        traj_msg_indices: list[tuple[int, int]],
        trajectory_steps: list[tuple[dict, dict]],
        already_compressed: set,
        current_step: int,
        steps: list["AgentStep"],
    ) -> int:
        """Proactive context dropping: replace observations the agent already extracted from.

        For each observation older than 3 steps:
          1. Check if agent's subsequent thought referenced its content
          2. Check it's NOT an error/test failure (always keep those)
          3. Check agent is NOT about to edit that file
          4. If safe: replace with a short stub summarizing what was found

        Returns tokens saved.
        """
        tokens_saved = 0
        # Only process steps that are old enough (not in last 3)
        max_replaceable = len(traj_msg_indices) - 3

        for traj_idx in range(max(0, max_replaceable - 5), max_replaceable):
            if traj_idx in already_compressed:
                continue
            if traj_idx >= len(traj_msg_indices):
                continue

            asst_msg_idx, tool_msg_idx = traj_msg_indices[traj_idx]
            if tool_msg_idx >= len(messages):
                continue

            tool_content = messages[tool_msg_idx].get("content", "")
            if not tool_content or len(tool_content) < 300:
                continue  # already small, not worth replacing

            # Safety: never drop errors or test failures
            if any(p in tool_content[:2000] for p in (
                "Traceback", "Error:", "FAILED", "AssertionError",
                "TypeError", "ValueError", "did not appear verbatim",
            )):
                continue

            # Get the step info
            if traj_idx >= len(steps):
                continue
            step = steps[traj_idx]

            # Get subsequent thoughts (what the agent extracted)
            subsequent_thoughts = []
            for s in steps[traj_idx + 1: traj_idx + 4]:
                if s.agent_thought:
                    subsequent_thoughts.append(s.agent_thought)

            if not subsequent_thoughts:
                continue

            # Check if agent referenced content from this observation
            obs_tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{4,}\b", tool_content[:2000]))
            common_words = {"self", "None", "True", "False", "return", "import", "class",
                           "print", "with", "that", "this", "have", "will", "file", "line",
                           "test", "result", "data", "name", "path", "type", "value",
                           "from", "args", "function", "should", "would", "could"}
            obs_tokens -= common_words

            thought_text = " ".join(subsequent_thoughts)
            thought_tokens = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{4,}\b", thought_text))
            overlap = obs_tokens & thought_tokens

            # Agent extracted info (referenced 3+ tokens) → safe to replace
            if len(overlap) < 3:
                continue

            # Safety: don't drop if agent's NEXT action edits this same file
            if traj_idx + 1 < len(steps):
                next_step = steps[traj_idx + 1]
                if (next_step.tool_name == "str_replace_editor" and
                    next_step.tool_args.get("command") in ("str_replace", "create", "insert")):
                    next_path = next_step.tool_args.get("path", "")
                    this_path = step.tool_args.get("path", "")
                    if next_path and this_path and next_path == this_path:
                        continue

            # Build replacement stub
            tool_name = step.tool_name
            tool_args = step.tool_args
            extracted = sorted(overlap)[:8]

            if tool_name == "str_replace_editor" and tool_args.get("command") == "view":
                path = tool_args.get("path", "unknown")
                stub = f"[Previously read {path} — noted: {', '.join(extracted)}]"
            elif tool_name == "bash":
                cmd = tool_args.get("command", "")[:60]
                stub = f"[Ran: {cmd} — extracted: {', '.join(extracted)}]"
            elif tool_name == "browse":
                path = tool_args.get("path", ".")
                stub = f"[Browsed {path} — noted: {', '.join(extracted)}]"
            else:
                stub = f"[{tool_name} output processed — key info: {', '.join(extracted)}]"

            # Replace in messages
            old_tokens = count_tokens(tool_content)
            messages[tool_msg_idx]["content"] = stub
            new_tokens = count_tokens(stub)
            tokens_saved += old_tokens - new_tokens
            already_compressed.add(traj_idx)

        return tokens_saved

    def run_task(
        self,
        task_id: str,
        problem_statement: str,
        testbed_path: str,
        condition: str = "trot_v3",
        fail_to_pass: Optional[list[str]] = None,
        pass_to_pass: Optional[list[str]] = None,
    ) -> EvalResult:
        """Run agent with TROT v3: efficient + persistent, never kills early."""
        start_time = time.time()
        steps: list[AgentStep] = []
        tier_counts: dict[str, int] = {}
        total_input_tokens = 0
        total_output_tokens = 0
        total_obs_raw = 0
        total_obs_served = 0
        reflection_input_tokens = 0
        reflection_output_tokens = 0
        reflection_calls = 0
        history_tokens_saved = 0
        submitted = False
        error_msg = ""
        critical_bypass_count = 0
        post_edit_bypass_count = 0

        # Progress nudger state
        self._nudges_given_count = 0

        repo_name = task_id.split(".")[0]
        repo_index = RepoIndex(testbed_path, repo_name)
        executor = IndexedToolExecutor(testbed_path, repo_index)

        self._obs_compressor.set_problem_statement(problem_statement)

        user_message = INSTANCE_TEMPLATE.format(
            working_dir="/testbed",
            problem_statement=problem_statement,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        trajectory_steps: list[tuple[dict, dict]] = []
        traj_msg_indices: list[tuple[int, int]] = []
        deterministically_compressed: set[int] = set()

        # Track last edit step for post-edit protection
        last_edit_step_idx = -100

        # Track files fully read (view command) vs only browsed
        self._files_fully_read = set()
        self._browse_edit_nudges = 0

        # Test-before-submit: track if agent has run tests successfully
        self._has_test_pass = False
        self._last_test_passed = False
        self._last_test_step = -1
        self._test_submit_blocks = 0

        # V3.1: Same-diff submit blocker
        self._submit_diffs: list[str] = []
        self._same_diff_nudged = False
        self._last_submit_block_step = -100

        # V3.1: Consecutive identical bash output detection
        self._last_bash_outputs: list[str] = []
        self._bash_spin_nudges = 0

        # V3.1: Early exit budget — stop if no progress after step 25
        self._last_source_edit_step = -1
        self._last_test_improvement_step = -1

        if self._gate:
            self._gate.reset()
        if self._spin_detector:
            self._spin_detector._bash_history = []
            self._spin_detector._file_views = {}
            self._spin_detector._last_edit_step = -1
            self._spin_detector._test_results = []
            self._spin_detector._nudges_given = 0
            self._spin_detector._force_stop = False
            self._spin_detector._spin_events = 0

        try:
            for step_idx in range(self._max_steps):
                # Deterministic history compression before LLM call
                if step_idx > 0 and not self._disable_history_compress:
                    pre_tokens = sum(count_tokens(m.get("content", "")) for m in messages if m.get("role") == "tool")
                    self._compressor.compress(messages, step_idx)
                    if self._deduplicator:
                        self._deduplicator.deduplicate(messages, step_idx)
                    post_tokens = sum(count_tokens(m.get("content", "")) for m in messages if m.get("role") == "tool")
                    saved_this_step = pre_tokens - post_tokens
                    history_tokens_saved += saved_this_step

                    if saved_this_step > 0 and traj_msg_indices:
                        for traj_idx in range(len(traj_msg_indices)):
                            _, tool_msg_idx = traj_msg_indices[traj_idx]
                            if tool_msg_idx < len(messages) and messages[tool_msg_idx].get("content", "").startswith("["):
                                deterministically_compressed.add(traj_idx)

                    # === TROT v3: Proactive context dropping ===
                    # Replace old observations the agent already extracted info from
                    if step_idx >= 4 and len(traj_msg_indices) >= 4:
                        drop_saved = self._try_extract_and_replace(
                            messages, traj_msg_indices, trajectory_steps,
                            deterministically_compressed, step_idx, steps,
                        )
                        history_tokens_saved += drop_saved

                step_start = time.time()
                response = self._call_llm(messages)
                step_latency = (time.time() - step_start) * 1000

                if response is None:
                    error_msg = "LLM call returned None"
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                usage = response.usage
                step_input_tokens = usage.prompt_tokens if usage else 0
                step_output_tokens = usage.completion_tokens if usage else 0
                total_input_tokens += step_input_tokens
                total_output_tokens += step_output_tokens

                if not assistant_msg.tool_calls:
                    # Check if agent is quitting without having made source edits
                    has_source_edit = any(
                        s.tool_name == "str_replace_editor"
                        and s.tool_args.get("command") in ("str_replace", "insert")
                        and not ToolExecutor._is_test_file(s.tool_args.get("path", "").replace("/testbed/", ""))
                        for s in steps
                    )
                    if not has_source_edit and step_idx < self._max_steps - 5:
                        messages.append({"role": "assistant", "content": assistant_msg.content or ""})
                        messages.append({"role": "user", "content":
                            "You have not made any edits to the source code yet. "
                            "You must fix the bug by editing the source files. "
                            "Based on what you've read, make your edit now."
                        })
                        continue
                    messages.append({"role": "assistant", "content": assistant_msg.content or ""})
                    break

                agent_thought = assistant_msg.content or ""

                # Handle tool calls
                all_tool_calls_serialized = []
                all_tool_results = []
                primary_tool_name = ""
                primary_tool_args = {}
                primary_served_obs = ""
                primary_raw_obs = ""
                primary_tier = "full"
                primary_raw_tokens = 0
                primary_served_tokens = 0

                for tc_idx, tool_call in enumerate(assistant_msg.tool_calls):
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    raw_observation = executor.execute(tool_name, tool_args)

                    # === TROT v3: Track test results for submit gate ===
                    if tool_name == "bash":
                        bash_cmd = tool_args.get("command", "")
                        if any(kw in bash_cmd for kw in ("pytest", "python -m pytest", "unittest", "python -m unittest")):
                            self._last_test_step = step_idx
                            if "passed" in raw_observation.lower() and "failed" not in raw_observation.lower() and "error" not in raw_observation.lower():
                                self._has_test_pass = True
                                self._last_test_passed = True
                            else:
                                self._last_test_passed = False

                    # === TROT v3 LAYER 1: Critical bypass ===
                    is_critical = (
                        False if self._disable_critical_bypass
                        else self._is_critical_observation(tool_name, tool_args, raw_observation)
                    )

                    # === TROT v3 LAYER 2: Post-edit protection ===
                    # Also protect observations after a submit was blocked (agent needs test output to debug)
                    in_post_submit_block_window = (step_idx - getattr(self, '_last_submit_block_step', -100)) <= 3
                    in_post_edit_window = (step_idx - last_edit_step_idx) <= self._post_edit_protection or in_post_submit_block_window

                    # Track edits
                    if tool_name == "str_replace_editor" and tool_args.get("command") in ("str_replace", "create", "insert"):
                        last_edit_step_idx = step_idx

                    # Decide compression path — gate-first, snorkel as fallback
                    if is_critical or in_post_edit_window:
                        # LAYER 1-2: Veto — serve full observation
                        served_obs = raw_observation
                        tier = "critical_bypass" if is_critical else "post_edit_bypass"
                        raw_tokens = count_tokens(raw_observation)
                        served_tokens = raw_tokens
                        if is_critical:
                            critical_bypass_count += 1
                        else:
                            post_edit_bypass_count += 1
                    elif self._gate and tool_name not in ("submit", "browse"):
                        # LAYER 3: HOP gate (XGBoost) decides first
                        served_obs, tier, raw_tokens, served_tokens = self._gate.process(
                            tool_name=tool_name,
                            args=tool_args,
                            raw_observation=raw_observation,
                            agent_thought=agent_thought,
                            task=problem_statement,
                        )
                        # LAYER 4: Apply snorkel rules only on gate's "full" decisions
                        # to catch obvious waste (empty output, passing tests, etc.)
                        if tier == "full":
                            snorkel_compressed = self._should_compress_observation(tool_name, tool_args, raw_observation)
                            if snorkel_compressed is not None:
                                served_obs = snorkel_compressed
                                tier = "snorkel_compress"
                                served_tokens = count_tokens(served_obs)
                    else:
                        served_obs = raw_observation
                        tier = "full" if tool_name != "browse" else "index"
                        raw_tokens = count_tokens(raw_observation)
                        served_tokens = raw_tokens

                    # Delivery-time compression (edit confirmations, test noise, etc.)
                    # Skip for critical observations and already-compressed observations
                    if not is_critical and tier != "snorkel_compress":
                        self._obs_compressor.track_tool_call(tool_name, tool_args)
                        served_obs = self._obs_compressor.compress(tool_name, tool_args, served_obs, thought=agent_thought)
                        served_tokens = count_tokens(served_obs)
                    else:
                        self._obs_compressor.track_tool_call(tool_name, tool_args)

                    total_obs_raw += raw_tokens
                    total_obs_served += served_tokens
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1

                    all_tool_calls_serialized.append({
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    })
                    all_tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": served_obs,
                    })

                    if tool_name == "submit":
                        # === TROT v3: Oracle submit gate ===
                        # 1. Check fail_to_pass tests pass
                        # 2. Sample check pass_to_pass for regressions
                        # Only allow unconditionally past 90% of budget.
                        budget_pct = step_idx / self._max_steps
                        force_allow = budget_pct >= 0.90

                        if self._disable_submit_gate or force_allow or not fail_to_pass:
                            submitted = True
                        else:
                            gate_result = executor.run_tests(
                                fail_to_pass=fail_to_pass, pass_to_pass=None, timeout=60
                            )
                            if not gate_result["resolved"]:
                                self._test_submit_blocks += 1
                                self._last_submit_block_step = step_idx
                                failing_tests = ", ".join(fail_to_pass[:3])
                                served_obs = (
                                    f"SUBMISSION BLOCKED: The target tests still FAIL.\n"
                                    f"Failing tests: {failing_tests}\n"
                                    f"Test output (last 500 chars): {gate_result.get('output', '')[-500:]}\n\n"
                                    f"Your fix is not correct yet. Review the test output above, "
                                    f"fix your code, and try again. You have {self._max_steps - step_idx} steps remaining."
                                )
                                all_tool_results[-1]["content"] = served_obs
                                served_tokens = count_tokens(served_obs)
                                total_obs_served += served_tokens - count_tokens(raw_observation)
                            elif pass_to_pass:
                                import random
                                p2p_sample = random.sample(pass_to_pass, min(50, len(pass_to_pass)))
                                p2p_result = executor.run_tests(
                                    fail_to_pass=p2p_sample, pass_to_pass=None, timeout=60
                                )
                                if p2p_result["resolved"]:
                                    submitted = True
                                else:
                                    self._test_submit_blocks += 1
                                    served_obs = (
                                        f"SUBMISSION BLOCKED: Your fix passes the target tests but BREAKS existing tests.\n"
                                        f"Regression test output (last 500 chars): {p2p_result.get('output', '')[-500:]}\n\n"
                                        f"Your fix introduced a regression. Review the failures above, "
                                        f"adjust your fix to not break existing behavior, and try again. "
                                        f"You have {self._max_steps - step_idx} steps remaining."
                                    )
                                    all_tool_results[-1]["content"] = served_obs
                                    served_tokens = count_tokens(served_obs)
                                    total_obs_served += served_tokens - count_tokens(raw_observation)
                            else:
                                submitted = True

                    if tc_idx == 0:
                        primary_tool_name = tool_name
                        primary_tool_args = tool_args
                        primary_served_obs = served_obs
                        primary_raw_obs = raw_observation
                        primary_tier = tier
                        primary_raw_tokens = raw_tokens
                        primary_served_tokens = served_tokens

                step = AgentStep(
                    step_idx=step_idx,
                    tool_name=primary_tool_name,
                    tool_args=primary_tool_args,
                    agent_thought=agent_thought,
                    raw_observation=primary_raw_obs,
                    served_observation=primary_served_obs,
                    raw_obs_tokens=primary_raw_tokens,
                    served_obs_tokens=primary_served_tokens,
                    tier_served=primary_tier,
                    input_tokens=step_input_tokens,
                    output_tokens=step_output_tokens,
                    latency_ms=step_latency,
                    cumulative_input_tokens=total_input_tokens,
                    cumulative_output_tokens=total_output_tokens,
                )
                steps.append(step)

                asst_dict = {
                    "role": "assistant",
                    "content": agent_thought,
                    "tool_calls": all_tool_calls_serialized,
                }
                asst_msg_idx = len(messages)
                messages.append(asst_dict)
                first_tool_msg_idx = len(messages)
                for tool_result in all_tool_results:
                    messages.append(tool_result)
                trajectory_steps.append((asst_dict, all_tool_results[0]))
                traj_msg_indices.append((asst_msg_idx, first_tool_msg_idx))

                if submitted:
                    break

                # Spin detection (kept from HOP — catches genuine loops)
                if self._spin_detector:
                    spin_signal = self._spin_detector.record_action(
                        step_idx=step_idx,
                        tool_name=primary_tool_name,
                        tool_args=primary_tool_args,
                        observation=primary_raw_obs,
                    )
                    if spin_signal == "FORCE_SUBMIT":
                        # Don't force-submit if tests still fail and budget remains
                        budget_pct = step_idx / self._max_steps
                        if budget_pct >= 0.90 or not fail_to_pass:
                            submitted = True
                            break
                        else:
                            gate_result = executor.run_tests(
                                fail_to_pass=fail_to_pass, pass_to_pass=None, timeout=60
                            )
                            if gate_result["resolved"]:
                                submitted = True
                                break
                            else:
                                # Reset spin detector — give agent another chance
                                self._spin_detector._nudges_given = 0
                                self._spin_detector._force_stop = False
                                messages.append({"role": "user", "content":
                                    "SYSTEM NOTE: You are stuck in a loop but your fix is NOT correct yet. "
                                    "The target tests still fail. Try a DIFFERENT approach — "
                                    "re-read the problem statement and the failing test, then attempt "
                                    "a fundamentally different fix."
                                })
                    elif spin_signal:
                        messages.append({"role": "user", "content": spin_signal})

                # === TROT v3: Progress Nudger (GUIDE only, never EXIT) ===
                nudge_msg = self._should_nudge(step_idx, steps)
                if nudge_msg:
                    messages.append({"role": "user", "content": nudge_msg})

                # === V3.1: Same-diff submit blocker ===
                if primary_tool_name == "submit":
                    diff_signature = primary_raw_obs[:500] if primary_raw_obs else ""
                    if diff_signature and diff_signature in self._submit_diffs:
                        if not self._same_diff_nudged:
                            self._same_diff_nudged = True
                            messages.append({"role": "user", "content":
                                "SYSTEM NOTE: You have submitted the SAME changes before and they were incorrect. "
                                "Do NOT submit the same diff again. You must take a fundamentally different approach: "
                                "re-read the failing test, understand what it actually expects, and write a different fix."
                            })
                    self._submit_diffs.append(diff_signature)

                # === V3.1: Consecutive identical bash output spin detection ===
                if primary_tool_name == "bash" and primary_raw_obs:
                    obs_sig = primary_raw_obs.strip()[:200]
                    if self._last_bash_outputs and obs_sig == self._last_bash_outputs[-1]:
                        self._last_bash_outputs.append(obs_sig)
                        if len(self._last_bash_outputs) >= 3 and self._bash_spin_nudges < 2:
                            self._bash_spin_nudges += 1
                            self._last_bash_outputs = []
                            messages.append({"role": "user", "content":
                                "SYSTEM NOTE: You are running the same command and getting identical output repeatedly. "
                                "This is not making progress. Change your approach: try a different command, "
                                "read a different file, or modify your fix."
                            })
                    else:
                        self._last_bash_outputs = [obs_sig]

                # === V3.1: Early exit on no-progress tasks ===
                if primary_tool_name == "str_replace_editor" and primary_tool_args.get("command") in ("str_replace", "create", "insert"):
                    self._last_source_edit_step = step_idx
                if step_idx >= 30 and self._last_source_edit_step < step_idx - 10:
                    # No source edit in last 10 steps past step 30 → force submit
                    if not submitted and step_idx % 5 == 0:
                        messages.append({"role": "user", "content":
                            "SYSTEM NOTE: You have not made any source edits in the last 10 steps. "
                            "Either submit your current changes now, or make a concrete edit. "
                            "Continuing to explore without editing is not making progress."
                        })

                # === TROT v3: Read-before-edit enforcement ===
                # If agent edited a file it only browsed (never read with view),
                # nudge it to read the actual code next time.
                if primary_tool_name == "str_replace_editor" and primary_tool_args.get("command") in ("str_replace", "create", "insert"):
                    edited_path = primary_tool_args.get("path", "")
                    if edited_path and edited_path not in self._files_fully_read:
                        self._browse_edit_nudges = getattr(self, "_browse_edit_nudges", 0) + 1
                        if self._browse_edit_nudges <= 2:
                            messages.append({"role": "user", "content":
                                "SYSTEM NOTE: You edited a file you've only browsed (seen the outline), not fully read. "
                                "Before your next edit, use the editor's view command with a specific view_range to read "
                                "the actual code around the area you're modifying. The outline may not show the exact "
                                "lines you need to change."
                            })
                # Track files fully read (view command without just browsing)
                if primary_tool_name == "str_replace_editor" and primary_tool_args.get("command") == "view":
                    viewed_path = primary_tool_args.get("path", "")
                    if viewed_path:
                        self._files_fully_read.add(viewed_path)

                # Targeted Haiku reflection (same as HOP — only for huge obs)
                s = step_idx + 1
                target_idx = s - self._a

                if target_idx > 0 and target_idx < len(trajectory_steps) and reflection_calls < self._max_reflection_calls:
                    if target_idx not in deterministically_compressed:
                        target_asst, target_tool = trajectory_steps[target_idx]
                        target_serialized = self._serialize_step(target_asst, target_tool)
                        l_orig = count_tokens(target_serialized)

                        if l_orig > self._theta:
                            ctx_start = max(0, target_idx - self._b)
                            ctx_end = min(len(trajectory_steps), s)
                            ctx_steps = trajectory_steps[ctx_start:ctx_end]
                            ctx_serialized = self._serialize_context(ctx_steps, target_idx - ctx_start)

                            ref_result = self._call_reflection(ctx_serialized)
                            if ref_result:
                                reduced_text, ref_in_tok, ref_out_tok = ref_result
                                reflection_input_tokens += ref_in_tok
                                reflection_output_tokens += ref_out_tok
                                reflection_calls += 1

                                l_reduced = count_tokens(reduced_text)

                                if l_orig - l_reduced > self._theta // 2:
                                    reduced_asst_content, reduced_tool_content = self._parse_reduced_step(
                                        reduced_text, target_asst, target_tool
                                    )
                                    new_asst = {**target_asst, "content": reduced_asst_content}
                                    new_tool = {**target_tool, "content": reduced_tool_content}
                                    trajectory_steps[target_idx] = (new_asst, new_tool)

                                    if target_idx < len(traj_msg_indices):
                                        a_idx, t_idx = traj_msg_indices[target_idx]
                                        if a_idx < len(messages):
                                            messages[a_idx] = new_asst
                                        if t_idx < len(messages):
                                            messages[t_idx] = new_tool

        except Exception as e:
            import traceback as _tb
            _tb_str = "".join(_tb.format_tb(e.__traceback__))
            error_msg = f"{type(e).__name__}: {str(e)[:200]}\n{_tb_str[-800:]}"

        duration = time.time() - start_time
        obs_savings = (1 - total_obs_served / max(total_obs_raw, 1)) * 100 if total_obs_raw > 0 else 0.0

        patch = executor.get_patch()
        test_results = executor.run_tests(
            fail_to_pass=fail_to_pass,
            pass_to_pass=pass_to_pass,
        )
        resolved = test_results["resolved"] and bool(patch)

        spin_metrics = {}
        if self._spin_detector:
            spin_metrics = self._spin_detector.get_metrics()

        compressor_metrics = self._compressor.get_metrics()
        dedup_metrics = self._deduplicator.get_metrics() if self._deduplicator else {}
        dedup_tokens_saved = dedup_metrics.get("dedup_tokens_saved", 0)

        return EvalResult(
            task_id=task_id,
            condition=condition,
            submitted=submitted,
            num_steps=len(steps),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_observation_tokens_raw=total_obs_raw,
            total_observation_tokens_served=total_obs_served,
            observation_savings_pct=round(obs_savings, 1),
            patch_produced=patch[:5000],
            resolved=resolved,
            tests_passed=test_results["num_passed"],
            tests_failed=test_results["num_failed"],
            test_exit_code=test_results["exit_code"],
            f2p_passed=test_results.get("f2p_passed", 0),
            f2p_total=test_results.get("f2p_total", 0),
            p2p_passed=test_results.get("p2p_passed", 0),
            p2p_total=test_results.get("p2p_total", 0),
            steps=steps,
            tier_distribution=tier_counts,
            error=error_msg,
            duration_seconds=round(duration, 1),
            reflection_input_tokens=reflection_input_tokens,
            reflection_output_tokens=reflection_output_tokens,
            reflection_calls=reflection_calls,
            history_tokens_saved=history_tokens_saved + compressor_metrics["history_tokens_saved"] + dedup_tokens_saved,
            history_compressions_applied=compressor_metrics.get("compressions_applied", 0),
            spin_events_detected=spin_metrics.get("spin_events_detected", 0),
            spin_nudges_given=spin_metrics.get("nudges_given", 0) + self._nudges_given_count,
            spin_force_stopped=spin_metrics.get("force_stopped", False),
        )

    def _serialize_step(self, asst: dict, tool: dict) -> str:
        tool_calls = asst.get("tool_calls", [])
        call_str = ""
        if tool_calls:
            func = tool_calls[0].get("function", {})
            call_str = f'<call tool="{func.get("name", "")}">{func.get("arguments", "")}</call>'
        return (
            f"<assistant>{asst.get('content', '')}\n{call_str}</assistant>\n"
            f"<tool_result>{tool.get('content', '')}</tool_result>"
        )

    def _serialize_context(self, ctx_steps: list[tuple[dict, dict]], target_offset: int) -> str:
        parts = []
        for i, (asst, tool) in enumerate(ctx_steps):
            marker = " [REDUCE THIS STEP]" if i == target_offset else ""
            step_content = self._serialize_step(asst, tool)
            parts.append(f'<step id="{i}"{marker}>\n{step_content}\n</step>')
        return "\n".join(parts)

    def _parse_reduced_step(self, reduced_text: str, orig_asst: dict, orig_tool: dict) -> tuple[str, str]:
        asst_match = re.search(r"<assistant>(.*?)</assistant>", reduced_text, re.DOTALL)
        tool_match = re.search(r"<tool_result>(.*?)</tool_result>", reduced_text, re.DOTALL)

        if asst_match and tool_match:
            reduced_asst = asst_match.group(1).strip()
            reduced_asst = re.sub(r"<call[^>]*>.*?</call>", "", reduced_asst, flags=re.DOTALL).strip()
            reduced_tool = tool_match.group(1).strip()
        elif tool_match:
            reduced_asst = orig_asst.get("content", "")
            reduced_tool = tool_match.group(1).strip()
        else:
            reduced_asst = orig_asst.get("content", "")
            reduced_tool = reduced_text

        return reduced_asst, reduced_tool

    def _call_reflection(self, context: str) -> Optional[tuple[str, int, int]]:
        try:
            response = self._client.chat.completions.create(
                model=self._reflect_model,
                messages=[
                    {"role": "system", "content": AGENTDIET_REFLECTION_PROMPT},
                    {"role": "user", "content": context},
                ],
                max_tokens=4096,
            )
            usage = response.usage
            in_tok = usage.prompt_tokens if usage else 0
            out_tok = usage.completion_tokens if usage else 0
            content = response.choices[0].message.content or ""
            return content, in_tok, out_tok
        except Exception:
            return None

    def _call_llm(self, messages: list[dict], retries: int = 30):
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS_WITH_BROWSE,
                    tool_choice="auto",
                    max_tokens=min(8192, getattr(self, '_max_output_tokens', 8192)),
                    parallel_tool_calls=False,
                )
                return response
            except Exception as e:
                err_str = str(e)
                if "maximum context length" in err_str or "max_model_len" in err_str or ("400" in err_str and "token" in err_str):
                    print(f"    [Context overflow] Compacting history...")
                    if hasattr(self, '_compact_messages'):
                        self._compact_messages(messages)
                    else:
                        LiveAgentRunner._compact_messages(self, messages)
                    continue
                if attempt < retries - 1:
                    is_conn_err = "Connection" in err_str or "ConnectError" in err_str or "refused" in err_str or "503" in err_str
                    wait = min(60, 2 ** (attempt + 1)) if not is_conn_err else min(120, 10 * (attempt + 1))
                    print(f"    API error (attempt {attempt + 1}/{retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


class SlidingWindowRunner:
    """Baseline: mask observations older than M steps with a fixed placeholder.

    Reimplements the core mechanism from "The Complexity Trap" (Lindenbauer et al., 2025).
    Observations older than M steps are replaced with a short placeholder in the
    conversation history before each API call. No learning, no per-call prediction.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 30,
        window_size: int = 8,
    ):
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._max_steps = max_steps
        self._window_size = window_size

    def run_task(
        self,
        task_id: str,
        problem_statement: str,
        testbed_path: str,
        condition: str = "sliding_window",
        fail_to_pass: Optional[list[str]] = None,
        pass_to_pass: Optional[list[str]] = None,
    ) -> EvalResult:
        """Run agent with sliding window observation masking."""
        start_time = time.time()
        steps: list[AgentStep] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_obs_raw = 0
        total_obs_served = 0
        submitted = False
        error_msg = ""

        executor = ToolExecutor(testbed_path)

        user_message = INSTANCE_TEMPLATE.format(
            working_dir="/testbed",
            problem_statement=problem_statement,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        # Track which message indices are tool results (for masking)
        tool_result_indices: list[int] = []

        try:
            for step_idx in range(self._max_steps):
                # Apply sliding window: mask tool results older than M steps
                self._apply_window(messages, tool_result_indices, step_idx)

                step_start = time.time()
                response = self._call_llm(messages)
                step_latency = (time.time() - step_start) * 1000

                if response is None:
                    error_msg = "LLM call returned None"
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                usage = response.usage
                step_input_tokens = usage.prompt_tokens if usage else 0
                step_output_tokens = usage.completion_tokens if usage else 0
                total_input_tokens += step_input_tokens
                total_output_tokens += step_output_tokens

                if not assistant_msg.tool_calls:
                    messages.append({"role": "assistant", "content": assistant_msg.content or ""})
                    break

                tool_call = assistant_msg.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                agent_thought = assistant_msg.content or ""

                raw_observation = executor.execute(tool_name, tool_args)
                raw_tokens = count_tokens(raw_observation)
                total_obs_raw += raw_tokens
                total_obs_served += raw_tokens  # served in full at delivery time

                step = AgentStep(
                    step_idx=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    agent_thought=agent_thought,
                    raw_observation=raw_observation,
                    served_observation=raw_observation,
                    raw_obs_tokens=raw_tokens,
                    served_obs_tokens=raw_tokens,
                    tier_served="full",
                    input_tokens=step_input_tokens,
                    output_tokens=step_output_tokens,
                    latency_ms=step_latency,
                    cumulative_input_tokens=total_input_tokens,
                    cumulative_output_tokens=total_output_tokens,
                )
                steps.append(step)

                if tool_name == "submit":
                    submitted = True

                messages.append(
                    {
                        "role": "assistant",
                        "content": agent_thought,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": raw_observation,
                    }
                )
                # Track this tool result's index and step
                tool_result_indices.append((len(messages) - 1, step_idx))

                if submitted:
                    break

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:300]}"

        duration = time.time() - start_time
        obs_savings = (1 - total_obs_served / max(total_obs_raw, 1)) * 100 if total_obs_raw > 0 else 0.0

        patch = executor.get_patch()
        test_results = executor.run_tests(fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass)
        resolved = test_results["resolved"] and bool(patch)

        return EvalResult(
            task_id=task_id,
            condition=condition,
            submitted=submitted,
            num_steps=len(steps),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_observation_tokens_raw=total_obs_raw,
            total_observation_tokens_served=total_obs_served,
            observation_savings_pct=round(obs_savings, 1),
            patch_produced=patch[:5000],
            resolved=resolved,
            tests_passed=test_results["num_passed"],
            tests_failed=test_results["num_failed"],
            test_exit_code=test_results["exit_code"],
            f2p_passed=test_results.get("f2p_passed", 0),
            f2p_total=test_results.get("f2p_total", 0),
            p2p_passed=test_results.get("p2p_passed", 0),
            p2p_total=test_results.get("p2p_total", 0),
            steps=steps,
            tier_distribution={"full": len(steps)},
            error=error_msg,
            duration_seconds=round(duration, 1),
        )

    def _apply_window(self, messages: list[dict], tool_result_indices: list, current_step: int):
        """Replace tool results older than window_size steps with a placeholder."""
        placeholder = "[Previous observation omitted for brevity.]"
        for msg_idx, step_idx in tool_result_indices:
            if current_step - step_idx > self._window_size:
                if messages[msg_idx]["content"] != placeholder:
                    messages[msg_idx]["content"] = placeholder

    def _call_llm(self, messages: list[dict], retries: int = 30):
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=8192,
                )
                return response
            except Exception as e:
                err_str = str(e)
                if "maximum context length" in err_str or "max_model_len" in err_str or ("400" in err_str and "token" in err_str):
                    print(f"    [Context overflow] Compacting history...")
                    LiveAgentRunner._compact_messages(self, messages)
                    continue
                if attempt < retries - 1:
                    is_conn_err = "Connection" in err_str or "ConnectError" in err_str or "refused" in err_str or "503" in err_str
                    wait = min(60, 2 ** (attempt + 1)) if not is_conn_err else min(120, 10 * (attempt + 1))
                    print(f"    API error (attempt {attempt + 1}/{retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


TACO_RULES = [
    # (pattern_type, regex_or_check, replacement_template)
    # Rule 1: pip install output -> keep only final status
    ("pip_install", r"(Collecting |Downloading |Installing |Using cached ).*", None),
    # Rule 2: pytest passing tests -> summary only
    ("pytest_pass", r"^(PASSED|tests/.*PASSED|.*::.*PASSED)$", None),
    # Rule 3: git diff headers repeated
    ("git_diff_header", r"^(diff --git|index [0-9a-f]+\.\.[0-9a-f]+|--- a/|\+\+\+ b/).*$", None),
    # Rule 4: make/build entering/leaving directory
    ("build_noise", r"^make(\[\d+\])?: (Entering|Leaving) directory.*$", None),
    # Rule 5: Python cache/bytecode mentions
    ("pycache", r".*(__pycache__|\.pyc|\.pyo|\.egg-info).*", None),
    # Rule 6: Empty lines in sequences (keep max 1)
    ("empty_lines", r"\n{3,}", "\n\n"),
]


class TACORunner:
    """Baseline: self-evolved compression rules applied at delivery time.

    Reimplements the core mechanism from TACO (Ren et al., 2026): rule-based
    compression of tool outputs using patterns discovered from trajectory data.
    Rules are static (pre-evolved), applied deterministically at zero LLM cost.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 30,
    ):
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._max_steps = max_steps
        self._rules = self._compile_rules()

    def _compile_rules(self):
        """Compile regex rules for efficient application."""
        compiled = []
        for rule_type, pattern, replacement in TACO_RULES:
            if pattern:
                compiled.append((rule_type, re.compile(pattern, re.MULTILINE), replacement))
        return compiled

    def _is_critical(self, observation: str) -> bool:
        """Check if observation contains critical signals (Eq. 1 from paper).

        Critical observations (errors, exceptions, syntax traces) pass through
        unchanged because they inform subsequent agent decisions.
        """
        critical_patterns = [
            "Traceback (most recent call last)",
            "SyntaxError:",
            "IndentationError:",
            "NameError:",
            "TypeError:",
            "ValueError:",
            "ImportError:",
            "ModuleNotFoundError:",
            "AttributeError:",
            "FileNotFoundError:",
            "PermissionError:",
            "OSError:",
            "RuntimeError:",
            "KeyError:",
            "IndexError:",
            "AssertionError:",
            "Exception:",
            "Error:",
            "FAILED",
            "error:",
            "fatal:",
        ]
        obs_lower = observation.lower()
        for pat in critical_patterns:
            if pat.lower() in obs_lower:
                return True
        return False

    def _apply_rules(self, observation: str, tool_name: str) -> str:
        """Apply TACO-style compression rules to an observation.

        Per Eq. 1: O_t passes through unchanged if Critical, otherwise
        compressed via rule-based filtering F_R_t(O_t | C_t).
        """
        # Critical observation bypass (paper Eq. 1)
        if self._is_critical(observation):
            return observation

        result = observation

        # Rule: pip install -> keep only "Successfully installed..." line
        if "Collecting " in result or "Installing collected" in result:
            lines = result.split("\n")
            kept = [l for l in lines if l.startswith("Successfully installed") or l.startswith("ERROR") or l.startswith("WARNING")]
            if kept:
                result = "\n".join(kept)
            else:
                result = "[pip install completed]"
            return result

        # Rule: pytest output -> keep failures + summary
        if "passed" in result and ("pytest" in result or "PASSED" in result or "FAILED" in result):
            lines = result.split("\n")
            kept = []
            in_failure = False
            for line in lines:
                if "FAILED" in line or "ERROR" in line or "ERRORS" in line:
                    in_failure = True
                    kept.append(line)
                elif in_failure and (line.startswith(" ") or line.startswith(">")):
                    kept.append(line)
                elif "passed" in line and ("failed" in line or "error" in line or line.strip().startswith("=")):
                    in_failure = False
                    kept.append(line)
                elif line.strip().startswith("=") and ("passed" in line or "failed" in line):
                    kept.append(line)
                elif "FAILED" not in result and line.strip().startswith("="):
                    kept.append(line)
            if kept:
                result = "\n".join(kept)
            return result

        # Rule: edit confirmation -> strip re-shown file content
        if "has been edited" in result and "Here's the result of running" in result:
            idx = result.find("Here's the result of running")
            if idx > 0:
                result = result[:idx].rstrip() + "\n[file content omitted after edit confirmation]"
            return result

        # Rule: submit output -> minimal acknowledgment
        if tool_name == "submit":
            return "[submit acknowledged]"

        # General rules: strip noise patterns
        for rule_type, pattern, replacement in self._rules:
            if replacement:
                result = pattern.sub(replacement, result)
            else:
                # Remove matching lines
                lines = result.split("\n")
                lines = [l for l in lines if not pattern.match(l)]
                result = "\n".join(lines)

        # Collapse excessive empty lines
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")

        return result

    def run_task(
        self,
        task_id: str,
        problem_statement: str,
        testbed_path: str,
        condition: str = "taco",
        fail_to_pass: Optional[list[str]] = None,
        pass_to_pass: Optional[list[str]] = None,
    ) -> EvalResult:
        """Run agent with TACO-style rule-based compression."""
        start_time = time.time()
        steps: list[AgentStep] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_obs_raw = 0
        total_obs_served = 0
        submitted = False
        error_msg = ""

        executor = ToolExecutor(testbed_path)

        user_message = INSTANCE_TEMPLATE.format(
            working_dir="/testbed",
            problem_statement=problem_statement,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            for step_idx in range(self._max_steps):
                step_start = time.time()
                response = self._call_llm(messages)
                step_latency = (time.time() - step_start) * 1000

                if response is None:
                    error_msg = "LLM call returned None"
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                usage = response.usage
                step_input_tokens = usage.prompt_tokens if usage else 0
                step_output_tokens = usage.completion_tokens if usage else 0
                total_input_tokens += step_input_tokens
                total_output_tokens += step_output_tokens

                if not assistant_msg.tool_calls:
                    messages.append({"role": "assistant", "content": assistant_msg.content or ""})
                    break

                tool_call = assistant_msg.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                agent_thought = assistant_msg.content or ""

                raw_observation = executor.execute(tool_name, tool_args)
                raw_tokens = count_tokens(raw_observation)

                # Apply TACO rules
                served_obs = self._apply_rules(raw_observation, tool_name)
                served_tokens = count_tokens(served_obs)

                total_obs_raw += raw_tokens
                total_obs_served += served_tokens

                tier = "compressed" if served_tokens < raw_tokens else "full"

                step = AgentStep(
                    step_idx=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    agent_thought=agent_thought,
                    raw_observation=raw_observation,
                    served_observation=served_obs,
                    raw_obs_tokens=raw_tokens,
                    served_obs_tokens=served_tokens,
                    tier_served=tier,
                    input_tokens=step_input_tokens,
                    output_tokens=step_output_tokens,
                    latency_ms=step_latency,
                    cumulative_input_tokens=total_input_tokens,
                    cumulative_output_tokens=total_output_tokens,
                )
                steps.append(step)

                if tool_name == "submit":
                    submitted = True

                messages.append(
                    {
                        "role": "assistant",
                        "content": agent_thought,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": served_obs,
                    }
                )

                if submitted:
                    break

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:300]}"

        duration = time.time() - start_time
        obs_savings = (1 - total_obs_served / max(total_obs_raw, 1)) * 100 if total_obs_raw > 0 else 0.0

        patch = executor.get_patch()
        test_results = executor.run_tests(fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass)
        resolved = test_results["resolved"] and bool(patch)

        tier_counts = {}
        for s in steps:
            tier_counts[s.tier_served] = tier_counts.get(s.tier_served, 0) + 1

        return EvalResult(
            task_id=task_id,
            condition=condition,
            submitted=submitted,
            num_steps=len(steps),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_observation_tokens_raw=total_obs_raw,
            total_observation_tokens_served=total_obs_served,
            observation_savings_pct=round(obs_savings, 1),
            patch_produced=patch[:5000],
            resolved=resolved,
            tests_passed=test_results["num_passed"],
            tests_failed=test_results["num_failed"],
            test_exit_code=test_results["exit_code"],
            f2p_passed=test_results.get("f2p_passed", 0),
            f2p_total=test_results.get("f2p_total", 0),
            p2p_passed=test_results.get("p2p_passed", 0),
            p2p_total=test_results.get("p2p_total", 0),
            steps=steps,
            tier_distribution=tier_counts,
            error=error_msg,
            duration_seconds=round(duration, 1),
        )

    def _call_llm(self, messages: list[dict], retries: int = 30):
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=8192,
                )
                return response
            except Exception as e:
                err_str = str(e)
                if "maximum context length" in err_str or "max_model_len" in err_str or ("400" in err_str and "token" in err_str):
                    print(f"    [Context overflow] Compacting history...")
                    LiveAgentRunner._compact_messages(self, messages)
                    continue
                if attempt < retries - 1:
                    is_conn_err = "Connection" in err_str or "ConnectError" in err_str or "refused" in err_str or "503" in err_str
                    wait = min(60, 2 ** (attempt + 1)) if not is_conn_err else min(120, 10 * (attempt + 1))
                    print(f"    API error (attempt {attempt + 1}/{retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise


class TruncationRunner:
    """Baseline: truncate all observations to a fixed character limit.

    The simplest possible compression: if an observation exceeds max_chars,
    keep only the first max_chars characters plus a truncation notice.
    No learning, no pattern matching, no LLM cost.
    """

    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "claude-sonnet-4-6",
        max_steps: int = 30,
        max_chars: int = 3000,
    ):
        self._client = OpenAI(api_key=api_key, base_url=api_base)
        self._model = model
        self._max_steps = max_steps
        self._max_chars = max_chars

    def _truncate(self, observation: str) -> str:
        if len(observation) <= self._max_chars:
            return observation
        return observation[: self._max_chars] + f"\n\n... [truncated at {self._max_chars} chars, {len(observation) - self._max_chars} chars omitted]"

    def run_task(
        self,
        task_id: str,
        problem_statement: str,
        testbed_path: str,
        condition: str = "truncation",
        fail_to_pass: Optional[list[str]] = None,
        pass_to_pass: Optional[list[str]] = None,
    ) -> EvalResult:
        """Run agent with fixed-length truncation."""
        start_time = time.time()
        steps: list[AgentStep] = []
        total_input_tokens = 0
        total_output_tokens = 0
        total_obs_raw = 0
        total_obs_served = 0
        submitted = False
        error_msg = ""

        executor = ToolExecutor(testbed_path)

        user_message = INSTANCE_TEMPLATE.format(
            working_dir="/testbed",
            problem_statement=problem_statement,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            for step_idx in range(self._max_steps):
                step_start = time.time()
                response = self._call_llm(messages)
                step_latency = (time.time() - step_start) * 1000

                if response is None:
                    error_msg = "LLM call returned None"
                    break

                choice = response.choices[0]
                assistant_msg = choice.message

                usage = response.usage
                step_input_tokens = usage.prompt_tokens if usage else 0
                step_output_tokens = usage.completion_tokens if usage else 0
                total_input_tokens += step_input_tokens
                total_output_tokens += step_output_tokens

                if not assistant_msg.tool_calls:
                    messages.append({"role": "assistant", "content": assistant_msg.content or ""})
                    break

                tool_call = assistant_msg.tool_calls[0]
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    tool_args = {}

                agent_thought = assistant_msg.content or ""

                raw_observation = executor.execute(tool_name, tool_args)
                raw_tokens = count_tokens(raw_observation)

                served_obs = self._truncate(raw_observation)
                served_tokens = count_tokens(served_obs)

                total_obs_raw += raw_tokens
                total_obs_served += served_tokens

                tier = "truncated" if served_tokens < raw_tokens else "full"

                step = AgentStep(
                    step_idx=step_idx,
                    tool_name=tool_name,
                    tool_args=tool_args,
                    agent_thought=agent_thought,
                    raw_observation=raw_observation,
                    served_observation=served_obs,
                    raw_obs_tokens=raw_tokens,
                    served_obs_tokens=served_tokens,
                    tier_served=tier,
                    input_tokens=step_input_tokens,
                    output_tokens=step_output_tokens,
                    latency_ms=step_latency,
                    cumulative_input_tokens=total_input_tokens,
                    cumulative_output_tokens=total_output_tokens,
                )
                steps.append(step)

                if tool_name == "submit":
                    submitted = True

                messages.append(
                    {
                        "role": "assistant",
                        "content": agent_thought,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(tool_args),
                                },
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": served_obs,
                    }
                )

                if submitted:
                    break

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)[:300]}"

        duration = time.time() - start_time
        obs_savings = (1 - total_obs_served / max(total_obs_raw, 1)) * 100 if total_obs_raw > 0 else 0.0

        patch = executor.get_patch()
        test_results = executor.run_tests(fail_to_pass=fail_to_pass, pass_to_pass=pass_to_pass)
        resolved = test_results["resolved"] and bool(patch)

        tier_counts = {}
        for s in steps:
            tier_counts[s.tier_served] = tier_counts.get(s.tier_served, 0) + 1

        return EvalResult(
            task_id=task_id,
            condition=condition,
            submitted=submitted,
            num_steps=len(steps),
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
            total_observation_tokens_raw=total_obs_raw,
            total_observation_tokens_served=total_obs_served,
            observation_savings_pct=round(obs_savings, 1),
            patch_produced=patch[:5000],
            resolved=resolved,
            tests_passed=test_results["num_passed"],
            tests_failed=test_results["num_failed"],
            test_exit_code=test_results["exit_code"],
            f2p_passed=test_results.get("f2p_passed", 0),
            f2p_total=test_results.get("f2p_total", 0),
            p2p_passed=test_results.get("p2p_passed", 0),
            p2p_total=test_results.get("p2p_total", 0),
            steps=steps,
            tier_distribution=tier_counts,
            error=error_msg,
            duration_seconds=round(duration, 1),
        )

    def _call_llm(self, messages: list[dict], retries: int = 30):
        for attempt in range(retries):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    max_tokens=8192,
                )
                return response
            except Exception as e:
                err_str = str(e)
                if "maximum context length" in err_str or "max_model_len" in err_str or ("400" in err_str and "token" in err_str):
                    print(f"    [Context overflow] Compacting history...")
                    LiveAgentRunner._compact_messages(self, messages)
                    continue
                if attempt < retries - 1:
                    is_conn_err = "Connection" in err_str or "ConnectError" in err_str or "refused" in err_str or "503" in err_str
                    wait = min(60, 2 ** (attempt + 1)) if not is_conn_err else min(120, 10 * (attempt + 1))
                    print(f"    API error (attempt {attempt + 1}/{retries}): {e}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise

