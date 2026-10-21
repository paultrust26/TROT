"""SWE-Agent Integration Wrapper — hooks HOP into real agent loops.

Provides three integration modes:
  1. SWEAgentHOPWrapper — wraps SWE-agent's tool execution
  2. OpenHandsHOPWrapper — wraps OpenHands/OpenDevin tool execution
  3. GenericAgentHOPWrapper — wraps any agent with a tool-call interface

The wrapper intercepts tool outputs between execution and context injection,
applying the observation allocation policy.

Also provides:
  - TrajectoryRecorder — records full trajectories for offline analysis
  - LiveMetrics — real-time monitoring of savings and under-service
  - ConfigurablePolicy — runtime-adjustable allocation parameters

Usage:
    # As SWE-agent wrapper
    wrapper = SWEAgentHOPWrapper(
        models_dir="./models",
        risk_threshold=0.15,
        record_trajectory=True,
    )

    # In agent loop (SWE-agent style):
    for action in agent.get_actions():
        raw_observation = environment.execute(action)
        processed = wrapper.process_observation(
            tool=action.tool_name,
            args=action.tool_args,
            raw_output=raw_observation,
            agent_thought=action.thought,
            task=task_description,
        )
        agent.receive_observation(processed.content)

    # After task completes:
    metrics = wrapper.get_metrics()
    trajectory = wrapper.get_trajectory()
"""

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .inference_gate import (
    ObservationAllocator,
    AgentObservationInterceptor,
    AllocationDecision,
    TrajectoryState,
    TIERS,
    TIER_COSTS,
)
from .observation_store import ObservationStore, ToolType, ObservationMiddleware
from .observation_actions import estimate_tokens, return_diff, ReducedObservation


# =============================================================================
# SessionRedundancyTracker — detects redundant/expired observations
# Inspired by AgentDiet's waste taxonomy (Xiao et al. 2026), adapted for
# preemptive gating (zero-cost, no LLM reflection step needed).
# =============================================================================


@dataclass
class RedundancySignal:
    """Signals produced by the redundancy tracker for a single observation."""

    is_duplicate_content: bool = False
    is_unchanged_reread: bool = False
    file_access_count: int = 0
    steps_since_last_view: int = 0
    content_overlap_ratio: float = 0.0
    is_superseded: bool = False
    context_tokens_so_far: int = 0
    observation_tokens: int = 0
    is_below_threshold: bool = False

    @property
    def should_hard_block(self) -> bool:
        """Hard-rule: skip gating entirely and block/summarize."""
        return self.is_duplicate_content or (self.is_unchanged_reread and self.steps_since_last_view < 8)

    @property
    def redundancy_score(self) -> float:
        """0-1 score: how redundant this observation is. Higher = more redundant."""
        score = 0.0
        if self.is_unchanged_reread:
            score += 0.5
        if self.file_access_count > 2:
            score += 0.2
        if self.content_overlap_ratio > 0.8:
            score += 0.3
        if self.is_superseded:
            score += 0.4
        return min(score, 1.0)

    def as_features(self) -> dict[str, float]:
        """Extra features to feed into XGBoost alongside existing features."""
        return {
            "redundancy_is_duplicate": float(self.is_duplicate_content),
            "redundancy_is_unchanged_reread": float(self.is_unchanged_reread),
            "redundancy_file_access_count": float(self.file_access_count),
            "redundancy_steps_since_last_view": float(min(self.steps_since_last_view, 30) / 30.0),
            "redundancy_content_overlap": self.content_overlap_ratio,
            "redundancy_is_superseded": float(self.is_superseded),
            "redundancy_context_budget_used": float(self.context_tokens_so_far / 100_000.0),
            "redundancy_obs_token_ratio": float(min(self.observation_tokens, 10000) / 10000.0),
            "redundancy_is_below_threshold": float(self.is_below_threshold),
            "redundancy_score": self.redundancy_score,
        }


class SessionRedundancyTracker:
    """Tracks session state to detect redundant and expired observations.

    Three waste types (mapped from AgentDiet taxonomy to preemptive signals):

    1. REDUNDANT — same content seen before in this session
       - File re-read with identical content hash
       - str_replace_editor view of same file within N steps
       - Bash command producing same output as a prior call

    2. EXPIRED (proxy) — content likely no longer relevant
       - File viewed many steps ago without being referenced in agent thought
       - Exploration path abandoned (agent moved to different file/area)
       - Test output from before the last edit (superseded)

    3. USELESS (already handled by tier model, but we add the θ threshold)
       - Short observations below token threshold (AgentDiet's θ=500)
       - These cost more to gate than they save
    """

    TOKEN_THRESHOLD = 500  # AgentDiet's θ: observations below this aren't worth reducing

    def __init__(self):
        self._step: int = 0
        self._context_tokens: int = 0  # running total of tokens served into context

        # Redundancy tracking
        self._content_hashes: dict[str, str] = {}  # hash(content) → step_seen
        self._file_view_history: dict[str, list[int]] = {}  # file_path → [step1, step2, ...]
        self._file_content_at_step: dict[str, str] = {}  # file_path → hash of last content
        self._file_prior_content: dict[str, str] = {}  # file_path → actual content (for diff delivery)
        self._bash_output_hashes: dict[str, int] = {}  # hash(cmd+output) → step

        # Expiration tracking
        self._last_edit_step: int = 0
        self._last_test_step: int = 0
        self._files_referenced_in_thought: set[str] = set()
        self._agent_focus_files: list[str] = []  # recent files the agent is working on

    def update_step(self, step: int):
        self._step = step

    def track_thought(self, thought: str, known_files: list[str]):
        """Track which files the agent references in its reasoning."""
        for f in known_files:
            basename = Path(f).name
            stem = Path(f).stem
            if basename in thought or stem in thought:
                self._files_referenced_in_thought.add(f)
                if f not in self._agent_focus_files:
                    self._agent_focus_files.append(f)
                    # Keep only last 5 focus files
                    if len(self._agent_focus_files) > 5:
                        self._agent_focus_files.pop(0)

    def track_edit(self, file_path: str):
        """Called when agent edits a file."""
        self._last_edit_step = self._step
        if file_path and file_path not in self._agent_focus_files:
            self._agent_focus_files.append(file_path)
        # Track that this file was edited since last read (for diff tier)
        if not hasattr(self, "_files_edited_since_read"):
            self._files_edited_since_read = set()
        if file_path:
            self._files_edited_since_read.add(file_path)

    def track_test(self):
        """Called when agent runs tests."""
        self._last_test_step = self._step

    def add_served_tokens(self, tokens: int):
        """Track cumulative tokens served into context."""
        self._context_tokens += tokens

    def check(
        self,
        tool_type: str,
        file_path: str,
        command: str,
        observation: str,
    ) -> RedundancySignal:
        """Check an observation for redundancy/expiration signals.

        Call this BEFORE the gate model predicts a tier. If the signal says
        should_hard_block, skip the model entirely and serve a one-liner.
        Otherwise, pass signal.as_features() as extra features to the model.
        """
        import hashlib as _hashlib

        obs_tokens = estimate_tokens(observation)
        content_hash = _hashlib.sha256(observation.encode()).hexdigest()[:16]

        signal = RedundancySignal(
            observation_tokens=obs_tokens,
            context_tokens_so_far=self._context_tokens,
            is_below_threshold=(obs_tokens < self.TOKEN_THRESHOLD),
        )

        # --- Redundancy: exact content duplication ---
        if content_hash in self._content_hashes:
            signal.is_duplicate_content = True
            signal.steps_since_last_view = self._step - int(self._content_hashes[content_hash])
        self._content_hashes[content_hash] = str(self._step)

        # --- Redundancy: file re-read with unchanged content ---
        if file_path and tool_type == "read":
            prev_hash = self._file_content_at_step.get(file_path)
            if prev_hash == content_hash:
                signal.is_unchanged_reread = True

            # Access count and recency
            history = self._file_view_history.get(file_path, [])
            signal.file_access_count = len(history) + 1
            if history:
                signal.steps_since_last_view = self._step - history[-1]
            else:
                signal.steps_since_last_view = self._step  # first access

            # Update history
            history.append(self._step)
            self._file_view_history[file_path] = history
            self._file_content_at_step[file_path] = content_hash
            # Store actual content for diff-aware delivery (cap at 50KB to bound memory)
            if len(observation) < 50_000:
                self._file_prior_content[file_path] = observation

        # --- Redundancy: bash command producing same output ---
        if tool_type == "bash" and command:
            cmd_output_key = _hashlib.md5(f"{command}:{content_hash}".encode()).hexdigest()[:12]
            if cmd_output_key in self._bash_output_hashes:
                signal.is_duplicate_content = True
                signal.steps_since_last_view = self._step - self._bash_output_hashes[cmd_output_key]
            self._bash_output_hashes[cmd_output_key] = self._step

        # --- Expiration: test output superseded by later edit ---
        if tool_type == "test" and self._last_edit_step > self._last_test_step:
            # There was an edit since last test — prior test results are stale
            # (but this IS the new test, so it's NOT superseded)
            pass
        elif tool_type == "read" and file_path:
            # Check if this file was already explored and agent moved on
            if file_path not in self._agent_focus_files and signal.file_access_count == 1 and self._step > 5:
                # First view of a file that's not in agent's focus area, late in session
                # Likely exploration — more aggressive gating is safe
                signal.is_superseded = False  # can't know yet, but redundancy_score accounts for this

        # --- Content overlap with prior observation of same file ---
        if file_path and tool_type == "read" and signal.file_access_count > 1:
            # Estimate overlap: if unchanged, overlap is 100%
            if signal.is_unchanged_reread:
                signal.content_overlap_ratio = 1.0
            else:
                # Partial overlap (file was edited between views) — estimate conservatively
                signal.content_overlap_ratio = 0.3

        return signal

    def format_redundancy_block(self, file_path: str, signal: RedundancySignal) -> str:
        """Generate an informative replacement message for redundant observations."""
        if signal.is_unchanged_reread:
            step = signal.steps_since_last_view
            return (
                f"[File unchanged — you read {file_path} {step} steps ago and it has not been modified since. "
                f"The content is already in your conversation context above. "
                f"Proceed with your edit or next action based on what you already know.]"
            )
        if signal.is_duplicate_content:
            return (
                f"[Duplicate output — this exact content appeared {signal.steps_since_last_view} steps ago in this conversation. "
                f"It is already in your context. Proceed with your planned action rather than re-reading.]"
            )
        return (
            f"[Content compressed — this observation was identified as redundant with information you already have. "
            f"If you need specific details, use a targeted view with line ranges or search for specific symbols.]"
        )

    def get_prior_content(self, file_path: str) -> Optional[str]:
        """Get previously stored content for a file (for diff delivery)."""
        return self._file_prior_content.get(file_path)

    def has_prior_observation(self, file_path: str) -> bool:
        """Check if we have a prior observation for this file."""
        return file_path in self._file_content_at_step

    def content_changed(self, file_path: str, current_content: str) -> bool:
        """Check if file content differs from last observation."""
        import hashlib as _hashlib

        current_hash = _hashlib.sha256(current_content.encode()).hexdigest()[:16]
        prev_hash = self._file_content_at_step.get(file_path)
        return prev_hash is not None and prev_hash != current_hash

    def reset(self):
        """Reset between tasks."""
        self._step = 0
        self._context_tokens = 0
        self._content_hashes = {}
        self._file_view_history = {}
        self._file_content_at_step = {}
        self._file_prior_content = {}
        self._bash_output_hashes = {}
        self._last_edit_step = 0
        self._last_test_step = 0
        self._files_referenced_in_thought = set()
        self._agent_focus_files = []


# =============================================================================
# observe_more tool definition (added to agent's available tools)
# =============================================================================

OBSERVE_MORE_TOOL = {
    "name": "observe_more",
    "description": (
        "Request higher-fidelity view of a previously reduced observation. "
        "Use when you need more detail than what was provided."
    ),
    "parameters": {
        "obs_id": {
            "type": "string",
            "description": "The observation ID shown in brackets after a reduced observation",
        },
        "level": {
            "type": "string",
            "enum": ["full", "outline", "relevant_spans"],
            "default": "full",
        },
    },
}


# =============================================================================
# ProcessedObservation — unified return type for all wrappers
# =============================================================================


@dataclass
class ProcessedObservation:
    """Unified return type from all HOP wrappers."""

    content: str
    obs_id: str
    tool: str
    tool_type: str
    tier_served: str
    raw_tokens: int
    served_tokens: int
    compression_ratio: float
    escalation_available: bool
    allocation_decision: Optional[AllocationDecision] = None


# =============================================================================
# TrajectoryRecorder — records everything for offline analysis
# =============================================================================


@dataclass
class RecordedStep:
    """A single recorded step in the trajectory."""

    step_idx: int
    tool: str
    tool_args: dict
    raw_output: str
    raw_tokens: int
    served_tier: str
    served_content: str
    served_tokens: int
    allocation_decision: dict
    trajectory_state: dict
    agent_thought: str
    timestamp: float


class TrajectoryRecorder:
    """Records full trajectories for offline analysis and replay."""

    def __init__(self):
        self._steps: list[RecordedStep] = []
        self._task_id: str = ""
        self._task_description: str = ""
        self._start_time: float = time.time()

    def set_task(self, task_id: str, description: str):
        """Set the current task metadata."""
        self._task_id = task_id
        self._task_description = description

    def record(self, step: RecordedStep):
        """Record a single step."""
        self._steps.append(step)

    def save(self, path: Path):
        """Save the trajectory as JSONL for offline analysis."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            # Header line with metadata
            header = {
                "_type": "trajectory_header",
                "task_id": self._task_id,
                "task_description": self._task_description,
                "start_time": self._start_time,
                "total_steps": len(self._steps),
            }
            f.write(json.dumps(header) + "\n")

            # One line per step
            for step in self._steps:
                record = {
                    "_type": "step",
                    "step_idx": step.step_idx,
                    "tool": step.tool,
                    "tool_args": step.tool_args,
                    "raw_output": step.raw_output[:5000],  # Truncate for storage
                    "raw_tokens": step.raw_tokens,
                    "served_tier": step.served_tier,
                    "served_content": step.served_content[:2000],
                    "served_tokens": step.served_tokens,
                    "allocation_decision": step.allocation_decision,
                    "trajectory_state": step.trajectory_state,
                    "agent_thought": step.agent_thought[:500],
                    "timestamp": step.timestamp,
                }
                f.write(json.dumps(record) + "\n")

    def compute_offline_metrics(self) -> dict:
        """Analyze the recorded trajectory and compute aggregate metrics."""
        if not self._steps:
            return {"steps": 0}

        total_raw = sum(s.raw_tokens for s in self._steps)
        total_served = sum(s.served_tokens for s in self._steps)

        # Tier distribution
        tier_counts: dict[str, int] = {}
        for s in self._steps:
            tier_counts[s.served_tier] = tier_counts.get(s.served_tier, 0) + 1

        # Escalation tracking (observe_more calls)
        escalation_count = sum(1 for s in self._steps if s.tool == "observe_more")

        # Per-tool-type savings
        tool_savings: dict[str, dict] = {}
        for s in self._steps:
            tool_type = s.allocation_decision.get("tool_type", "unknown")
            if tool_type not in tool_savings:
                tool_savings[tool_type] = {"raw": 0, "served": 0, "count": 0}
            tool_savings[tool_type]["raw"] += s.raw_tokens
            tool_savings[tool_type]["served"] += s.served_tokens
            tool_savings[tool_type]["count"] += 1

        per_tool_compression = {}
        for tt, stats in tool_savings.items():
            if stats["raw"] > 0:
                per_tool_compression[tt] = {
                    "savings_pct": round((1 - stats["served"] / stats["raw"]) * 100, 1),
                    "count": stats["count"],
                }

        # Under-service estimate: steps followed by escalation or re-call of same tool
        potential_under_service = 0
        for i, s in enumerate(self._steps[:-1]):
            next_step = self._steps[i + 1]
            # Re-call same tool = possible under-service
            if next_step.tool == s.tool and next_step.tool_args == s.tool_args and s.served_tier != "full":
                potential_under_service += 1
            # Escalation right after = definite under-service
            if next_step.tool == "observe_more":
                potential_under_service += 1

        non_full_steps = sum(1 for s in self._steps if s.served_tier != "full")

        return {
            "steps": len(self._steps),
            "total_raw_tokens": total_raw,
            "total_served_tokens": total_served,
            "overall_savings_pct": round((1 - total_served / max(total_raw, 1)) * 100, 1),
            "tier_distribution": tier_counts,
            "escalation_count": escalation_count,
            "per_tool_compression": per_tool_compression,
            "potential_under_service_count": potential_under_service,
            "estimated_under_service_rate": round(potential_under_service / max(non_full_steps, 1), 3),
            "task_id": self._task_id,
            "duration_seconds": round(time.time() - self._start_time, 1),
        }

    def get_steps(self) -> list[RecordedStep]:
        """Return all recorded steps."""
        return self._steps.copy()

    def reset(self):
        """Reset the recorder for a new task."""
        self._steps = []
        self._task_id = ""
        self._task_description = ""
        self._start_time = time.time()


# =============================================================================
# LiveMetrics — real-time monitoring of savings and under-service
# =============================================================================


class LiveMetrics:
    """Real-time monitoring of HOP savings and under-service indicators."""

    def __init__(self):
        self._total_raw_tokens: int = 0
        self._total_served_tokens: int = 0
        self._steps_processed: int = 0
        self._tier_counts: dict[str, int] = {}
        self._escalation_count: int = 0
        self._tool_type_raw: dict[str, int] = {}
        self._tool_type_served: dict[str, int] = {}
        self._potential_under_service: int = 0
        self._last_step_tool: str = ""
        self._last_step_args: dict = {}
        self._last_step_tier: str = ""

    def update(self, step: RecordedStep):
        """Update metrics with a new step."""
        self._total_raw_tokens += step.raw_tokens
        self._total_served_tokens += step.served_tokens
        self._steps_processed += 1
        self._tier_counts[step.served_tier] = self._tier_counts.get(step.served_tier, 0) + 1

        tool_type = step.allocation_decision.get("tool_type", "unknown")
        self._tool_type_raw[tool_type] = self._tool_type_raw.get(tool_type, 0) + step.raw_tokens
        self._tool_type_served[tool_type] = self._tool_type_served.get(tool_type, 0) + step.served_tokens

        # Track escalations
        if step.tool == "observe_more":
            self._escalation_count += 1

        # Estimate under-service: re-call same tool+args after reduction
        if (
            step.tool == self._last_step_tool
            and step.tool_args == self._last_step_args
            and self._last_step_tier != "full"
        ):
            self._potential_under_service += 1

        self._last_step_tool = step.tool
        self._last_step_args = step.tool_args
        self._last_step_tier = step.served_tier

    def get_summary(self) -> dict:
        """Return comprehensive metrics summary."""
        savings_pct = (
            round((1 - self._total_served_tokens / max(self._total_raw_tokens, 1)) * 100, 1)
            if self._total_raw_tokens > 0
            else 0.0
        )

        non_full = sum(v for k, v in self._tier_counts.items() if k != "full")
        escalation_rate = round(self._escalation_count / max(non_full, 1), 3) if non_full > 0 else 0.0

        # Per-tool-type average compression
        avg_compression_per_tool: dict[str, float] = {}
        for tt in self._tool_type_raw:
            raw = self._tool_type_raw[tt]
            served = self._tool_type_served.get(tt, raw)
            if raw > 0:
                avg_compression_per_tool[tt] = round(1 - served / raw, 3)

        estimated_under_service_rate = round(self._potential_under_service / max(non_full, 1), 3)

        return {
            "total_raw_tokens": self._total_raw_tokens,
            "total_served_tokens": self._total_served_tokens,
            "savings_pct": savings_pct,
            "tier_distribution": dict(self._tier_counts),
            "escalation_rate": escalation_rate,
            "estimated_under_service_rate": estimated_under_service_rate,
            "steps_processed": self._steps_processed,
            "avg_compression_per_tool_type": avg_compression_per_tool,
        }

    def print_live(self):
        """Print single-line live status to stderr."""
        summary = self.get_summary()
        line = (
            f"[HOP] steps={summary['steps_processed']} "
            f"savings={summary['savings_pct']:.1f}% "
            f"escalations={self._escalation_count} "
            f"under_service~{summary['estimated_under_service_rate']:.1%} "
            f"tokens={self._total_served_tokens}/{self._total_raw_tokens}"
        )
        print(f"\r{line}", end="", file=sys.stderr, flush=True)

    def reset(self):
        """Reset all metrics."""
        self._total_raw_tokens = 0
        self._total_served_tokens = 0
        self._steps_processed = 0
        self._tier_counts = {}
        self._escalation_count = 0
        self._tool_type_raw = {}
        self._tool_type_served = {}
        self._potential_under_service = 0
        self._last_step_tool = ""
        self._last_step_args = {}
        self._last_step_tier = ""


# =============================================================================
# ConfigurablePolicy — runtime-adjustable allocation parameters
# =============================================================================


class ConfigurablePolicy:
    """Runtime-adjustable observation allocation parameters.

    Allows dynamic tuning of the allocation policy without restarting
    the agent or reloading models.
    """

    def __init__(
        self,
        base_rho: float = 0.15,
        conservative_mode: bool = False,
        tool_overrides: Optional[dict[str, str]] = None,
    ):
        self._base_rho = base_rho
        self._conservative_mode = conservative_mode
        self._tool_overrides: dict[str, str] = tool_overrides or {}

    @property
    def base_rho(self) -> float:
        return self._base_rho

    @property
    def conservative_mode(self) -> bool:
        return self._conservative_mode

    def set_rho(self, rho: float):
        """Set the base risk threshold. Higher = more aggressive compression."""
        if not 0.0 < rho < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        self._base_rho = rho

    def set_tool_override(self, tool_type: str, fixed_tier: str):
        """Force a specific tier for a tool type (useful for debugging).

        Args:
            tool_type: One of 'read', 'bash', 'test', 'search'.
            fixed_tier: The tier to always use for this tool type.
        """
        if tool_type not in TIERS:
            raise ValueError(f"Unknown tool_type '{tool_type}'. Expected one of {list(TIERS.keys())}")
        valid_tiers = TIERS[tool_type] + ["full"]
        if fixed_tier not in valid_tiers:
            raise ValueError(f"Invalid tier '{fixed_tier}' for {tool_type}. Expected one of {valid_tiers}")
        self._tool_overrides[tool_type] = fixed_tier

    def remove_tool_override(self, tool_type: str):
        """Remove a tool override."""
        self._tool_overrides.pop(tool_type, None)

    def set_conservative_mode(self, enabled: bool):
        """Enable/disable conservative mode (all full for safety)."""
        self._conservative_mode = enabled

    def get_override_for(self, tool_type: str) -> Optional[str]:
        """Get the override tier for a tool type, if any."""
        if self._conservative_mode:
            return "full"
        return self._tool_overrides.get(tool_type)

    def get_config(self) -> dict:
        """Return current policy configuration."""
        return {
            "base_rho": self._base_rho,
            "conservative_mode": self._conservative_mode,
            "tool_overrides": dict(self._tool_overrides),
        }


# =============================================================================
# SWEAgentHOPWrapper — main integration class for SWE-agent
# =============================================================================


class SWEAgentHOPWrapper:
    """HOP integration wrapper for SWE-agent.

    Intercepts tool outputs between execution and context injection,
    applies the observation allocation policy, records trajectories,
    and provides real-time metrics.

    SWE-agent tool mapping:
      - open / goto / scroll_down / scroll_up -> read
      - edit / submit -> (pass-through, no gating)
      - find_file / search_dir / search_file -> search
      - bash command / create -> bash (or test if test command)
    """

    # SWE-agent tool name classification
    _SWE_READ_TOOLS = {"open", "goto", "scroll_down", "scroll_up", "view"}
    _SWE_SEARCH_TOOLS = {"find_file", "search_dir", "search_file", "find", "grep"}
    _SWE_BASH_TOOLS = {"bash", "terminal", "execute"}
    _SWE_EDIT_TOOLS = {"edit", "submit", "insert", "replace"}
    _SWE_TEST_KEYWORDS = ("pytest", "test", "unittest", "npm test", "cargo test", "make test")

    def __init__(
        self,
        models_dir: Optional[str] = None,
        risk_threshold: float = 0.15,
        record_trajectory: bool = True,
        live_metrics: bool = True,
        conservative_mode: bool = False,
        tool_overrides: Optional[dict[str, str]] = None,
        warmup_steps: int = 3,
        first_access_full: bool = True,
    ):
        # Core allocator
        models_path = Path(models_dir) if models_dir else None
        self._allocator = ObservationAllocator(
            models_dir=models_path,
            base_rho=risk_threshold,
            fallback_to_heuristic=True,
        )
        self._interceptor = AgentObservationInterceptor(self._allocator)

        # Observation store for escalation support
        self._store = ObservationStore()

        # Policy control
        self._policy = ConfigurablePolicy(
            base_rho=risk_threshold,
            conservative_mode=conservative_mode,
            tool_overrides=tool_overrides,
        )

        # Redundancy tracker (AgentDiet-inspired waste detection)
        self._redundancy = SessionRedundancyTracker()

        # Recording
        self._recorder = TrajectoryRecorder() if record_trajectory else None
        self._live_metrics = LiveMetrics() if live_metrics else None

        # State
        self._step_idx: int = 0
        self._task_description: str = ""

        # Safety: warmup + first-access protection
        self._warmup_steps = warmup_steps
        self._first_access_full = first_access_full
        self._files_accessed: set[str] = set()

    def process_observation(
        self,
        tool: str,
        args: dict,
        raw_output: str,
        agent_thought: str = "",
        task: str = "",
    ) -> ProcessedObservation:
        """Process a tool observation through the HOP allocation pipeline.

        Args:
            tool: The tool name (SWE-agent tool names like 'open', 'bash', etc.)
            args: Tool arguments dict.
            raw_output: The full raw output from tool execution.
            agent_thought: The agent's reasoning before this tool call.
            task: Task description / problem statement.

        Returns:
            ProcessedObservation with the allocated content.
        """
        if task and not self._task_description:
            self._task_description = task
            self._interceptor.problem_statement = task
            if self._recorder:
                self._recorder.set_task("", task)

        self._step_idx += 1
        self._redundancy.update_step(self._step_idx)

        # Classify the tool
        tool_type = self._classify_swe_tool(tool, args)

        # Track agent thought for expiration detection
        file_path = args.get("file_path", args.get("path", ""))
        known_files = list(self._interceptor.session_reads) + list(self._interceptor.session_edits)
        self._redundancy.track_thought(agent_thought, known_files)

        # Check policy overrides
        override_tier = self._policy.get_override_for(tool_type or "unknown")

        # Safety: warmup period — pass through everything in first N steps
        if self._step_idx <= self._warmup_steps:
            override_tier = "full"

        # Safety: first-access protection — never gate first read of a file
        if self._first_access_full and file_path and file_path not in self._files_accessed:
            self._files_accessed.add(file_path)
            if tool_type in ("read", "search"):
                override_tier = "full"

        # Track all file accesses (even if not first)
        if file_path:
            self._files_accessed.add(file_path)

        # Skip gating for edit tools (pass-through)
        if tool in self._SWE_EDIT_TOOLS:
            self._interceptor.record_edit(file_path)
            self._redundancy.track_edit(file_path)
            return self._make_passthrough(tool, tool_type or "unknown", raw_output)

        # Track test calls
        if tool_type == "test":
            self._redundancy.track_test()

        # --- Capture prior content BEFORE check() mutates state ---
        _diff_prior_content = None
        if tool_type == "read" and file_path and not override_tier:
            if self._redundancy.has_prior_observation(file_path):
                _diff_prior_content = self._redundancy.get_prior_content(file_path)

        # --- Redundancy check (preemptive, before model) ---
        command = args.get("command", args.get("cmd", ""))
        redundancy_signal = self._redundancy.check(
            tool_type=tool_type or "unknown",
            file_path=file_path,
            command=command,
            observation=raw_output,
        )

        # Contextual relevance check: don't hard-block if content is relevant to the task
        context_relevant = False
        if redundancy_signal.should_hard_block and self._task_description and file_path:
            basename = Path(file_path).name if file_path else ""
            stem = Path(file_path).stem if file_path else ""
            if basename and (basename in self._task_description or stem in self._task_description.lower()):
                context_relevant = True
            elif agent_thought and basename and (basename in agent_thought or stem in agent_thought):
                context_relevant = True

        # FIX 2: Skip gating entirely for tiny observations (< 50 tokens)
        raw_tokens_precheck = estimate_tokens(raw_output)
        if raw_tokens_precheck < 50:
            return self._make_passthrough(tool, tool_type or "unknown", raw_output)

        # Hard-block: skip model entirely for clearly redundant observations
        if redundancy_signal.should_hard_block and not override_tier and not context_relevant:
            block_msg = self._redundancy.format_redundancy_block(file_path, redundancy_signal)
            served_tokens = estimate_tokens(block_msg)
            raw_tokens = estimate_tokens(raw_output)
            # FIX 1: Never expand — if block message is larger than raw, pass through
            if served_tokens >= raw_tokens:
                return self._make_passthrough(tool, tool_type or "unknown", raw_output)
            self._redundancy.add_served_tokens(served_tokens)

            # Store for possible escalation
            obs_tool_type = self._tool_type_enum(tool_type)
            obs_id = self._store.store(
                tool_type=obs_tool_type,
                tool_args=args,
                raw_output=raw_output,
                served_tier="redundancy_block",
                served_content=block_msg,
                step_idx=self._step_idx,
            )
            final_content = self._store.format_compact_with_id(obs_id, block_msg)

            recorded = RecordedStep(
                step_idx=self._step_idx,
                tool=tool,
                tool_args=args,
                raw_output=raw_output,
                raw_tokens=raw_tokens,
                served_tier="redundancy_block",
                served_content=final_content,
                served_tokens=served_tokens,
                allocation_decision={
                    "tool_type": tool_type or "unknown",
                    "selected_tier": "redundancy_block",
                    "reasoning": "hard-block: redundant content",
                },
                trajectory_state={"step_idx": self._step_idx},
                agent_thought=agent_thought,
                timestamp=time.time(),
            )
            if self._recorder:
                self._recorder.record(recorded)
            if self._live_metrics:
                self._live_metrics.update(recorded)

            return ProcessedObservation(
                content=final_content,
                obs_id=obs_id,
                tool=tool,
                tool_type=tool_type or "unknown",
                tier_served="redundancy_block",
                raw_tokens=raw_tokens,
                served_tokens=served_tokens,
                compression_ratio=served_tokens / max(raw_tokens, 1),
                escalation_available=True,
                allocation_decision=None,
            )

        # --- Diff-aware delivery: if file was read before and content changed, serve diff ---
        # FIX 3: Only diff when content ACTUALLY changed (edit happened), not when
        # agent zooms into a different view range of the same file
        _view_range = args.get("view_range")
        _prior_had_view_range = False
        if file_path and hasattr(self._redundancy, "_file_last_view_range"):
            _prior_had_view_range = file_path in self._redundancy._file_last_view_range
            _prior_view_range = self._redundancy._file_last_view_range.get(file_path)
        # Track view_range for future comparisons
        if file_path:
            if not hasattr(self._redundancy, "_file_last_view_range"):
                self._redundancy._file_last_view_range = {}
            self._redundancy._file_last_view_range[file_path] = _view_range

        # Determine if this is a genuine file change (edit between reads) vs view-range change
        _is_genuine_change = (
            _diff_prior_content is not None
            and _diff_prior_content != raw_output
            and file_path in getattr(self._redundancy, "_files_edited_since_read", set())
        )
        # Track edits between reads
        if not hasattr(self._redundancy, "_files_edited_since_read"):
            self._redundancy._files_edited_since_read = set()
        if tool_type == "read" and file_path:
            self._redundancy._files_edited_since_read.discard(file_path)

        if _is_genuine_change:
            prior_content = _diff_prior_content
            diff_obs = return_diff(file_path, raw_output, prior_content)
            served_content = diff_obs.content
            served_tokens = diff_obs.tokens_estimate
            raw_tokens = diff_obs.original_tokens
            # FIX 1 (continued): Never expand — if diff is larger than raw, pass through
            if served_tokens >= raw_tokens:
                return self._make_passthrough(tool, tool_type or "unknown", raw_output)
            if True:
                self._redundancy.add_served_tokens(served_tokens)

                obs_tool_type = self._tool_type_enum(tool_type)
                obs_id = self._store.store(
                    tool_type=obs_tool_type,
                    tool_args=args,
                    raw_output=raw_output,
                    served_tier="diff",
                    served_content=served_content,
                    step_idx=self._step_idx,
                )
                final_content = self._store.format_compact_with_id(obs_id, served_content)

                recorded = RecordedStep(
                    step_idx=self._step_idx,
                    tool=tool,
                    tool_args=args,
                    raw_output=raw_output,
                    raw_tokens=raw_tokens,
                    served_tier="diff",
                    served_content=final_content,
                    served_tokens=served_tokens,
                    allocation_decision={
                        "tool_type": tool_type,
                        "selected_tier": "diff",
                        "reasoning": "diff-aware: file changed since last read",
                    },
                    trajectory_state={"step_idx": self._step_idx},
                    agent_thought=agent_thought,
                    timestamp=time.time(),
                )
                if self._recorder:
                    self._recorder.record(recorded)
                if self._live_metrics:
                    self._live_metrics.update(recorded)

                return ProcessedObservation(
                    content=final_content,
                    obs_id=obs_id,
                    tool=tool,
                    tool_type=tool_type,
                    tier_served="diff",
                    raw_tokens=raw_tokens,
                    served_tokens=served_tokens,
                    compression_ratio=served_tokens / max(raw_tokens, 1),
                    escalation_available=True,
                    allocation_decision=None,
                )

        # Pass redundancy features to interceptor for model input
        self._interceptor.set_redundancy_features(redundancy_signal.as_features())

        # Apply allocation via interceptor (or override)
        if override_tier:
            # Policy override: force a specific tier
            result = self._interceptor.intercept(
                tool=self._normalize_tool_name(tool),
                args=self._normalize_args(tool, args),
                observation=raw_output,
                reasoning=agent_thought,
                problem=self._task_description,
            )
            # Override the tier (interceptor already computed features/state)
            if override_tier == "full":
                served_content = raw_output
            else:
                served_content = result.observation
            tier = override_tier
        else:
            # Normal allocation
            result = self._interceptor.intercept(
                tool=self._normalize_tool_name(tool),
                args=self._normalize_args(tool, args),
                observation=raw_output,
                reasoning=agent_thought,
                problem=self._task_description,
            )
            served_content = result.observation
            tier = result.tier

        # FIX 1 (final guard): If any tier produces more tokens than raw, fall back to full
        raw_tokens = estimate_tokens(raw_output)
        served_candidate_tokens = estimate_tokens(served_content)
        if tier != "full" and served_candidate_tokens >= raw_tokens:
            served_content = raw_output
            tier = "full"

        # Store in observation store for escalation
        obs_tool_type = self._tool_type_enum(tool_type)
        obs_id = self._store.store(
            tool_type=obs_tool_type,
            tool_args=args,
            raw_output=raw_output,
            served_tier=tier,
            served_content=served_content,
            step_idx=self._step_idx,
        )

        # Append obs_id to content if not full tier
        final_content = served_content
        if tier != "full":
            final_content = self._store.format_compact_with_id(obs_id, served_content)

        raw_tokens = estimate_tokens(raw_output)
        served_tokens = estimate_tokens(final_content)
        self._redundancy.add_served_tokens(served_tokens)

        # Build allocation decision dict for recording
        allocation_dict = {
            "tool_type": result.tool_type,
            "selected_tier": result.tier,
            "threshold_used": result.allocation.threshold_used if result.allocation else 0.0,
            "reasoning": result.allocation.reasoning if result.allocation else "",
            "cost_multiplier": result.allocation.cost_multiplier if result.allocation else 1.0,
            "redundancy_score": redundancy_signal.redundancy_score,
        }

        # Record step
        recorded = RecordedStep(
            step_idx=self._step_idx,
            tool=tool,
            tool_args=args,
            raw_output=raw_output,
            raw_tokens=raw_tokens,
            served_tier=tier,
            served_content=final_content,
            served_tokens=served_tokens,
            allocation_decision=allocation_dict,
            trajectory_state={
                "step_idx": self._step_idx,
                "files_read": len(self._interceptor.session_reads),
                "files_edited": len(self._interceptor.session_edits),
                "recent_failures": self._interceptor.recent_failures,
            },
            agent_thought=agent_thought,
            timestamp=time.time(),
        )

        if self._recorder:
            self._recorder.record(recorded)
        if self._live_metrics:
            self._live_metrics.update(recorded)

        return ProcessedObservation(
            content=final_content,
            obs_id=obs_id,
            tool=tool,
            tool_type=tool_type or "unknown",
            tier_served=tier,
            raw_tokens=raw_tokens,
            served_tokens=served_tokens,
            compression_ratio=served_tokens / max(raw_tokens, 1),
            escalation_available=(tier != "full"),
            allocation_decision=result.allocation if result else None,
        )

    def handle_observe_more(self, obs_id: str, level: str = "full") -> str:
        """Handle the observe_more tool call from the agent.

        Args:
            obs_id: The observation ID to escalate.
            level: Fidelity level to serve ('full', 'outline', 'relevant_spans').

        Returns:
            The higher-fidelity observation content.
        """
        content = self._store.observe_more(obs_id, level)
        if content is None:
            return f"[Error: observation '{obs_id}' not found or expired]"

        # Record the escalation
        escalation_tokens = estimate_tokens(content)
        recorded = RecordedStep(
            step_idx=self._step_idx + 1,
            tool="observe_more",
            tool_args={"obs_id": obs_id, "level": level},
            raw_output=content,
            raw_tokens=escalation_tokens,
            served_tier="full",
            served_content=content,
            served_tokens=escalation_tokens,
            allocation_decision={"tool_type": "escalation", "selected_tier": level},
            trajectory_state={"step_idx": self._step_idx},
            agent_thought="Agent requested escalation",
            timestamp=time.time(),
        )

        if self._recorder:
            self._recorder.record(recorded)
        if self._live_metrics:
            self._live_metrics.update(recorded)

        self._step_idx += 1
        return content

    def get_metrics(self) -> dict:
        """Get current real-time metrics summary."""
        metrics = {}

        if self._live_metrics:
            metrics.update(self._live_metrics.get_summary())

        # Add store stats
        store_stats = self._store.get_stats()
        metrics["store_observations"] = store_stats.get("observations", 0)
        metrics["store_escalation_requests"] = store_stats.get("escalation_requests", 0)
        metrics["store_auto_escalations"] = store_stats.get("auto_escalations", 0)

        return metrics

    def get_trajectory(self) -> dict:
        """Get the full recorded trajectory for offline analysis."""
        if not self._recorder:
            return {"error": "Trajectory recording disabled"}

        return {
            "task_description": self._task_description,
            "total_steps": self._step_idx,
            "offline_metrics": self._recorder.compute_offline_metrics(),
            "steps": [
                {
                    "step_idx": s.step_idx,
                    "tool": s.tool,
                    "tool_args": s.tool_args,
                    "served_tier": s.served_tier,
                    "raw_tokens": s.raw_tokens,
                    "served_tokens": s.served_tokens,
                    "allocation_decision": s.allocation_decision,
                    "timestamp": s.timestamp,
                }
                for s in self._recorder.get_steps()
            ],
        }

    def get_observe_more_tool(self) -> dict:
        """Return the observe_more tool definition to add to agent's tools."""
        return OBSERVE_MORE_TOOL

    @property
    def policy(self) -> ConfigurablePolicy:
        """Access the configurable policy for runtime tuning."""
        return self._policy

    def reset(self):
        """Reset all state between tasks."""
        self._step_idx = 0
        self._task_description = ""
        self._files_accessed = set()
        self._interceptor.reset()
        self._redundancy.reset()
        self._store = ObservationStore()
        if self._recorder:
            self._recorder.reset()
        if self._live_metrics:
            self._live_metrics.reset()

    # --- Private helpers ---

    def _classify_swe_tool(self, tool: str, args: dict) -> Optional[str]:
        """Classify SWE-agent tool name to our tool type categories."""
        if tool in self._SWE_READ_TOOLS:
            return "read"
        if tool in self._SWE_SEARCH_TOOLS:
            return "search"
        if tool in self._SWE_BASH_TOOLS:
            cmd = args.get("command", args.get("cmd", ""))
            if any(kw in cmd for kw in self._SWE_TEST_KEYWORDS):
                return "test"
            return "bash"
        if tool in self._SWE_EDIT_TOOLS:
            return None  # Edits are pass-through
        return None

    def _normalize_tool_name(self, tool: str) -> str:
        """Map SWE-agent tool names to names expected by the interceptor."""
        mapping = {
            "open": "read_file",
            "goto": "read_file",
            "scroll_down": "read_file",
            "scroll_up": "read_file",
            "view": "view_file",
            "find_file": "find_file",
            "search_dir": "grep",
            "search_file": "grep",
            "find": "find",
            "grep": "grep",
            "bash": "bash",
            "terminal": "bash",
            "execute": "execute_bash",
        }
        return mapping.get(tool, tool)

    def _normalize_args(self, tool: str, args: dict) -> dict:
        """Normalize SWE-agent args to standard form expected by interceptor."""
        normalized = dict(args)
        # Normalize file path key
        if "path" in normalized and "file_path" not in normalized:
            normalized["file_path"] = normalized["path"]
        # Normalize command key
        if "cmd" in normalized and "command" not in normalized:
            normalized["command"] = normalized["cmd"]
        # Normalize search pattern
        if "search_term" in normalized and "pattern" not in normalized:
            normalized["pattern"] = normalized["search_term"]
        return normalized

    def _tool_type_enum(self, tool_type: Optional[str]) -> ToolType:
        """Convert string tool type to ToolType enum."""
        mapping = {
            "read": ToolType.READ,
            "bash": ToolType.BASH,
            "test": ToolType.TEST,
            "search": ToolType.SEARCH,
        }
        return mapping.get(tool_type or "", ToolType.BASH)

    def _make_passthrough(self, tool: str, tool_type: str, raw_output: str) -> ProcessedObservation:
        """Create a pass-through observation (no gating)."""
        tokens = estimate_tokens(raw_output)
        return ProcessedObservation(
            content=raw_output,
            obs_id="",
            tool=tool,
            tool_type=tool_type,
            tier_served="full",
            raw_tokens=tokens,
            served_tokens=tokens,
            compression_ratio=1.0,
            escalation_available=False,
            allocation_decision=None,
        )


# =============================================================================
# OpenHandsHOPWrapper — adapted for OpenHands/OpenDevin format
# =============================================================================


class OpenHandsHOPWrapper:
    """HOP integration wrapper for OpenHands/OpenDevin.

    OpenHands uses different action/observation conventions:
      - CmdRunAction -> bash/test
      - FileReadAction -> read
      - FileWriteAction -> edit (pass-through)
      - BrowseURLAction -> read (web content)
      - IPythonRunCellAction -> bash

    The wrapper translates OpenHands action types to HOP's tool taxonomy
    before applying the observation allocation policy.
    """

    # OpenHands action type classification
    _OH_READ_ACTIONS = {"FileReadAction", "BrowseURLAction"}
    _OH_BASH_ACTIONS = {"CmdRunAction", "IPythonRunCellAction"}
    _OH_EDIT_ACTIONS = {"FileWriteAction", "FileEditAction"}
    _OH_SEARCH_ACTIONS = {"FileSearchAction"}

    def __init__(
        self,
        models_dir: Optional[str] = None,
        risk_threshold: float = 0.15,
        record_trajectory: bool = True,
        live_metrics: bool = True,
    ):
        # Delegate to SWE-agent wrapper with OpenHands-specific classification
        self._inner = SWEAgentHOPWrapper(
            models_dir=models_dir,
            risk_threshold=risk_threshold,
            record_trajectory=record_trajectory,
            live_metrics=live_metrics,
        )

    def process_action_result(
        self,
        action_type: str,
        action_args: dict,
        observation_content: str,
        agent_thought: str = "",
        task: str = "",
    ) -> ProcessedObservation:
        """Process an OpenHands action result through HOP.

        Args:
            action_type: OpenHands action class name (e.g., 'CmdRunAction').
            action_args: Action arguments (command, path, etc.).
            observation_content: The full observation content from execution.
            agent_thought: Agent's reasoning before this action.
            task: Task description.

        Returns:
            ProcessedObservation with allocated content.
        """
        # Translate OpenHands action to normalized tool name
        tool = self._translate_action_type(action_type)
        args = self._translate_action_args(action_type, action_args)

        return self._inner.process_observation(
            tool=tool,
            args=args,
            raw_output=observation_content,
            agent_thought=agent_thought,
            task=task,
        )

    def handle_observe_more(self, obs_id: str, level: str = "full") -> str:
        """Handle escalation request."""
        return self._inner.handle_observe_more(obs_id, level)

    def get_metrics(self) -> dict:
        """Get metrics."""
        return self._inner.get_metrics()

    def get_trajectory(self) -> dict:
        """Get trajectory."""
        return self._inner.get_trajectory()

    def reset(self):
        """Reset between tasks."""
        self._inner.reset()

    def _translate_action_type(self, action_type: str) -> str:
        """Map OpenHands action type to normalized tool name."""
        mapping = {
            "CmdRunAction": "bash",
            "IPythonRunCellAction": "bash",
            "FileReadAction": "read_file",
            "FileWriteAction": "edit",
            "FileEditAction": "edit",
            "BrowseURLAction": "read_file",
            "FileSearchAction": "grep",
        }
        return mapping.get(action_type, "bash")

    def _translate_action_args(self, action_type: str, args: dict) -> dict:
        """Normalize OpenHands action args."""
        normalized = {}

        if action_type in ("CmdRunAction", "IPythonRunCellAction"):
            normalized["command"] = args.get("command", args.get("code", ""))
        elif action_type == "FileReadAction":
            normalized["file_path"] = args.get("path", "")
        elif action_type in ("FileWriteAction", "FileEditAction"):
            normalized["file_path"] = args.get("path", "")
        elif action_type == "BrowseURLAction":
            normalized["file_path"] = args.get("url", "")
        elif action_type == "FileSearchAction":
            normalized["pattern"] = args.get("query", args.get("regex", ""))
            normalized["file_path"] = args.get("path", "")

        return normalized


# =============================================================================
# GenericAgentHOPWrapper — wraps any agent with a tool-call interface
# =============================================================================


class GenericAgentHOPWrapper:
    """HOP wrapper for any agent with a standard tool-call interface.

    Works with any agent that:
      1. Produces tool calls with (tool_name, arguments, thought)
      2. Receives observations as strings

    You provide a tool_type_classifier function that maps your agent's
    tool names to HOP tool types ('read', 'bash', 'test', 'search').
    """

    def __init__(
        self,
        tool_type_classifier: Optional[dict[str, str]] = None,
        models_dir: Optional[str] = None,
        risk_threshold: float = 0.15,
        record_trajectory: bool = True,
        live_metrics: bool = True,
    ):
        """Initialize the generic wrapper.

        Args:
            tool_type_classifier: Dict mapping tool names to tool types.
                Example: {"cat": "read", "ls": "search", "run_tests": "test"}
                If None, uses a sensible default classification.
            models_dir: Path to trained tier risk models.
            risk_threshold: Base risk threshold (rho).
            record_trajectory: Whether to record full trajectory.
            live_metrics: Whether to track live metrics.
        """
        self._tool_classifier = tool_type_classifier or {}
        self._inner = SWEAgentHOPWrapper(
            models_dir=models_dir,
            risk_threshold=risk_threshold,
            record_trajectory=record_trajectory,
            live_metrics=live_metrics,
        )

    def register_tool(self, tool_name: str, tool_type: str):
        """Register a tool name with its HOP tool type.

        Args:
            tool_name: Your agent's tool name.
            tool_type: One of 'read', 'bash', 'test', 'search', 'edit'.
        """
        valid_types = {"read", "bash", "test", "search", "edit"}
        if tool_type not in valid_types:
            raise ValueError(f"Invalid tool_type '{tool_type}'. Expected one of {valid_types}")
        self._tool_classifier[tool_name] = tool_type

    def process(
        self,
        tool_name: str,
        tool_args: dict,
        raw_output: str,
        thought: str = "",
        task: str = "",
    ) -> ProcessedObservation:
        """Process a tool call through HOP.

        Args:
            tool_name: The tool that was called.
            tool_args: Arguments passed to the tool.
            raw_output: Full output from tool execution.
            thought: Agent's reasoning before this call.
            task: Task description.

        Returns:
            ProcessedObservation with allocated content.
        """
        # Determine tool type from classifier
        tool_type = self._tool_classifier.get(tool_name)

        # Map to a normalized tool name the inner wrapper understands
        if tool_type == "read":
            normalized_tool = "read_file"
        elif tool_type == "search":
            normalized_tool = "grep"
        elif tool_type == "test":
            normalized_tool = "bash"
            # Ensure command contains test keyword for classification
            if "command" not in tool_args:
                tool_args = dict(tool_args)
                tool_args["command"] = f"pytest {tool_args.get('test', '')}"
        elif tool_type == "edit":
            normalized_tool = "edit"
        elif tool_type == "bash":
            normalized_tool = "bash"
        else:
            # Unknown tool type: pass through as bash
            normalized_tool = "bash"

        return self._inner.process_observation(
            tool=normalized_tool,
            args=tool_args,
            raw_output=raw_output,
            agent_thought=thought,
            task=task,
        )

    def handle_observe_more(self, obs_id: str, level: str = "full") -> str:
        """Handle escalation."""
        return self._inner.handle_observe_more(obs_id, level)

    def get_metrics(self) -> dict:
        """Get metrics."""
        return self._inner.get_metrics()

    def get_trajectory(self) -> dict:
        """Get trajectory."""
        return self._inner.get_trajectory()

    @property
    def policy(self) -> ConfigurablePolicy:
        """Access the configurable policy."""
        return self._inner.policy

    def reset(self):
        """Reset between tasks."""
        self._inner.reset()


# =============================================================================
# Demo — synthetic agent loop
# =============================================================================


def main():
    """Demo with a synthetic agent loop showing HOP in action."""
    print("=" * 70)
    print("SWE-Agent HOP Integration Wrapper — Demo")
    print("=" * 70)

    # Create wrapper
    wrapper = SWEAgentHOPWrapper(
        models_dir=None,  # No trained models; uses heuristic fallback
        risk_threshold=0.15,
        record_trajectory=True,
        live_metrics=True,
    )

    task = (
        "Fix the TypeError in django/db/models/query.py where "
        "QuerySet.order_by() raises an error when using F() expressions "
        "with transforms."
    )

    # Simulate 5 tool calls with different types
    synthetic_calls = [
        {
            "tool": "find_file",
            "args": {"search_term": "order_by", "path": "django/db/models/"},
            "output": "\n".join(
                f"django/db/models/{f}.py" for f in ["query", "sql/compiler", "expressions", "lookups", "fields"]
            ),
            "thought": "Let me find files related to order_by to understand the codebase.",
        },
        {
            "tool": "open",
            "args": {"path": "django/db/models/query.py"},
            "output": (
                "class QuerySet:\n"
                + "    def order_by(self, *field_names):\n"
                + "        assert not self.query.is_sliced\n"
                + "        obj = self._chain()\n"
                + "        obj.query.clear_ordering(force=True)\n"
                + "        obj.query.add_ordering(*field_names)\n"
                + "        return obj\n\n"
            )
            * 20,  # Simulate a large file
            "thought": "This is likely the file with the bug. Let me read it.",
        },
        {
            "tool": "bash",
            "args": {"command": "python -m pytest tests/queries/test_ordering.py -x"},
            "output": (
                "FAILED tests/queries/test_ordering.py::TestOrderBy::test_f_with_transform\n"
                "TypeError: Expected str, got Transform\n"
                "\n"
                "Traceback (most recent call last):\n"
                "  File 'tests/queries/test_ordering.py', line 42\n"
                "    qs = Model.objects.order_by(F('name').desc())\n"
                "  File 'django/db/models/query.py', line 1234\n"
                "    obj.query.add_ordering(*field_names)\n"
                "TypeError: Expected str, got Transform\n"
                "\n"
                "1 failed, 14 passed in 2.3s\n"
            ),
            "thought": "Let me run the failing test to see the exact error.",
        },
        {
            "tool": "open",
            "args": {"path": "django/db/models/sql/compiler.py"},
            "output": (
                "class SQLCompiler:\n"
                + "    def get_order_by(self):\n"
                + "        result = []\n"
                + "        for expr in self.query.order_by:\n"
                + "            if hasattr(expr, 'resolve_expression'):\n"
                + "                expr = expr.resolve_expression(self.query)\n"
                + "            result.append(expr)\n"
                + "        return result\n\n"
            )
            * 15,
            "thought": "Let me check the SQL compiler to see how it processes order_by.",
        },
        {
            "tool": "search_dir",
            "args": {"search_term": "add_ordering", "path": "django/db/models/"},
            "output": "\n".join(
                [
                    "django/db/models/sql/query.py:456:    def add_ordering(self, *ordering):",
                    "django/db/models/sql/query.py:460:        for item in ordering:",
                    "django/db/models/sql/query.py:461:            if isinstance(item, str):",
                    "django/db/models/sql/query.py:463:            elif hasattr(item, 'resolve_expression'):",
                    "django/db/models/sql/query.py:465:            else:",
                    "django/db/models/sql/query.py:466:                raise TypeError(f'Expected str, got {type(item).__name__}')",
                    "django/db/models/query.py:1234:        obj.query.add_ordering(*field_names)",
                    "tests/queries/test_ordering.py:42:        qs = Model.objects.order_by(F('name').desc())",
                ]
            ),
            "thought": "Let me search for add_ordering to find where the type check happens.",
        },
    ]

    print(f"\nTask: {task}")
    print(f"\n{'Step':<5} {'Tool':<15} {'Tier':<18} {'Raw tok':<10} {'Served tok':<12} {'Savings':<10}")
    print("-" * 75)

    for i, call in enumerate(synthetic_calls):
        result = wrapper.process_observation(
            tool=call["tool"],
            args=call["args"],
            raw_output=call["output"],
            agent_thought=call["thought"],
            task=task,
        )

        savings = (1 - result.compression_ratio) * 100
        print(
            f"{i + 1:<5} {call['tool']:<15} {result.tier_served:<18} "
            f"{result.raw_tokens:<10} {result.served_tokens:<12} "
            f"{savings:.1f}%"
        )

    # Print metrics
    print(f"\n{'=' * 75}")
    print("Final Metrics:")
    metrics = wrapper.get_metrics()
    for key, val in sorted(metrics.items()):
        if isinstance(val, dict):
            print(f"  {key}:")
            for k2, v2 in val.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {key}: {val}")

    # Print trajectory summary
    trajectory = wrapper.get_trajectory()
    print(f"\nTrajectory: {trajectory['total_steps']} steps recorded")
    offline = trajectory["offline_metrics"]
    print(f"  Overall savings: {offline.get('overall_savings_pct', 0)}%")
    print(f"  Under-service estimate: {offline.get('estimated_under_service_rate', 0):.1%}")

    # Demo escalation
    print(f"\n{'=' * 75}")
    print("Escalation Demo:")
    # Find an obs_id to escalate
    if trajectory["steps"]:
        first_reduced = next(
            (s for s in trajectory["steps"] if s["served_tier"] != "full"),
            None,
        )
        if first_reduced:
            # Get the obs_id from the store
            store_stats = wrapper._store.get_stats()
            print(f"  Store has {store_stats.get('observations', 0)} stored observations")
            print("  (In real usage, agent would call observe_more(obs_id) to escalate)")

    # Demo ConfigurablePolicy
    print(f"\n{'=' * 75}")
    print("ConfigurablePolicy Demo:")
    policy = wrapper.policy
    print(f"  Initial config: {policy.get_config()}")
    policy.set_rho(0.30)
    print(f"  After set_rho(0.30): {policy.get_config()}")
    policy.set_tool_override("search", "count")
    print(f"  After override search->count: {policy.get_config()}")
    policy.set_conservative_mode(True)
    print(f"  After conservative_mode=True: {policy.get_config()}")

    print(f"\n{'=' * 75}")
    print("Demo complete. observe_more tool definition:")
    print(f"  {json.dumps(OBSERVE_MORE_TOOL, indent=2)}")


if __name__ == "__main__":
    main()
