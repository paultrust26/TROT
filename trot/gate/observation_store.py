"""Observation Store — stores raw tool outputs and serves graduated fidelity with escalation.

The store holds full raw observations outside the LLM context. When a tool returns,
the middleware stores the full output here, selects a fidelity level via the policy,
and injects only the compact rendering. The agent can request higher fidelity via
observe_more(obs_id, level).

Auto-escalation triggers:
- Agent re-calls the same tool (detects it needs more)
- Agent fails a test after an edit (may need full context)
- Consecutive low-confidence predictions
- Agent explicitly requests escalation
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .observation_actions import (
    estimate_tokens,
    generate_observation,
    ReducedObservation,
    ACTIONS_BY_COST,
)


class ToolType(Enum):
    READ = "read"
    BASH = "bash"
    TEST = "test"
    SEARCH = "search"


TIER_ORDER = {
    ToolType.READ: [
        "block_with_pointer",
        "return_summary",
        "return_ast_outline",
        "return_relevant_spans",
        "return_cached_view",
        "return_diff",
        "allow_full",
    ],
    ToolType.BASH: ["exit_code_only", "first_error", "structured_summary", "full"],
    ToolType.TEST: ["exit_code_only", "failing_names", "first_failure", "all_failures", "full"],
    ToolType.SEARCH: ["count_only", "filenames_only", "top_matches", "full"],
}


@dataclass
class StoredObservation:
    obs_id: str
    tool_type: ToolType
    tool_args: dict
    raw_output: str
    raw_tokens: int
    served_tier: str
    served_content: str
    served_tokens: int
    timestamp: float
    step_idx: int
    escalation_count: int = 0
    was_auto_escalated: bool = False


class ObservationStore:
    """In-memory store for raw observations with escalation support."""

    def __init__(self, max_observations: int = 200):
        self._store: dict[str, StoredObservation] = {}
        self._max = max_observations
        self._call_history: list[tuple[str, dict, int]] = []  # (tool_type, args_key, step_idx)
        self.total_raw_tokens: int = 0
        self.total_served_tokens: int = 0
        self.escalation_requests: int = 0
        self.auto_escalations: int = 0

    def store(
        self,
        tool_type: ToolType,
        tool_args: dict,
        raw_output: str,
        served_tier: str,
        served_content: str,
        step_idx: int,
    ) -> str:
        """Store a raw observation, return obs_id."""
        obs_id = self._make_id(tool_args, step_idx)
        raw_tokens = estimate_tokens(raw_output)
        served_tokens = estimate_tokens(served_content)

        obs = StoredObservation(
            obs_id=obs_id,
            tool_type=tool_type,
            tool_args=tool_args,
            raw_output=raw_output,
            raw_tokens=raw_tokens,
            served_tier=served_tier,
            served_content=served_content,
            served_tokens=served_tokens,
            timestamp=time.time(),
            step_idx=step_idx,
        )
        self._store[obs_id] = obs
        self._call_history.append((tool_type.value, self._args_key(tool_args), step_idx))

        self.total_raw_tokens += raw_tokens
        self.total_served_tokens += served_tokens

        if len(self._store) > self._max:
            oldest_id = min(self._store, key=lambda k: self._store[k].timestamp)
            del self._store[oldest_id]

        return obs_id

    def observe_more(self, obs_id: str, requested_level: str = "full") -> Optional[str]:
        """Escalate: return higher-fidelity rendering of stored observation."""
        if obs_id not in self._store:
            return None

        obs = self._store[obs_id]
        obs.escalation_count += 1
        self.escalation_requests += 1

        if requested_level == "full":
            return obs.raw_output

        if obs.tool_type == ToolType.READ:
            file_path = obs.tool_args.get("file_path", "")
            result = generate_observation(
                action=requested_level,
                file_path=file_path,
                file_content=obs.raw_output,
                task_description=obs.tool_args.get("task_description", ""),
            )
            return result.content

        return obs.raw_output

    def should_auto_escalate(self, tool_type: ToolType, tool_args: dict, step_idx: int) -> tuple[bool, Optional[str]]:
        """Check if auto-escalation should trigger. Returns (should_escalate, obs_id)."""
        args_key = self._args_key(tool_args)

        for obs in reversed(list(self._store.values())):
            if obs.tool_type != tool_type:
                continue
            if self._args_key(obs.tool_args) == args_key and obs.served_tier != "allow_full":
                obs.was_auto_escalated = True
                self.auto_escalations += 1
                return True, obs.obs_id

        return False, None

    def check_test_failure_escalation(self, edited_files: list[str]) -> list[str]:
        """After a test failure, escalate recent reads of edited files."""
        escalate_ids = []
        for obs in reversed(list(self._store.values())):
            if obs.tool_type != ToolType.READ:
                continue
            file_path = obs.tool_args.get("file_path", "")
            if file_path in edited_files and obs.served_tier != "allow_full":
                escalate_ids.append(obs.obs_id)
                obs.was_auto_escalated = True
                self.auto_escalations += 1
        return escalate_ids

    def consecutive_blocks_count(self, lookback: int = 5) -> int:
        """Count consecutive reduced observations at end of history."""
        count = 0
        for obs in reversed(list(self._store.values())):
            if obs.served_tier in ("allow_full", "full"):
                break
            count += 1
            if count >= lookback:
                break
        return count

    def format_compact_with_id(self, obs_id: str, compact_content: str) -> str:
        """Format the compact observation with obs_id for injection into context."""
        return f'{compact_content}\n[obs_id: {obs_id} | Use observe_more("{obs_id}") for higher fidelity]'

    def get_stats(self) -> dict:
        """Return stats on store usage and escalation frequency."""
        if not self._store:
            return {"observations": 0}

        tiers = {}
        for obs in self._store.values():
            tiers[obs.served_tier] = tiers.get(obs.served_tier, 0) + 1

        savings = 1.0 - (self.total_served_tokens / max(self.total_raw_tokens, 1))
        return {
            "observations": len(self._store),
            "total_raw_tokens": self.total_raw_tokens,
            "total_served_tokens": self.total_served_tokens,
            "token_savings_pct": round(savings * 100, 1),
            "escalation_requests": self.escalation_requests,
            "auto_escalations": self.auto_escalations,
            "tier_distribution": tiers,
        }

    def _make_id(self, tool_args: dict, step_idx: int) -> str:
        key = f"{sorted(tool_args.items())}_{step_idx}_{time.time()}"
        return hashlib.sha256(key.encode()).hexdigest()[:10]

    def _args_key(self, tool_args: dict) -> str:
        relevant = {k: v for k, v in tool_args.items() if k != "task_description"}
        return hashlib.md5(str(sorted(relevant.items())).encode()).hexdigest()[:12]


class EscalationPolicy:
    """Decides when to auto-escalate and manages dynamic risk thresholds."""

    def __init__(
        self,
        base_risk_threshold: float = 0.05,
        max_consecutive_blocks: int = 3,
    ):
        self.base_threshold = base_risk_threshold
        self.max_consecutive_blocks = max_consecutive_blocks

    def check_triggers(self, store: ObservationStore, current_step: dict, trajectory_state: dict) -> list[str]:
        """Return list of obs_ids that should be auto-escalated."""
        escalate_ids = []

        # Trigger 1: agent re-calls same tool+args
        tool = current_step.get("tool", "")
        args = current_step.get("arguments", {})
        tool_type = self._classify_tool(tool)
        if tool_type:
            should, obs_id = store.should_auto_escalate(tool_type, args, trajectory_state.get("step_idx", 0))
            if should and obs_id:
                escalate_ids.append(obs_id)

        # Trigger 2: test failure after edit
        if trajectory_state.get("test_just_failed", False):
            edited = trajectory_state.get("recently_edited_files", [])
            escalate_ids.extend(store.check_test_failure_escalation(edited))

        # Trigger 3: too many consecutive blocks → loosen next prediction
        if store.consecutive_blocks_count() >= self.max_consecutive_blocks:
            recent = list(store._store.values())
            if recent:
                last_blocked = recent[-1]
                if last_blocked.served_tier != "allow_full":
                    escalate_ids.append(last_blocked.obs_id)

        return list(set(escalate_ids))

    def compute_threshold(self, trajectory_state: dict) -> float:
        """Dynamically adjust risk threshold based on trajectory state.

        Lower threshold (conservative, more full observations):
        - After test failures
        - Before first edit of a file
        - Early in exploration phase

        Higher threshold (aggressive, more reduction):
        - During search spirals
        - Re-reading unchanged files
        - Late in trajectory with budget depleting
        """
        threshold = self.base_threshold

        # Conservative adjustments (lower threshold → serve more)
        if trajectory_state.get("tests_failing", False):
            threshold *= 0.5
        if trajectory_state.get("step_idx", 0) < 5:
            threshold *= 0.7
        if trajectory_state.get("first_read_of_edit_target", False):
            threshold *= 0.3

        # Aggressive adjustments (higher threshold → reduce more)
        if trajectory_state.get("in_search_spiral", False):
            threshold *= 2.0
        if trajectory_state.get("hash_unchanged_reread", False):
            threshold *= 3.0
        if trajectory_state.get("budget_fraction_used", 0) > 0.8:
            threshold *= 1.5
        if trajectory_state.get("steps_without_productive", 0) > 10:
            threshold *= 2.0

        return min(max(threshold, 0.01), 0.3)

    def _classify_tool(self, tool: str) -> Optional[ToolType]:
        read_tools = {"read_file", "view_file", "str_replace_editor"}
        bash_tools = {"bash", "run_command", "execute_bash"}
        search_tools = {"grep", "find", "search", "find_file"}

        if tool in read_tools:
            return ToolType.READ
        if tool in bash_tools:
            return ToolType.BASH
        if tool in search_tools:
            return ToolType.SEARCH
        return None


class ObservationMiddleware:
    """Main integration point: wraps the agent loop with observation allocation.

    Usage:
        middleware = ObservationMiddleware(policy=policy, store=store)

        # In agent loop:
        raw_output = tool.execute(args)
        result = middleware.process(tool_type, args, raw_output, trajectory_state)
        # result.content goes into LLM context (compact + obs_id)
        # Full output stored in middleware.store
    """

    def __init__(
        self,
        store: Optional[ObservationStore] = None,
        escalation_policy: Optional[EscalationPolicy] = None,
        allocator=None,  # ObservationAllocator from inference_gate.py
    ):
        self.store = store or ObservationStore()
        self.policy = escalation_policy or EscalationPolicy()
        self.allocator = allocator
        self._step_idx = 0

    def process(
        self,
        tool_type: ToolType,
        tool_args: dict,
        raw_output: str,
        trajectory_state: dict,
    ) -> ReducedObservation:
        """Process a tool output: allocate fidelity, store full, return compact."""
        self._step_idx += 1

        # Check auto-escalation first (agent re-calling same tool)
        should_escalate, prior_obs_id = self.store.should_auto_escalate(tool_type, tool_args, self._step_idx)
        if should_escalate and prior_obs_id:
            full_content = self.store.observe_more(prior_obs_id, "full")
            if full_content:
                obs_id = self.store.store(tool_type, tool_args, raw_output, "allow_full", raw_output, self._step_idx)
                return ReducedObservation(
                    action="allow_full",
                    content=raw_output,
                    tokens_estimate=estimate_tokens(raw_output),
                    original_tokens=estimate_tokens(raw_output),
                    compression_ratio=1.0,
                    available_escalations=[],
                )

        # Select tier via allocator (or default to full if no model loaded)
        if self.allocator:
            tier = self.allocator.allocate(tool_type, tool_args, trajectory_state)
        else:
            tier = "allow_full"

        # Render at selected tier
        if tool_type == ToolType.READ:
            file_path = tool_args.get("file_path", "")
            rendered = generate_observation(
                action=tier,
                file_path=file_path,
                file_content=raw_output,
                task_description=tool_args.get("task_description", ""),
            )
        else:
            rendered = ReducedObservation(
                action=tier,
                content=raw_output,
                tokens_estimate=estimate_tokens(raw_output),
                original_tokens=estimate_tokens(raw_output),
                compression_ratio=1.0,
                available_escalations=[],
            )

        # Store full observation
        obs_id = self.store.store(tool_type, tool_args, raw_output, tier, rendered.content, self._step_idx)

        # Format with obs_id for context injection
        if tier != "allow_full":
            rendered.content = self.store.format_compact_with_id(obs_id, rendered.content)

        return rendered

    def handle_observe_more(self, obs_id: str, level: str = "full") -> Optional[str]:
        """Handle agent's explicit escalation request."""
        return self.store.observe_more(obs_id, level)

    def get_stats(self) -> dict:
        return self.store.get_stats()

    def reset(self):
        """Reset between tasks."""
        self.store = ObservationStore()
        self._step_idx = 0
