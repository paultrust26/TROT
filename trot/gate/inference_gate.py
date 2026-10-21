"""Observation Allocation Gate — per-tier risk prediction for graduated fidelity.

Instead of binary allow/block, predicts P(under_service | tier=t) for each
available fidelity tier. Selects the cheapest tier where risk < threshold rho.

Architecture:
  - One model per tool type (read, bash, test, search)
  - Each model outputs risk scores for all tiers of that tool type
  - Threshold rho is dynamically adjusted by trajectory state
  - Integrates with ObservationStore for escalation

Prediction target (trained from hindsight):
  For each (tool_call, tier) pair in training data:
    under_service = 1 if serving at this tier caused the agent to:
      - re-call the same tool (needing more info)
      - fail a task that succeeded with full observation
      - miss critical information (counterfactual validation)
    under_service = 0 otherwise
"""

import hashlib
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .feature_extractor import FeatureExtractor, ReadFeatures, Embedder
from .observation_actions import (
    generate_observation,
    block_with_pointer,
    return_summary,
    return_ast_outline,
    return_relevant_spans,
    allow_full,
    ReducedObservation,
    estimate_tokens,
)

# =============================================================================
# Tier definitions per tool type
# =============================================================================

TIERS = {
    "read": ["block", "summary", "outline", "relevant_spans", "full"],
    "bash": ["exit_code", "first_error", "structured", "full"],
    "test": ["exit_code", "failing_names", "first_failure", "all_failures", "full"],
    "search": ["count", "filenames", "top_matches", "full"],
}

# Cost multipliers (relative to full = 1.0) for budget accounting
TIER_COSTS = {
    "read": {"block": 0.01, "summary": 0.05, "outline": 0.10, "relevant_spans": 0.35, "full": 1.0},
    "bash": {"exit_code": 0.02, "first_error": 0.08, "structured": 0.30, "full": 1.0},
    "test": {"exit_code": 0.02, "failing_names": 0.05, "first_failure": 0.15, "all_failures": 0.50, "full": 1.0},
    "search": {"count": 0.01, "filenames": 0.05, "top_matches": 0.25, "full": 1.0},
}


@dataclass
class AllocationDecision:
    """Result of the observation allocator."""

    tool_type: str
    selected_tier: str
    risk_scores: dict[str, float]  # tier -> P(under_service | tier)
    threshold_used: float
    reasoning: str
    cost_multiplier: float
    escalation_available: bool


# =============================================================================
# TierRiskModel — wraps XGBoost/LightGBM for per-tier risk prediction
# =============================================================================


class TierRiskModel:
    """Predicts P(under_service | tier=t) for each tier of a given tool type.

    Wraps a trained gradient boosting model (XGBoost or LightGBM) that outputs
    calibrated probabilities. One model instance per tool type.

    The model takes the standard feature vector and outputs risk scores for
    all tiers simultaneously (multi-output regression with isotonic calibration).
    """

    def __init__(self, tool_type: str, model_path: Optional[Path] = None):
        self.tool_type = tool_type
        self.tiers = TIERS[tool_type]
        self.num_tiers = len(self.tiers)
        self.model = None
        self.calibrators = None  # Per-tier isotonic regression calibrators
        self.feature_names: list[str] = []

        if model_path and Path(model_path).exists():
            self._load(model_path)

    def _load(self, path: Path):
        """Load a trained model checkpoint.

        Supports two formats:
          - Legacy: checkpoint["model"] is a _TierModelWrapper with .predict(X) -> (n, n_tiers)
          - New (from train_from_extracted): checkpoint["models"] is a list of per-tier classifiers
        """
        with open(path, "rb") as f:
            checkpoint = pickle.load(f)

        if "models" in checkpoint:
            # New format: list of per-tier classifiers
            self._per_tier_models = checkpoint["models"]
            self._valid_tiers = checkpoint.get("valid_tiers", list(range(self.num_tiers)))
            self.model = self  # sentinel so is_loaded() returns True
        else:
            # Legacy format: single model wrapper
            self._per_tier_models = None
            self.model = checkpoint["model"]

        self.calibrators = checkpoint.get("calibrators", None)
        self.feature_names = checkpoint.get("feature_names", [])

    def predict_risk(self, features: np.ndarray) -> dict[str, float]:
        """Predict under-service risk for each tier.

        Args:
            features: Feature vector (1D array from FeatureExtractor or similar).

        Returns:
            Dict mapping tier name -> P(under_service | tier).
            Risk is monotonically decreasing as tier quality increases
            (higher tiers have lower under-service risk by construction).
        """
        if self.model is None:
            return {tier: 1.0 for tier in self.tiers}

        features_2d = features.reshape(1, -1)

        # New format: per-tier classifiers list
        if getattr(self, "_per_tier_models", None) is not None:
            raw_scores = np.ones(self.num_tiers, dtype=np.float64)
            for tier_idx in self._valid_tiers:
                clf = self._per_tier_models[tier_idx]
                if clf is not None:
                    raw_scores[tier_idx] = clf.predict_proba(features_2d)[0][1]
            raw_scores[-1] = 0.0  # full tier never under-served
        else:
            # Legacy format: single model wrapper with .predict()
            try:
                raw_scores = self.model.predict(features_2d)[0]
            except Exception:
                try:
                    raw_scores = np.array(
                        [self.model[i].predict_proba(features_2d)[0][1] for i in range(self.num_tiers)]
                    )
                except Exception:
                    return {tier: 1.0 for tier in self.tiers}

        # Apply isotonic calibration if available
        if self.calibrators is not None:
            calibrated = np.zeros_like(raw_scores)
            for i in range(self.num_tiers):
                if i < len(self.calibrators) and self.calibrators[i] is not None:
                    calibrated[i] = self.calibrators[i].predict([raw_scores[i]])[0]
                else:
                    calibrated[i] = raw_scores[i]
        else:
            calibrated = raw_scores

        # Enforce monotonicity: risk must decrease with tier quality
        risk_scores = {}
        prev_risk = 1.0
        for i, tier in enumerate(self.tiers):
            risk = float(np.clip(calibrated[i], 0.0, prev_risk))
            risk_scores[tier] = risk
            prev_risk = risk

        return risk_scores

    def is_loaded(self) -> bool:
        return self.model is not None


# =============================================================================
# DynamicThreshold — adjusts rho based on trajectory state
# =============================================================================


@dataclass
class TrajectoryState:
    """Accumulated state for dynamic threshold computation."""

    step_index: int = 0
    total_steps_estimate: int = 30
    files_read: list[str] = field(default_factory=list)
    files_edited: list[str] = field(default_factory=list)
    recent_failures: int = 0  # failures in last 5 steps
    consecutive_reads_without_edit: int = 0
    is_in_spiral: bool = False
    spiral_length: int = 0
    budget_fraction_used: float = 0.0
    last_tool_was_escalation: bool = False
    current_file_in_traceback: bool = False
    current_file_is_likely_edit_target: bool = False
    hash_unchanged_reread: bool = False
    tests_currently_failing: bool = False
    duplicate_call_detected: bool = False


class DynamicThreshold:
    """Adjusts rho (under-service risk threshold) based on trajectory state.

    Lower rho = more conservative = prefer higher-fidelity tiers.
    Higher rho = more aggressive = accept cheaper tiers more readily.

    Design principles:
      - Conservative (low rho) after failures, early exploration, first reads
        of likely edit targets, when tests are failing.
      - Aggressive (high rho) during spirals, re-reads of unchanged files,
        budget depletion, duplicate calls.
    """

    def __init__(
        self,
        base_rho: float = 0.15,
        min_rho: float = 0.02,
        max_rho: float = 0.60,
    ):
        self.base_rho = base_rho
        self.min_rho = min_rho
        self.max_rho = max_rho

    def compute(self, state: TrajectoryState) -> float:
        """Compute the dynamic threshold given current trajectory state.

        Progressive gating schedule:
          Steps 0-5:  rho=0 → model CANNOT gate, only deterministic rules fire
          Steps 5-15: rho ramps 0 → base_rho (model gains authority gradually)
          Steps 15+:  rho=base_rho with dynamic adjustments
        """
        step = state.step_index

        # --- Progressive gate: disable model for early steps ---
        if step <= 8:
            return 0.0  # Only deterministic rules (diff, redundancy_block) fire
        elif step < 20:
            # Linear ramp from 0 to base_rho over steps 9-20
            ramp = (step - 8) / 12.0
            rho = self.base_rho * ramp
        else:
            rho = self.base_rho

        # --- Conservative adjustments (lower rho) ---

        # Recent failures: agent is struggling, give it full info
        if state.recent_failures > 0:
            rho *= max(0.3, 1.0 - 0.2 * state.recent_failures)

        # First read of likely edit target: always give full
        if state.current_file_is_likely_edit_target:
            rho *= 0.3

        # File appears in traceback: high priority
        if state.current_file_in_traceback:
            rho *= 0.4

        # Tests are currently failing: agent needs full context
        if state.tests_currently_failing:
            rho *= 0.7

        # After escalation: agent explicitly asked for more, be generous
        if state.last_tool_was_escalation:
            rho *= 0.5

        # --- Aggressive adjustments (higher rho) ---

        # Search/read spiral: agent is going in circles
        if state.is_in_spiral:
            rho *= min(3.0, 1.0 + 0.3 * state.spiral_length)

        # Re-read of unchanged file: likely redundant
        if state.hash_unchanged_reread:
            rho *= 4.0

        # Duplicate call: very likely redundant
        if state.duplicate_call_detected:
            rho *= 5.0

        # Budget depletion: become more aggressive as budget runs out
        if state.budget_fraction_used > 0.7:
            budget_pressure = (state.budget_fraction_used - 0.7) / 0.3  # 0 to 1
            rho *= 1.0 + 2.0 * budget_pressure

        # Many consecutive reads without editing: probably spiraling
        if state.consecutive_reads_without_edit > 5:
            excess = state.consecutive_reads_without_edit - 5
            rho *= 1.0 + 0.2 * excess

        # FIX 5: Position-weighted compression — early observations repeat in ALL
        # future prompts, so compressing them has multiplicative token savings.
        if state.total_steps_estimate > 0:
            step_fraction = step / state.total_steps_estimate
            position_multiplier = 1.0 + 1.0 * (1.0 - step_fraction * 2)
            position_multiplier = max(0.5, min(2.0, position_multiplier))
            rho *= position_multiplier

        # Clamp to valid range
        return float(np.clip(rho, self.min_rho, self.max_rho))


# =============================================================================
# ObservationAllocator — the main gate
# =============================================================================


class ObservationAllocator:
    """Main allocation gate: takes features + tool_type, returns selected tier.

    Workflow:
      1. Extract features for the tool call
      2. Get risk predictions for all tiers from TierRiskModel
      3. Compute dynamic threshold rho from TrajectoryState
      4. Select cheapest tier where P(under_service | tier) < rho
      5. Return AllocationDecision with selected tier + metadata
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        base_rho: float = 0.15,
        fallback_to_heuristic: bool = True,
    ):
        self.models: dict[str, TierRiskModel] = {}
        self.threshold_controller = DynamicThreshold(base_rho=base_rho)
        self.fallback_to_heuristic = fallback_to_heuristic
        self.heuristic = HeuristicAllocator()

        # Load models for each tool type
        if models_dir:
            models_dir = Path(models_dir)
            for tool_type in TIERS:
                model_path = models_dir / f"tier_risk_{tool_type}.pkl"
                self.models[tool_type] = TierRiskModel(tool_type, model_path)
        else:
            for tool_type in TIERS:
                self.models[tool_type] = TierRiskModel(tool_type)

    def allocate(
        self,
        features: np.ndarray,
        tool_type: str,
        state: TrajectoryState,
    ) -> AllocationDecision:
        """Select the optimal tier for a tool call.

        Args:
            features: Feature vector from FeatureExtractor.
            tool_type: One of 'read', 'bash', 'test', 'search'.
            state: Current trajectory state for dynamic threshold.

        Returns:
            AllocationDecision with selected tier and metadata.
        """
        if tool_type not in TIERS:
            return AllocationDecision(
                tool_type=tool_type,
                selected_tier="full",
                risk_scores={},
                threshold_used=0.0,
                reasoning=f"Unknown tool type '{tool_type}', defaulting to full",
                cost_multiplier=1.0,
                escalation_available=False,
            )

        # --- Deterministic rules (fire BEFORE model, 100% confidence) ---
        rule_decision = self._apply_deterministic_rules(tool_type, state)
        if rule_decision is not None:
            return rule_decision

        model = self.models.get(tool_type)
        tiers = TIERS[tool_type]

        # Use heuristic fallback if no trained model
        if model is None or not model.is_loaded():
            if self.fallback_to_heuristic:
                return self.heuristic.allocate(tool_type, state)
            else:
                return AllocationDecision(
                    tool_type=tool_type,
                    selected_tier="full",
                    risk_scores={tier: 1.0 for tier in tiers},
                    threshold_used=0.0,
                    reasoning="No trained model, defaulting to full",
                    cost_multiplier=1.0,
                    escalation_available=False,
                )

        # Get per-tier risk predictions
        risk_scores = model.predict_risk(features)

        # Compute dynamic threshold
        rho = self.threshold_controller.compute(state)

        # Guard: for bash/test, only allow model-based reduction on repeated commands.
        # First-time bash/test commands always get full — the agent hasn't seen
        # this output before and needs it to make progress.
        if tool_type in ("bash", "test") and not state.duplicate_call_detected and not state.hash_unchanged_reread:
            return AllocationDecision(
                tool_type=tool_type,
                selected_tier="full",
                risk_scores=risk_scores,
                threshold_used=rho,
                reasoning="Guard: first-time bash/test command → always full",
                cost_multiplier=TIER_COSTS[tool_type]["full"],
                escalation_available=False,
            )

        # Select cheapest tier where risk < rho (tiers ordered cheap to expensive)
        selected_tier = tiers[-1]  # default: full (safest)
        for tier in tiers:
            if risk_scores.get(tier, 1.0) < rho:
                selected_tier = tier
                break

        cost = TIER_COSTS[tool_type][selected_tier]
        escalation_available = selected_tier != tiers[-1]

        reasoning_parts = [
            f"rho={rho:.3f}",
            f"risks=[{', '.join(f'{t}:{risk_scores[t]:.2f}' for t in tiers)}]",
            f"selected={selected_tier} (cost={cost:.2f})",
        ]

        return AllocationDecision(
            tool_type=tool_type,
            selected_tier=selected_tier,
            risk_scores=risk_scores,
            threshold_used=rho,
            reasoning=" | ".join(reasoning_parts),
            cost_multiplier=cost,
            escalation_available=escalation_available,
        )

    def _apply_deterministic_rules(
        self,
        tool_type: str,
        state: TrajectoryState,
    ) -> Optional[AllocationDecision]:
        """Apply deterministic rules that always produce correct decisions.

        These handle cases where the model already achieves 100% accuracy,
        removing trivial cases from the model's burden so it can focus on
        ambiguous decisions.

        Returns None if no rule fires (fall through to model).
        """
        tiers = TIERS[tool_type]
        risk_scores = {tier: 0.0 for tier in tiers}

        # RULE 1: Re-read of unchanged file → block
        # (Model already gets this 100% right; making it a hard rule is safer)
        if tool_type == "read" and state.hash_unchanged_reread and state.duplicate_call_detected:
            return AllocationDecision(
                tool_type=tool_type,
                selected_tier="block",
                risk_scores=risk_scores,
                threshold_used=0.0,
                reasoning="Rule: identical re-read of unchanged content → block",
                cost_multiplier=TIER_COSTS[tool_type]["block"],
                escalation_available=True,
            )

        # RULE 2: Duplicate bash command with same output → cached
        if tool_type == "bash" and state.duplicate_call_detected:
            return AllocationDecision(
                tool_type=tool_type,
                selected_tier="exit_code",
                risk_scores=risk_scores,
                threshold_used=0.0,
                reasoning="Rule: duplicate bash command → cached/exit_code",
                cost_multiplier=TIER_COSTS[tool_type]["exit_code"],
                escalation_available=True,
            )

        # RULE 3: Same test command with no edits since last run → block
        if tool_type == "test" and state.duplicate_call_detected:
            return AllocationDecision(
                tool_type=tool_type,
                selected_tier="exit_code",
                risk_scores=risk_scores,
                threshold_used=0.0,
                reasoning="Rule: same test with no edits since → block/cached",
                cost_multiplier=TIER_COSTS[tool_type]["exit_code"],
                escalation_available=True,
            )

        # RULE 4: Post-edit verification read → summary (agent just wants confirmation)
        if tool_type == "read" and state.last_tool_was_escalation is False:
            # Check via the state if this is a verification read
            # (previous action was edit of same file — tracked externally)
            pass

        # RULE 5: First read of file in traceback → always full (never gate)
        if tool_type == "read" and state.current_file_in_traceback:
            current_file = state.files_read[-1] if state.files_read else ""
            if current_file and current_file not in state.files_read[:-1]:
                return AllocationDecision(
                    tool_type=tool_type,
                    selected_tier="full",
                    risk_scores=risk_scores,
                    threshold_used=1.0,
                    reasoning="Rule: first read of traceback file → always full",
                    cost_multiplier=TIER_COSTS[tool_type]["full"],
                    escalation_available=False,
                )

        return None


# =============================================================================
# HeuristicAllocator — rule-based fallback when no trained model
# =============================================================================


class HeuristicAllocator:
    """Rule-based tier allocation as fallback when no trained model is loaded.

    Rules (applied in priority order):
      1. Exact duplicate call -> block / exit_code (cheapest tier)
      2. Hash unchanged re-read -> cached_view equivalent (summary for reads)
      3. Search spiral detected -> count / filenames
      4. File in traceback or likely edit target -> full
      5. Everything else -> outline / first_error (moderate tier)
    """

    def allocate(
        self,
        tool_type: str,
        state: TrajectoryState,
    ) -> AllocationDecision:
        """Apply heuristic rules to select a tier."""
        tiers = TIERS[tool_type]
        risk_scores = {tier: 0.5 for tier in tiers}  # Placeholder scores

        # Rule 1: Exact duplicate call -> cheapest tier
        if state.duplicate_call_detected:
            selected = tiers[0]  # block / exit_code / count
            reasoning = "Heuristic: exact duplicate call detected, blocking"
            return self._build_decision(tool_type, selected, risk_scores, 0.0, reasoning)

        # Rule 2: Hash unchanged re-read -> summary/exit_code
        if state.hash_unchanged_reread:
            if tool_type == "read":
                selected = "summary"
            elif tool_type == "bash":
                selected = "exit_code"
            elif tool_type == "test":
                selected = "exit_code"
            else:
                selected = "filenames"
            reasoning = "Heuristic: content unchanged since last observation"
            return self._build_decision(tool_type, selected, risk_scores, 0.0, reasoning)

        # Rule 3: Search/read spiral -> count/filenames
        if state.is_in_spiral and state.spiral_length > 4:
            if tool_type == "read":
                selected = "outline"
            elif tool_type == "search":
                selected = "filenames" if state.spiral_length < 7 else "count"
            elif tool_type == "bash":
                selected = "first_error"
            else:
                selected = "failing_names"
            reasoning = f"Heuristic: spiral detected (length={state.spiral_length})"
            return self._build_decision(tool_type, selected, risk_scores, 0.0, reasoning)

        # Rule 4: File in traceback or likely edit target -> full
        if state.current_file_in_traceback or state.current_file_is_likely_edit_target:
            selected = "full"
            reasoning = "Heuristic: high-priority file (traceback/edit target), serving full"
            return self._build_decision(tool_type, selected, risk_scores, 0.0, reasoning)

        # Rule 5: Default — for tool types without trained models, prefer full
        # Only apply moderate tiers to 'read' (which has a trained model for most cases)
        default_tiers = {
            "read": "outline",
            "bash": "full",
            "test": "first_failure",
            "search": "top_matches",
        }
        selected = default_tiers.get(tool_type, tiers[-1])
        reasoning = "Heuristic: default tier (no trained model for this tool type)"
        return self._build_decision(tool_type, selected, risk_scores, 0.0, reasoning)

    def _build_decision(
        self,
        tool_type: str,
        selected_tier: str,
        risk_scores: dict[str, float],
        rho: float,
        reasoning: str,
    ) -> AllocationDecision:
        tiers = TIERS[tool_type]
        cost = TIER_COSTS[tool_type].get(selected_tier, 1.0)
        escalation_available = selected_tier != tiers[-1]
        return AllocationDecision(
            tool_type=tool_type,
            selected_tier=selected_tier,
            risk_scores=risk_scores,
            threshold_used=rho,
            reasoning=reasoning,
            cost_multiplier=cost,
            escalation_available=escalation_available,
        )


# =============================================================================
# TrainingDataGenerator — converts tool_use_labels output to tier risk data
# =============================================================================


class TrainingDataGenerator:
    """Generates (features, tier, under_service_label) tuples for training.

    Takes traces with hindsight labels (from tool_use_labels.py) and produces
    training data for the per-tier risk model. For each tool call in the trace,
    generates one sample per tier with a binary under-service label.

    Under-service signal detection:
      1. Re-call: agent calls the same tool again within N steps (needed more)
      2. Task failure: serving at this tier correlated with task failure
      3. Information miss: counterfactual — output at this tier would have
         missed symbols that the agent used in later edits

    The key insight: for a given tool call, lower tiers have higher under-service
    risk. A call that was productive at full tier would likely be under-served
    at block tier. We model this gradient, not just binary useful/useless.
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        re_call_horizon: int = 5,
        info_miss_threshold: float = 0.3,
    ):
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.re_call_horizon = re_call_horizon
        self.info_miss_threshold = info_miss_threshold

    def generate_from_trace(
        self,
        trace: dict,
        labels: list[dict],
    ) -> list[dict]:
        """Generate training samples from a single trace with labels.

        Args:
            trace: Full trace dict with steps.
            labels: List of ToolUseLabel dicts (from tool_use_labels.py output).

        Returns:
            List of training sample dicts, each with:
              - features: np.ndarray
              - tool_type: str
              - tier: str
              - under_service: int (0 or 1)
              - weight: float (sample weight for imbalanced tiers)
        """
        steps = trace["steps"]
        task_id = trace.get("task_id", "")
        samples = []

        # Index labels by step_idx for fast lookup
        label_by_step = {lab["step_idx"]: lab for lab in labels}

        for step_idx, step in enumerate(steps):
            tool = step.get("tool", "")
            tool_type = self._classify_tool_type(tool, step.get("arguments", {}))

            if tool_type is None:
                continue

            if tool_type not in TIERS:
                continue

            # Get the hindsight label for this step
            label = label_by_step.get(step_idx)
            if label is None:
                continue

            # Extract features
            features = self._extract_features_for_step(trace, step_idx, tool_type)
            if features is None:
                continue

            # Compute under-service signal for each tier
            tier_labels = self._compute_tier_labels(trace, step_idx, step, label, tool_type)

            # Generate one sample per tier
            tiers = TIERS[tool_type]
            for tier_idx, tier in enumerate(tiers):
                under_service = tier_labels.get(tier, 1)

                # Sample weight: higher weight for boundary cases (where model
                # will actually make a difference)
                weight = self._compute_sample_weight(tier_idx, len(tiers), label)

                samples.append(
                    {
                        "task_id": task_id,
                        "step_idx": step_idx,
                        "tool_type": tool_type,
                        "tier": tier,
                        "tier_idx": tier_idx,
                        "features": features,
                        "under_service": under_service,
                        "weight": weight,
                    }
                )

        return samples

    def generate_batch(
        self,
        traces: list[dict],
        labels_per_trace: list[list[dict]],
    ) -> dict:
        """Generate training data from multiple traces.

        Returns:
            Dict with keys per tool_type, each containing:
              - X: np.ndarray of shape (n_samples, n_features)
              - y: np.ndarray of shape (n_samples, n_tiers) — per-tier labels
              - weights: np.ndarray of shape (n_samples,)
        """
        all_samples: dict[str, list[dict]] = {tt: [] for tt in TIERS}

        for trace, labels in zip(traces, labels_per_trace):
            samples = self.generate_from_trace(trace, labels)
            for sample in samples:
                all_samples[sample["tool_type"]].append(sample)

        result = {}
        for tool_type, samples in all_samples.items():
            if not samples:
                continue

            tiers = TIERS[tool_type]
            n_tiers = len(tiers)
            n_samples = len(samples) // n_tiers  # Each step produces n_tiers samples

            if n_samples == 0:
                continue

            # Group by step to build per-step feature + multi-tier label
            step_groups: dict[tuple[str, int], dict] = {}
            for sample in samples:
                key = (sample["task_id"], sample["step_idx"])
                if key not in step_groups:
                    step_groups[key] = {
                        "features": sample["features"],
                        "labels": np.zeros(n_tiers, dtype=np.float32),
                        "weight": 0.0,
                    }
                tier_idx = sample["tier_idx"]
                step_groups[key]["labels"][tier_idx] = sample["under_service"]
                step_groups[key]["weight"] = max(step_groups[key]["weight"], sample["weight"])

            groups = list(step_groups.values())
            X = np.array([g["features"] for g in groups], dtype=np.float32)
            y = np.array([g["labels"] for g in groups], dtype=np.float32)
            weights = np.array([g["weight"] for g in groups], dtype=np.float32)

            result[tool_type] = {"X": X, "y": y, "weights": weights}

        return result

    def _classify_tool_type(self, tool: str, args: dict) -> Optional[str]:
        """Map a tool name to our tool type categories."""
        if tool in ("read_file", "view_file"):
            return "read"
        if tool == "str_replace_editor" and args.get("command") == "view":
            return "read"
        if tool in ("bash", "run_command", "execute_bash"):
            cmd = args.get("command", "")
            # Distinguish test runs from general bash
            if any(kw in cmd for kw in ("pytest", "test", "unittest", "npm test", "cargo test")):
                return "test"
            return "bash"
        if tool in ("grep", "find", "search", "find_file"):
            return "search"
        return None

    def _extract_features_for_step(
        self,
        trace: dict,
        step_idx: int,
        tool_type: str,
    ) -> Optional[np.ndarray]:
        """Extract features for a tool call at step_idx."""
        step = trace["steps"][step_idx]
        tool = step.get("tool", "")

        # For read tools, use the full FeatureExtractor
        if tool_type == "read" and tool in ("read_file", "view_file", "str_replace_editor"):
            try:
                read_features = self.feature_extractor.extract(trace, step_idx)
                return read_features.to_array()
            except Exception:
                pass

        # For other tool types, build a simpler feature vector
        return self._extract_generic_features(trace, step_idx, tool_type)

    def _extract_generic_features(
        self,
        trace: dict,
        step_idx: int,
        tool_type: str,
    ) -> np.ndarray:
        """Extract a generic feature vector for non-read tool types."""
        steps = trace["steps"]
        step = steps[step_idx]
        args = step.get("arguments", {})
        obs = step.get("observation", "")
        total_steps = len(steps)

        # Session state features
        progress = step_idx / max(total_steps - 1, 1)
        files_read = set()
        files_edited = set()
        last_edit_step = -1
        tests_failing = False
        similar_calls = 0
        steps_since_last_edit = step_idx

        for i in range(step_idx):
            s = steps[i]
            t = s.get("tool", "")
            s_args = s.get("arguments", {})
            s_obs = s.get("observation", "")

            if t in ("read_file", "view_file"):
                files_read.add(s_args.get("file_path", ""))
            elif t in ("edit_file", "str_replace_editor", "write_file"):
                if not (t == "str_replace_editor" and s_args.get("command") == "view"):
                    files_edited.add(s_args.get("file_path", ""))
                    last_edit_step = i
            elif t in ("bash", "run_command", "execute_bash"):
                if s_obs and ("FAILED" in s_obs or "Error" in s_obs):
                    tests_failing = True
                elif s_obs and ("PASSED" in s_obs or "OK" in s_obs):
                    tests_failing = False

            # Count similar prior calls
            if s.get("tool") == step.get("tool"):
                similar_calls += 1

        if last_edit_step >= 0:
            steps_since_last_edit = step_idx - last_edit_step

        # Command/argument features
        cmd = args.get("command", args.get("pattern", args.get("query", "")))
        cmd_length = len(cmd) if cmd else 0
        obs_length = len(obs)

        # Build feature vector (20 base + 10 redundancy = 30 features)
        base_features = [
            progress,
            float(step_idx),
            float(len(files_read)),
            float(len(files_edited)),
            float(steps_since_last_edit),
            float(similar_calls),
            1.0 if tests_failing else 0.0,
            float(cmd_length) / 200.0,
            float(obs_length) / 5000.0,
            float(min(obs_length, 10000)) / 10000.0,
            # Observation content signals
            1.0 if obs and "Error" in obs else 0.0,
            1.0 if obs and "Traceback" in obs else 0.0,
            1.0 if obs and "PASSED" in obs else 0.0,
            1.0 if obs and "FAILED" in obs else 0.0,
            # Tool-type specific
            1.0 if tool_type == "bash" else 0.0,
            1.0 if tool_type == "test" else 0.0,
            1.0 if tool_type == "search" else 0.0,
            # Session dynamics
            float(len(files_read)) / max(step_idx, 1),
            float(len(files_edited)) / max(step_idx, 1),
            float(step_idx) / max(total_steps, 1),
        ]

        # Offline redundancy features (computed from trace replay)
        redundancy = self._compute_offline_redundancy(trace, step_idx, step, tool_type)
        base_features.extend(redundancy)

        return np.array(base_features, dtype=np.float32)

    def _compute_offline_redundancy(
        self,
        trace: dict,
        step_idx: int,
        step: dict,
        tool_type: str,
    ) -> list[float]:
        """Compute 10 redundancy features from trace replay (offline equivalent of SessionRedundancyTracker)."""
        steps = trace["steps"]
        obs = step.get("observation", "")
        args = step.get("arguments", {})
        file_path = args.get("file_path", "")
        cmd = args.get("command", args.get("pattern", ""))

        obs_tokens = len(obs) // 4  # approximate
        content_hash = hashlib.sha256(obs.encode()).hexdigest()[:16] if obs else ""

        is_duplicate = 0.0
        is_unchanged_reread = 0.0
        file_access_count = 0.0
        steps_since_last_view = float(step_idx)
        content_overlap = 0.0
        is_superseded = 0.0
        context_tokens_so_far = 0
        is_below_threshold = 1.0 if obs_tokens < 500 else 0.0

        last_edit_step = -1
        last_test_step = -1

        for i in range(step_idx):
            s = steps[i]
            s_obs = s.get("observation", "")
            s_args = s.get("arguments", {})
            s_tool = s.get("tool", "")
            context_tokens_so_far += len(s_obs) // 4

            # Track edits and tests for supersession
            if s_tool in ("edit_file", "str_replace_editor", "write_file"):
                if not (s_tool == "str_replace_editor" and s_args.get("command") == "view"):
                    last_edit_step = i
            if s_tool in ("bash", "run_command") and any(
                kw in s_args.get("command", "") for kw in ("pytest", "test", "unittest")
            ):
                last_test_step = i

            # Check for exact duplicate content
            if s_obs and content_hash and hashlib.sha256(s_obs.encode()).hexdigest()[:16] == content_hash:
                is_duplicate = 1.0
                steps_since_last_view = float(step_idx - i)

            # File re-read tracking
            if file_path and s_args.get("file_path") == file_path:
                file_access_count += 1
                steps_since_last_view = float(step_idx - i)
                # Check if content is unchanged
                if s_obs == obs:
                    is_unchanged_reread = 1.0
                    content_overlap = 1.0
                elif s_obs and obs:
                    content_overlap = max(content_overlap, 0.3)

        # Supersession: test output after edit makes prior tests stale
        if tool_type == "test" and last_edit_step > last_test_step and last_test_step >= 0:
            is_superseded = 1.0

        # Redundancy score (composite)
        score = 0.0
        if is_unchanged_reread > 0:
            score += 0.5
        if file_access_count > 2:
            score += 0.2
        if content_overlap > 0.8:
            score += 0.3
        if is_superseded > 0:
            score += 0.4
        score = min(score, 1.0)

        return [
            is_duplicate,
            is_unchanged_reread,
            file_access_count,
            min(steps_since_last_view, 30) / 30.0,
            content_overlap,
            is_superseded,
            context_tokens_so_far / 100_000.0,
            min(obs_tokens, 10000) / 10000.0,
            is_below_threshold,
            score,
        ]

    def _compute_tier_labels(
        self,
        trace: dict,
        step_idx: int,
        step: dict,
        label: dict,
        tool_type: str,
    ) -> dict[str, int]:
        """Compute under-service label for each tier.

        Logic per tier (from cheapest to most expensive):
          - Cheapest tier (block/exit_code/count): under-served unless the call
            was completely redundant/dead_end.
          - Intermediate tiers: under-served if the agent needed specific
            information that this tier would not have provided.
          - Full tier: never under-served (by definition).

        Returns:
            Dict mapping tier -> 0 (adequate) or 1 (under-served).
        """
        tiers = TIERS[tool_type]
        outcome = label.get("outcome", "uncertain")
        was_productive = label.get("was_productive", False)
        is_duplicate = label.get("is_duplicate_call", False)
        is_near_duplicate = label.get("is_near_duplicate", False)
        output_utilization = label.get("output_utilization", 0.0)
        led_to_edit = label.get("led_to_edit", False)
        part_of_spiral = label.get("part_of_search_spiral", False)
        line_utilization = label.get("line_utilization_ratio", 0.0)

        # Detect re-call (agent called same tool again soon after)
        re_called = self._detect_re_call(trace, step_idx, step)

        tier_labels = {}

        for tier_idx, tier in enumerate(tiers):
            if tier == tiers[-1]:
                # Full tier is never under-served
                tier_labels[tier] = 0
                continue

            if tier == tiers[0]:
                # Cheapest tier: under-served unless completely useless
                if outcome in ("dead_end", "redundant") and is_duplicate:
                    tier_labels[tier] = 0  # Blocking was fine
                elif not was_productive and not led_to_edit and not re_called:
                    tier_labels[tier] = 0  # Call was useless anyway
                else:
                    tier_labels[tier] = 1  # Would have under-served
                continue

            # Intermediate tiers: graduated based on information needs
            tier_position = tier_idx / (len(tiers) - 1)  # 0=cheapest, 1=full

            # If the call was unproductive and not re-called, any tier suffices
            if outcome in ("dead_end", "redundant") and not re_called:
                tier_labels[tier] = 0
                continue

            # If spiral and not highly productive, intermediate tiers are fine
            if part_of_spiral and output_utilization < 0.2:
                tier_labels[tier] = 0
                continue

            # For productive calls: would this tier have provided enough info?
            if was_productive:
                # Estimate: if tier provides tier_position fraction of info,
                # and agent used output_utilization of full output,
                # then under-served if tier_position < output_utilization
                info_needed = max(output_utilization, line_utilization)
                if tier_position >= info_needed:
                    tier_labels[tier] = 0  # This tier provides enough
                else:
                    tier_labels[tier] = 1  # Under-served
            elif re_called:
                # Agent re-called: needs more than the cheapest tiers
                # Higher intermediate tiers might suffice
                if tier_position >= 0.6:
                    tier_labels[tier] = 0  # Probably enough at this quality
                else:
                    tier_labels[tier] = 1
            else:
                # Uncertain: default to under-served for lower tiers
                tier_labels[tier] = 1 if tier_position < 0.5 else 0

        return tier_labels

    def _detect_re_call(
        self,
        trace: dict,
        step_idx: int,
        step: dict,
    ) -> bool:
        """Detect if the agent re-called the same/similar tool within horizon."""
        steps = trace["steps"]
        tool = step.get("tool", "")
        file_path = step.get("arguments", {}).get("file_path", "")
        cmd = step.get("arguments", {}).get("command", "")

        for j in range(step_idx + 1, min(step_idx + 1 + self.re_call_horizon, len(steps))):
            future_step = steps[j]
            future_tool = future_step.get("tool", "")
            future_args = future_step.get("arguments", {})

            # Same tool + same file
            if future_tool == tool and future_args.get("file_path", "") == file_path and file_path:
                return True

            # Same bash/search command pattern
            if future_tool == tool and tool in ("bash", "run_command", "execute_bash"):
                future_cmd = future_args.get("command", "")
                if cmd and future_cmd and self._command_similarity(cmd, future_cmd) > 0.6:
                    return True

            # Same search with refined query
            if future_tool == tool and tool in ("grep", "search", "find"):
                pattern = step.get("arguments", {}).get("pattern", "")
                future_pattern = future_args.get("pattern", "")
                if pattern and future_pattern and (pattern in future_pattern or future_pattern in pattern):
                    return True

        return False

    def _command_similarity(self, cmd_a: str, cmd_b: str) -> float:
        """Token-level Jaccard similarity between two commands."""
        tokens_a = set(re.findall(r"\w+", cmd_a.lower()))
        tokens_b = set(re.findall(r"\w+", cmd_b.lower()))
        if not tokens_a or not tokens_b:
            return 0.0
        return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)

    def _compute_sample_weight(
        self,
        tier_idx: int,
        num_tiers: int,
        label: dict,
    ) -> float:
        """Compute sample weight for training.

        Higher weight for:
          - Boundary tiers (where the decision matters most)
          - Productive calls (more costly if we under-serve)
          - Non-obvious cases (not clear duplicates or dead ends)
        """
        # Base weight: higher for middle tiers (boundary region)
        tier_position = tier_idx / max(num_tiers - 1, 1)
        boundary_weight = 1.0 - 2.0 * abs(tier_position - 0.5)  # peaks at middle

        # Productivity weight
        outcome = label.get("outcome", "uncertain")
        if outcome == "productive":
            prod_weight = 2.0
        elif outcome in ("dead_end", "redundant"):
            prod_weight = 0.5
        else:
            prod_weight = 1.0

        # Non-obvious cases get higher weight
        is_duplicate = label.get("is_duplicate_call", False)
        obvious_weight = 0.3 if is_duplicate else 1.0

        return float(np.clip(boundary_weight * prod_weight * obvious_weight, 0.1, 3.0))


# =============================================================================
# AgentObservationInterceptor — generalized interceptor for all tool types
# =============================================================================


class AgentObservationInterceptor:
    """Drop-in interceptor for agent tool calls across all tool types.

    Generalizes the old AgentReadInterceptor to handle read, bash, test,
    and search tool calls. When the agent makes a tool call, this interceptor:
      1. Classifies the tool type
      2. Extracts features from the current trajectory state
      3. Gets a tier allocation from the ObservationAllocator
      4. Returns a reduced observation at the selected fidelity tier

    Usage in SWE-agent or Claude Code integration:

        allocator = ObservationAllocator(models_dir=Path("./models"))
        interceptor = AgentObservationInterceptor(allocator)

        # In agent loop:
        for action in agent.actions():
            decision = interceptor.intercept(
                tool=action.tool,
                args=action.args,
                observation=execute_tool(action),
                reasoning=agent.last_thought,
                problem=task.problem_statement,
            )
            # decision.observation is the tier-appropriate content
            # decision.tier tells you what fidelity was served
            agent.receive_observation(decision.observation)
    """

    @dataclass
    class InterceptResult:
        """Result of intercepting a tool call."""

        tool_type: str
        tier: str
        observation: str
        original_tokens: int
        served_tokens: int
        compression_ratio: float
        allocation: AllocationDecision
        escalation_hint: str

    def __init__(
        self,
        allocator: ObservationAllocator,
        feature_extractor: Optional[FeatureExtractor] = None,
    ):
        self.allocator = allocator
        self.feature_extractor = feature_extractor or FeatureExtractor()

        # Session state tracking
        self.session_reads: list[str] = []
        self.session_edits: list[str] = []
        self.session_tool_calls: list[dict] = []
        self.step_count: int = 0
        self.file_hashes: dict[str, str] = {}
        self.recent_failures: int = 0
        self.consecutive_reads_without_edit: int = 0
        self.problem_statement: str = ""

    def set_redundancy_features(self, features: dict[str, float]):
        """Set redundancy features from the wrapper's SessionRedundancyTracker.

        These 10 features are appended to the base 20 features in _build_features().
        Call this before intercept() each step when using the redundancy tracker.
        """
        self._pending_redundancy_features = features

    def intercept(
        self,
        tool: str,
        args: dict,
        observation: str,
        reasoning: str = "",
        problem: str = "",
    ) -> "AgentObservationInterceptor.InterceptResult":
        """Intercept a tool call and return a tier-appropriate observation.

        Args:
            tool: Tool name (e.g., 'read_file', 'bash', 'grep').
            args: Tool arguments dict.
            observation: The full observation from executing the tool.
            reasoning: Agent's reasoning/thought before this call.
            problem: Problem statement / task description.

        Returns:
            InterceptResult with the tier-reduced observation.
        """
        if problem:
            self.problem_statement = problem

        tool_type = self._classify_tool(tool, args)
        file_path = args.get("file_path", "")

        # Build trajectory state
        state = self._build_trajectory_state(tool, args, observation, file_path)

        # Extract features
        features = self._build_features(tool_type, args, observation, reasoning)

        # Get allocation decision
        if tool_type and tool_type in TIERS:
            decision = self.allocator.allocate(features, tool_type, state)
        else:
            decision = AllocationDecision(
                tool_type=tool_type or "unknown",
                selected_tier="full",
                risk_scores={},
                threshold_used=0.0,
                reasoning="Non-gated tool type",
                cost_multiplier=1.0,
                escalation_available=False,
            )

        # Apply the tier reduction
        reduced_obs = self._apply_tier(
            tool_type or "unknown",
            decision.selected_tier,
            observation,
            file_path,
            args,
        )

        # Update session state
        self._update_state(tool, args, observation, file_path, decision)

        original_tokens = estimate_tokens(observation)
        served_tokens = estimate_tokens(reduced_obs)

        escalation_hint = ""
        if decision.escalation_available:
            escalation_hint = self._generate_escalation_hint(tool_type, decision.selected_tier)

        return self.InterceptResult(
            tool_type=tool_type or "unknown",
            tier=decision.selected_tier,
            observation=reduced_obs,
            original_tokens=original_tokens,
            served_tokens=served_tokens,
            compression_ratio=served_tokens / max(original_tokens, 1),
            allocation=decision,
            escalation_hint=escalation_hint,
        )

    def record_edit(self, file_path: str):
        """Call when agent edits a file (updates internal state)."""
        self.session_edits.append(file_path)
        self.consecutive_reads_without_edit = 0

    def reset(self):
        """Reset between tasks."""
        self.session_reads = []
        self.session_edits = []
        self.session_tool_calls = []
        self.step_count = 0
        self.file_hashes = {}
        self.recent_failures = 0
        self.consecutive_reads_without_edit = 0
        self.problem_statement = ""

    def _classify_tool(self, tool: str, args: dict) -> Optional[str]:
        """Classify tool into our categories."""
        if tool in ("read_file", "view_file"):
            return "read"
        if tool == "str_replace_editor" and args.get("command") == "view":
            return "read"
        if tool in ("bash", "run_command", "execute_bash"):
            cmd = args.get("command", "")
            if re.search(r"\bpytest\b|\bunittest\b|\bnpm test\b|\bcargo test\b", cmd):
                return "test"
            if re.search(r"python\s+.*-m\s+pytest\b|python\s+.*test", cmd):
                return "test"
            return "bash"
        if tool in ("grep", "find", "search", "find_file"):
            return "search"
        return None

    def _build_trajectory_state(
        self,
        tool: str,
        args: dict,
        observation: str,
        file_path: str,
    ) -> TrajectoryState:
        """Build TrajectoryState from session history."""
        # Check for hash-unchanged re-read
        hash_unchanged = False
        if file_path and file_path in self.file_hashes:
            current_hash = hashlib.sha256(observation.encode()).hexdigest()[:16]
            if current_hash == self.file_hashes.get(file_path):
                hash_unchanged = True

        # Check for duplicate call
        current_call_key = f"{tool}:{hashlib.md5(str(sorted(args.items())).encode()).hexdigest()[:12]}"
        duplicate = any(tc.get("call_key") == current_call_key for tc in self.session_tool_calls[-10:])

        # Detect spiral: many consecutive search/read without edits
        is_in_spiral = self.consecutive_reads_without_edit > 4
        spiral_length = self.consecutive_reads_without_edit if is_in_spiral else 0

        # Check if file is in traceback
        file_in_traceback = False
        if file_path and self.problem_statement:
            basename = Path(file_path).name
            file_in_traceback = basename in self.problem_statement and (
                "Traceback" in self.problem_statement or "Error" in self.problem_statement
            )

        # Check if file is likely edit target
        is_likely_edit_target = False
        if file_path and self.problem_statement:
            stem = Path(file_path).stem
            task_lower = self.problem_statement.lower()
            is_likely_edit_target = (
                Path(file_path).name in self.problem_statement or stem.replace("_", " ") in task_lower
            )

        # Budget usage
        total_calls = max(self.step_count, 1)
        budget_fraction = total_calls / max(30, total_calls)  # rough estimate

        return TrajectoryState(
            step_index=self.step_count,
            total_steps_estimate=30,
            files_read=self.session_reads.copy(),
            files_edited=self.session_edits.copy(),
            recent_failures=self.recent_failures,
            consecutive_reads_without_edit=self.consecutive_reads_without_edit,
            is_in_spiral=is_in_spiral,
            spiral_length=spiral_length,
            budget_fraction_used=budget_fraction,
            last_tool_was_escalation=False,
            current_file_in_traceback=file_in_traceback,
            current_file_is_likely_edit_target=is_likely_edit_target,
            hash_unchanged_reread=hash_unchanged,
            tests_currently_failing=self.recent_failures > 0,
            duplicate_call_detected=duplicate,
        )

    def _build_features(
        self,
        tool_type: Optional[str],
        args: dict,
        observation: str,
        reasoning: str,
    ) -> np.ndarray:
        """Build feature vector matching train_multitool_gate.py:build_feature_vector exactly.

        Feature layout per tool type:
          read:   27 common + 11 read-specific = 38
          bash:   27 common + 10 bash-specific = 37
          test:   27 common + 6 test-specific  = 33
          search: 27 common + 3 search-specific = 30
        """
        file_path = args.get("file_path", "")

        # --- 27 common features (must match training order exactly) ---
        # [0] position_normalized
        position_normalized = self.step_count / 30.0
        # [1] num_steps_so_far
        num_steps = float(self.step_count)
        # [2] obs_token_count / 5000 (capped at 2.0)
        obs_token_count = len(observation.split()) if observation else 0
        obs_token_norm = min(obs_token_count / 5000.0, 2.0)
        # [3] n_files_read
        n_files_read = float(len(self.session_reads))
        # [4] n_files_edited
        n_files_edited = float(len(self.session_edits))
        # [5] content_symbol_overlap
        symbol_overlap = self._compute_symbol_overlap(observation)
        # [6] content_has_function_defs
        has_defs = 0.0
        if observation and re.search(r"^\s*(def |class |function |async def )", observation, re.MULTILINE):
            has_defs = 1.0
        # [7] content_reasoning_edit_intent
        edit_intent = 0.0
        if reasoning:
            intent_words = (
                "fix",
                "change",
                "modify",
                "update",
                "replace",
                "remove",
                "add",
                "patch",
                "correct",
                "edit",
                "rewrite",
                "implement",
            )
            if any(w in reasoning.lower() for w in intent_words):
                edit_intent = 1.0
        # [8] reasoning_len / 300 (capped at 1.0)
        reasoning_len_norm = min(len(reasoning) / 300.0, 1.0)

        # [9-18] 10 redundancy features
        rf = getattr(self, "_pending_redundancy_features", None)
        if rf:
            redundancy = [
                rf.get("redundancy_is_duplicate", 0.0),
                rf.get("redundancy_is_unchanged_reread", 0.0),
                rf.get("redundancy_file_access_count", 0.0),
                rf.get("redundancy_steps_since_last_view", 0.0),
                rf.get("redundancy_content_overlap", 0.0),
                rf.get("redundancy_is_superseded", 0.0),
                rf.get("redundancy_context_budget_used", 0.0),
                rf.get("redundancy_obs_token_ratio", 0.0),
                rf.get("redundancy_is_below_threshold", 0.0),
                rf.get("redundancy_score", 0.0),
            ]
            self._pending_redundancy_features = None
        else:
            redundancy = [0.0] * 10

        # [19] reasoning_is_targeted
        reasoning_is_targeted = 0.0
        if reasoning:
            targeted_patterns = re.findall(r"\b[A-Z][a-z]+[A-Z]\w+\b|\b[a-z_]+\(", reasoning)
            if targeted_patterns:
                reasoning_is_targeted = 1.0
        # [20] reasoning_is_exploring
        reasoning_is_exploring = 0.0
        if reasoning:
            explore_phrases = (
                "let me check",
                "let me look",
                "let me explore",
                "let me see",
                "let me find",
                "let's look",
                "let's check",
            )
            if any(p in reasoning.lower() for p in explore_phrases):
                reasoning_is_exploring = 1.0
        # [21] reasoning_is_repeated (approximate: same tool+path as recent call)
        reasoning_is_repeated = 0.0
        if len(self.session_tool_calls) >= 2:
            recent = self.session_tool_calls[-1] if self.session_tool_calls else {}
            if recent.get("file_path") == file_path and recent.get("tool_type") == tool_type:
                reasoning_is_repeated = 1.0

        # [22] has_edited_yet
        has_edited_yet = 1.0 if self.session_edits else 0.0
        # [23] edits_to_reads_ratio
        edits_to_reads = float(len(self.session_edits)) / max(len(self.session_reads), 1)
        # [24] consecutive_same_tool / 5.0 (capped at 1.0)
        consecutive_same = 0
        for tc in reversed(self.session_tool_calls):
            if tc.get("tool_type") == tool_type:
                consecutive_same += 1
            else:
                break
        consecutive_same_norm = min(consecutive_same / 5.0, 1.0)
        # [25] steps_since_last_productive / 15.0 (capped at 1.0)
        steps_since_productive = self.step_count
        for i, tc in enumerate(reversed(self.session_tool_calls)):
            if tc.get("tool_type") == "edit" or (tc.get("tool_type") == "test" and not self.recent_failures):
                steps_since_productive = i
                break
        steps_since_productive_norm = min(steps_since_productive / 15.0, 1.0)
        # [26] obs_symbols_in_patch (not available at inference, default 0)
        obs_symbols_in_patch = 0.0

        common = [
            position_normalized,  # [0]
            num_steps,  # [1]
            obs_token_norm,  # [2]
            n_files_read,  # [3]
            n_files_edited,  # [4]
            symbol_overlap,  # [5]
            has_defs,  # [6]
            edit_intent,  # [7]
            reasoning_len_norm,  # [8]
            *redundancy,  # [9-18]
            reasoning_is_targeted,  # [19]
            reasoning_is_exploring,  # [20]
            reasoning_is_repeated,  # [21]
            has_edited_yet,  # [22]
            edits_to_reads,  # [23]
            consecutive_same_norm,  # [24]
            steps_since_productive_norm,  # [25]
            obs_symbols_in_patch,  # [26]
        ]

        # --- Tool-type-specific features ---
        if tool_type == "read":
            specific = self._read_specific_features(file_path, observation, reasoning)
        elif tool_type == "bash":
            specific = self._bash_specific_features(args, observation)
        elif tool_type == "test":
            specific = self._test_specific_features(observation)
        elif tool_type == "search":
            specific = self._search_specific_features(args, observation)
        else:
            specific = []

        return np.array(common + specific, dtype=np.float32)

    def _compute_symbol_overlap(self, observation: str) -> float:
        """Symbol overlap between observation identifiers and problem statement."""
        if not observation or not self.problem_statement:
            return 0.0
        obs_ids = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", observation[:3000]))
        problem_ids = set(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{2,}\b", self.problem_statement))
        if obs_ids and problem_ids:
            return min(len(obs_ids & problem_ids) / max(len(problem_ids), 1), 1.0)
        return 0.0

    def _read_specific_features(self, file_path: str, observation: str, reasoning: str) -> list[float]:
        """11 read-specific features matching training."""
        basename = Path(file_path).name if file_path else ""
        stem = Path(file_path).stem if file_path else ""
        ext = Path(file_path).suffix if file_path else ""

        # content_file_in_problem
        file_in_problem = 0.0
        if basename and self.problem_statement:
            if basename in self.problem_statement or stem in self.problem_statement.lower():
                file_in_problem = 1.0

        # content_file_in_reasoning
        file_in_reasoning = 0.0
        if basename and reasoning:
            if basename in reasoning or stem in reasoning.lower():
                file_in_reasoning = 1.0

        # content_file_previously_edited
        file_previously_edited = 1.0 if file_path in self.session_edits else 0.0

        # content_steps_since_last_edit
        last_edit_step = -1
        for i, tc in enumerate(reversed(self.session_tool_calls)):
            if tc.get("tool_type") == "edit":
                last_edit_step = self.step_count - i - 1
                break
        steps_since_edit = float(self.step_count - last_edit_step) if last_edit_step >= 0 else float(self.step_count)
        steps_since_edit_norm = min(steps_since_edit, 30) / 30.0

        # content_line_count_norm
        line_count = observation.count("\n") + 1 if observation else 0
        line_count_norm = min(line_count, 500) / 500.0

        # was_mentioned_in_prior_search
        was_in_search = 0.0
        for tc in self.session_tool_calls:
            if tc.get("tool_type") == "search" and file_path:
                was_in_search = 1.0
                break

        # ext == ".py"
        is_python = 1.0 if ext == ".py" else 0.0

        # "test" in file_path
        is_test_file = 1.0 if "test" in (file_path or "").lower() else 0.0

        # file_in_same_dir_as_edited
        file_in_same_dir = 0.0
        if file_path and self.session_edits:
            file_dir = str(Path(file_path).parent)
            for ed in self.session_edits:
                if str(Path(ed).parent) == file_dir:
                    file_in_same_dir = 1.0
                    break

        # file_in_same_package
        file_in_same_package = 0.0
        if file_path and self.session_edits:
            file_parts = Path(file_path).parts
            pkg = file_parts[1] if len(file_parts) > 1 else ""
            for ed in self.session_edits:
                ed_parts = Path(ed).parts
                ed_pkg = ed_parts[1] if len(ed_parts) > 1 else ""
                if pkg and pkg == ed_pkg:
                    file_in_same_package = 1.0
                    break

        # file_in_final_patch (not available at inference, default 0)
        file_in_final_patch = 0.0

        return [
            file_in_problem,
            file_in_reasoning,
            file_previously_edited,
            steps_since_edit_norm,
            line_count_norm,
            was_in_search,
            is_python,
            is_test_file,
            file_in_same_dir,
            file_in_same_package,
            file_in_final_patch,
        ]

    def _bash_specific_features(self, args: dict, observation: str) -> list[float]:
        """10 bash-specific features matching training."""
        cmd = args.get("command", "")
        cmd_lower = cmd.lower().split()[0] if cmd.strip() else ""

        cmd_type_grep = 1.0 if cmd_lower == "grep" else 0.0
        cmd_type_find = 1.0 if cmd_lower == "find" else 0.0
        cmd_type_python = 1.0 if cmd_lower in ("python", "python3") else 0.0
        cmd_type_git = 1.0 if cmd_lower == "git" else 0.0
        cmd_type_cat = 1.0 if cmd_lower == "cat" else 0.0
        cmd_type_sed = 1.0 if cmd_lower == "sed" else 0.0
        cmd_type_other = 1.0 if cmd_lower not in ("grep", "find", "python", "python3", "git", "cat", "sed") else 0.0

        has_error = 1.0 if ("Error" in observation or "Traceback" in observation) else 0.0
        output_lines = observation.count("\n") if observation else 0
        output_lines_norm = min(output_lines / 100.0, 1.0)

        is_file_producing = 0.0
        if cmd_lower in ("grep", "find", "rg", "ag"):
            is_file_producing = 1.0

        return [
            cmd_type_grep,
            cmd_type_find,
            cmd_type_python,
            cmd_type_git,
            cmd_type_cat,
            cmd_type_sed,
            cmd_type_other,
            has_error,
            output_lines_norm,
            is_file_producing,
        ]

    def _test_specific_features(self, observation: str) -> list[float]:
        """6 test-specific features matching training."""
        has_failures = 1.0 if "FAILED" in observation or "failed" in observation else 0.0
        last_line = observation.split("\n")[-1] if observation else ""
        has_errors = 1.0 if ("ERROR" in observation or "error" in last_line) else 0.0
        all_passed = 1.0 if ("passed" in observation and "failed" not in observation.lower()) else 0.0

        # Approximate test count from pytest output
        num_tests = 0
        for line in (observation or "").splitlines():
            m = re.search(r"(\d+) passed", line)
            if m:
                num_tests += int(m.group(1))
            m = re.search(r"(\d+) failed", line)
            if m:
                num_tests += int(m.group(1))
        num_tests_norm = min(num_tests / 50.0, 1.0)

        # steps_since_last_edit
        last_edit_step = -1
        for i, tc in enumerate(reversed(self.session_tool_calls)):
            if tc.get("tool_type") == "edit":
                last_edit_step = self.step_count - i - 1
                break
        steps_since_edit = float(self.step_count - last_edit_step) if last_edit_step >= 0 else float(self.step_count)
        steps_since_edit_norm = min(steps_since_edit, 30) / 30.0

        # edits_since_last_test
        edits_since_test = 0
        for tc in reversed(self.session_tool_calls):
            if tc.get("tool_type") == "test":
                break
            if tc.get("tool_type") == "edit":
                edits_since_test += 1
        edits_since_test_norm = min(edits_since_test / 5.0, 1.0)

        return [
            has_failures,
            has_errors,
            all_passed,
            num_tests_norm,
            steps_since_edit_norm,
            edits_since_test_norm,
        ]

    def _search_specific_features(self, args: dict, observation: str) -> list[float]:
        """3 search-specific features matching training."""
        # search_num_results: count lines that look like file paths
        result_lines = [
            l
            for l in (observation or "").splitlines()
            if l.strip() and ("/" in l or "\\" in l) and not l.startswith(" ")
        ]
        num_results_norm = min(len(result_lines) / 20.0, 1.0)

        # search_results_visited: how many of the found files have been read
        visited = 0
        for line in result_lines:
            path_match = re.search(r"[\w./]+\.\w+", line)
            if path_match and path_match.group() in self.session_reads:
                visited += 1
        visited_norm = min(visited / 5.0, 1.0)

        # search_pattern_in_problem
        pattern = args.get("pattern", args.get("query", args.get("command", "")))
        pattern_in_problem = 0.0
        if pattern and self.problem_statement:
            if pattern.lower() in self.problem_statement.lower():
                pattern_in_problem = 1.0

        return [
            num_results_norm,
            visited_norm,
            pattern_in_problem,
        ]

    def _apply_tier(
        self,
        tool_type: str,
        tier: str,
        observation: str,
        file_path: str,
        args: dict,
    ) -> str:
        """Apply tier reduction to the full observation."""
        if tier == "full":
            return observation

        if tool_type == "read":
            return self._apply_read_tier(tier, observation, file_path)
        elif tool_type == "bash":
            return self._apply_bash_tier(tier, observation, args)
        elif tool_type == "test":
            return self._apply_test_tier(tier, observation, args)
        elif tool_type == "search":
            return self._apply_search_tier(tier, observation, args)
        else:
            return observation

    def _apply_read_tier(self, tier: str, observation: str, file_path: str) -> str:
        """Apply read tier reduction using observation_actions module."""
        if tier == "block":
            result = block_with_pointer(file_path, observation)
            return result.content
        elif tier == "summary":
            result = return_summary(file_path, observation)
            return result.content
        elif tier == "outline":
            result = return_ast_outline(file_path, observation)
            return result.content
        elif tier == "relevant_spans":
            result = return_relevant_spans(file_path, observation, self.problem_statement)
            return result.content
        return observation

    def _apply_bash_tier(self, tier: str, observation: str, args: dict) -> str:
        """Apply bash tier reduction."""
        lines = observation.splitlines()

        if tier == "exit_code":
            # Just return exit code / success/failure indicator
            has_error = "Error" in observation or "Traceback" in observation
            exit_indicator = "FAILED" if has_error else "OK"
            return f"[exit: {exit_indicator}] (output truncated, {len(lines)} lines)"

        elif tier == "first_error":
            # Return only the first error/traceback
            error_lines = []
            in_error = False
            for line in lines:
                if "Error" in line or "Traceback" in line or "FAILED" in line:
                    in_error = True
                if in_error:
                    error_lines.append(line)
                    if len(error_lines) >= 20:
                        break
                    if line.strip() == "" and len(error_lines) > 3:
                        break

            if error_lines:
                truncated = len(lines) - len(error_lines)
                return f"[first error — {truncated} lines omitted]\n" + "\n".join(error_lines)
            else:
                # No error found, return first few lines
                preview = "\n".join(lines[:5])
                return f"[no error detected — showing first 5/{len(lines)} lines]\n{preview}"

        elif tier == "structured":
            # Return structured output: errors + last N lines + summary
            parts = []
            # Collect errors
            error_lines = [l for l in lines if "Error" in l or "FAILED" in l or "Warning" in l]
            if error_lines:
                parts.append("Errors/Warnings:")
                parts.extend(f"  {l.strip()}" for l in error_lines[:10])
                parts.append("")

            # Last 10 lines (often contain summary)
            if len(lines) > 10:
                parts.append(f"[... {len(lines) - 10} lines omitted ...]")
            parts.extend(lines[-10:])

            return "\n".join(parts)

        return observation

    def _apply_test_tier(self, tier: str, observation: str, args: dict) -> str:
        """Apply test tier reduction."""
        lines = observation.splitlines()

        if tier == "exit_code":
            passed = "PASSED" in observation or "OK" in observation
            failed_count = observation.count("FAILED")
            passed_count = observation.count("PASSED") + observation.count("passed")
            status = "PASSED" if passed and failed_count == 0 else "FAILED"
            return f"[test result: {status}] passed={passed_count} failed={failed_count}"

        elif tier == "failing_names":
            # Extract just the names of failing tests
            failing = []
            for line in lines:
                # Common test output patterns
                match = re.match(r"^(?:FAIL|FAILED|ERROR)\s+(.+)", line.strip())
                if match:
                    failing.append(match.group(1).strip())
                # pytest format: path::TestClass::test_name FAILED
                match = re.match(r"^(.+::.+)\s+FAILED", line.strip())
                if match:
                    failing.append(match.group(1).strip())

            if failing:
                return f"[{len(failing)} failing tests]\n" + "\n".join(f"  FAIL: {name}" for name in failing[:20])
            else:
                has_error = "Error" in observation or "FAILED" in observation
                return f"[test result: {'FAILED' if has_error else 'PASSED'}] (could not parse test names)"

        elif tier == "first_failure":
            # Return the first test failure with its traceback
            failure_start = -1
            for i, line in enumerate(lines):
                if "FAILED" in line or "ERROR" in line or "FAIL:" in line:
                    failure_start = i
                    break

            if failure_start >= 0:
                # Grab context around the failure (up to 30 lines)
                end = min(failure_start + 30, len(lines))
                for j in range(failure_start + 1, end):
                    if lines[j].strip().startswith(("FAILED", "PASSED", "ERROR", "=" * 10)):
                        end = j
                        break
                failure_block = "\n".join(lines[failure_start:end])
                remaining = sum(1 for l in lines[end:] if "FAILED" in l or "ERROR" in l)
                suffix = f"\n[{remaining} additional failures not shown]" if remaining > 0 else ""
                return f"[first failure]\n{failure_block}{suffix}"
            else:
                return "\n".join(lines[-15:])

        elif tier == "all_failures":
            # Return all failures but strip passing test output
            failure_blocks = []
            current_block = []
            in_failure = False

            for line in lines:
                if "FAILED" in line or "ERROR" in line or "FAIL:" in line:
                    if current_block and not in_failure:
                        current_block = []
                    in_failure = True
                if in_failure:
                    current_block.append(line)
                    if line.strip() == "" and len(current_block) > 3:
                        failure_blocks.append("\n".join(current_block))
                        current_block = []
                        in_failure = False

            if current_block:
                failure_blocks.append("\n".join(current_block))

            if failure_blocks:
                return "\n---\n".join(failure_blocks[:10])
            else:
                # Fallback: last 30 lines
                return "\n".join(lines[-30:])

        return observation

    def _apply_search_tier(self, tier: str, observation: str, args: dict) -> str:
        """Apply search tier reduction."""
        lines = observation.splitlines()

        if tier == "count":
            # Just the count of results
            result_count = len([l for l in lines if l.strip()])
            return f"[{result_count} results found] (details omitted)"

        elif tier == "filenames":
            # Extract unique filenames from results
            filenames = set()
            for line in lines:
                # Common patterns: path/to/file.py:123: ...
                match = re.match(r"^([^\s:]+\.\w+)", line.strip())
                if match:
                    filenames.add(match.group(1))
                # find output: just paths
                if "/" in line and not ":" in line.split("/")[-1]:
                    filenames.add(line.strip())

            if filenames:
                sorted_files = sorted(filenames)
                return f"[{len(sorted_files)} files matched]\n" + "\n".join(f"  {f}" for f in sorted_files[:30])
            else:
                return f"[{len(lines)} results] (could not parse filenames)"

        elif tier == "top_matches":
            # Return first N results (most relevant)
            max_results = 10
            non_empty = [l for l in lines if l.strip()]
            shown = non_empty[:max_results]
            remaining = len(non_empty) - max_results
            result = "\n".join(shown)
            if remaining > 0:
                result += f"\n[... {remaining} more results]"
            return result

        return observation

    def _update_state(
        self,
        tool: str,
        args: dict,
        observation: str,
        file_path: str,
        decision: AllocationDecision,
    ):
        """Update session state after a tool call."""
        tool_type = self._classify_tool(tool, args)
        call_key = f"{tool}:{hashlib.md5(str(sorted(args.items())).encode()).hexdigest()[:12]}"

        self.session_tool_calls.append(
            {
                "tool": tool,
                "tool_type": tool_type,
                "file_path": file_path,
                "call_key": call_key,
                "tier_served": decision.selected_tier,
            }
        )

        if tool_type == "read" and file_path:
            self.session_reads.append(file_path)
            obs_hash = hashlib.sha256(observation.encode()).hexdigest()[:16]
            self.file_hashes[file_path] = obs_hash
            self.consecutive_reads_without_edit += 1

        if tool_type == "search":
            self.consecutive_reads_without_edit += 1

        # Track failures
        if "Error" in observation or "FAILED" in observation or "Traceback" in observation:
            self.recent_failures = min(self.recent_failures + 1, 5)
        elif "PASSED" in observation or "OK" in observation:
            self.recent_failures = max(self.recent_failures - 1, 0)

        self.step_count += 1

    def _generate_escalation_hint(self, tool_type: Optional[str], tier: str) -> str:
        """Generate a hint for the agent on how to escalate if needed."""
        hints = {
            "read": {
                "block": "Use read_full(path, reason='...') to override, or read_spans(path, symbols=[...]) for targeted access.",
                "summary": "Use read_outline(path) for structure, or read_full(path, reason='...') for complete content.",
                "outline": "Use read_spans(path, symbols=['ClassName', 'function_name']) for specific sections.",
                "relevant_spans": "Use read_full(path, reason='need complete context') if shown spans are insufficient.",
            },
            "bash": {
                "exit_code": "Re-run with verbose flag or check specific output sections.",
                "first_error": "If more context needed, re-run or request structured output.",
                "structured": "Use full output if structured view is missing critical information.",
            },
            "test": {
                "exit_code": "Run with -v flag for test names, or re-run for full output.",
                "failing_names": "Re-run to see full failure tracebacks.",
                "first_failure": "Request all_failures tier if multiple failures need investigation.",
                "all_failures": "Request full output if passing test output is also needed.",
            },
            "search": {
                "count": "Re-run search to see filenames or full matches.",
                "filenames": "Re-run to see matching lines and context.",
                "top_matches": "Request full results if relevant matches might be beyond top 10.",
            },
        }

        tool_hints = hints.get(tool_type, {})
        return tool_hints.get(tier, "Request full tier for complete output.")


# =============================================================================
# Demo
# =============================================================================


def demo():
    """Demo the per-tier risk allocation on a synthetic scenario."""
    print("=" * 70)
    print("OBSERVATION ALLOCATOR — Per-Tier Risk Prediction Demo")
    print("=" * 70)

    allocator = ObservationAllocator(fallback_to_heuristic=True)
    interceptor = AgentObservationInterceptor(allocator)
    interceptor.problem_statement = "QuerySet.order_by() generates incorrect SQL with transforms"

    # Simulate scenarios with different tool types
    scenarios = [
        {
            "tool": "read_file",
            "args": {"file_path": "django/db/models/sql/compiler.py"},
            "observation": "class SQLCompiler:\n    def get_order_by(self):\n        ...\n" * 50,
            "reasoning": "The bug is about SQL generation in order_by. The compiler handles this.",
            "desc": "First read of likely edit target",
        },
        {
            "tool": "read_file",
            "args": {"file_path": "django/contrib/admin/views/main.py"},
            "observation": "class ChangeList:\n    def get_ordering(self):\n        ...\n" * 30,
            "reasoning": "Let me also check if admin uses order_by somewhere.",
            "desc": "Exploratory read (not edit target)",
        },
        {
            "tool": "grep",
            "args": {"pattern": "order_by", "path": "django/"},
            "observation": "\n".join(f"django/file{i}.py:42: order_by(...)" for i in range(25)),
            "reasoning": "Let me search for all uses of order_by.",
            "desc": "Search (first time)",
        },
        {
            "tool": "bash",
            "args": {"command": "python -m pytest tests/queries/test_ordering.py -v"},
            "observation": "FAILED tests/queries/test_ordering.py::TestOrderBy::test_transforms\nAssertionError: ...\nPASSED 14 tests\nFAILED 1 test",
            "reasoning": "Let me run the failing test to confirm the bug.",
            "desc": "Test execution",
        },
        {
            "tool": "read_file",
            "args": {"file_path": "django/db/models/sql/compiler.py"},
            "observation": "class SQLCompiler:\n    def get_order_by(self):\n        ...\n" * 50,
            "reasoning": "I need to re-read compiler.py after my failed fix attempt.",
            "desc": "Re-read (hash unchanged)",
        },
    ]

    print(f"\nProblem: {interceptor.problem_statement}")
    print(f"\n{'Step':<5} {'Description':<40} {'Tool':<12} {'Tier':<18} {'Compression':<12}")
    print("-" * 90)

    for i, scenario in enumerate(scenarios):
        result = interceptor.intercept(
            tool=scenario["tool"],
            args=scenario["args"],
            observation=scenario["observation"],
            reasoning=scenario["reasoning"],
            problem=interceptor.problem_statement,
        )

        comp_str = f"{result.compression_ratio:.0%}"
        print(f"{i:<5} {scenario['desc']:<40} {scenario['tool']:<12} {result.tier:<18} {comp_str:<12}")
        if result.escalation_hint:
            print(f"      Escalation: {result.escalation_hint}")

    print(f"\n{'=' * 90}")
    print(f"Total steps: {interceptor.step_count}")
    print(f"Files read: {len(interceptor.session_reads)}")
    print(f"Tiers used by the heuristic allocator (no trained model loaded)")


if __name__ == "__main__":
    demo()
