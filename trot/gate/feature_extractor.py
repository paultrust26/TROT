"""Feature Extractor — compute 38-dim feature vector for each read event.

Takes a trace + step index and produces the full feature vector needed for
the Stage 2 action-risk model. Designed for offline batch extraction over
collected traces.

Feature groups:
  - Content: model2vec cosine similarities (task↔file, task↔path)
  - Session: read history, edit history, progress
  - History: accumulated state
  - Risk: signals that a read is high-priority
  - Availability: what actions are possible
  - Context: surrounding task state
  - Cost: token estimates
  - Exploration budget: adaptive budget features
"""

import hashlib
import re
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from model2vec import StaticModel

    HAS_MODEL2VEC = True
except ImportError:
    HAS_MODEL2VEC = False


@dataclass
class ReadFeatures:
    """Full 43-feature vector for a single read event."""

    # Content features (2)
    task_file_cosine: float = 0.0
    task_path_cosine: float = 0.0

    # Session state features (4)
    times_read_this_file: int = 0
    steps_since_last_edit: int = 0
    session_progress: float = 0.0
    file_changed_since_last_read: bool = False

    # History features (2)
    files_already_read_count: int = 0
    prior_reads_utility_score: float = 0.0

    # Risk features (4)
    file_in_traceback: bool = False
    file_in_import_graph: bool = False
    first_read_of_likely_edit_target: bool = False
    file_is_small: bool = False

    # Availability mask features (3)
    has_prior_observation: bool = False
    has_recent_edit: bool = False
    hash_unchanged_since_last_read: bool = False

    # Task context features (3)
    estimated_edit_target: bool = False
    tests_failing: bool = False
    agent_in_editing_phase: bool = False

    # Cost features (3)
    file_line_count: int = 0
    estimated_full_tokens: int = 0
    estimated_span_tokens: int = 0

    # Exploration budget features (8)
    budget_remaining: float = 0.0
    budget_fraction_used: float = 0.0
    explore_exploit_ratio: float = 0.0
    reads_since_last_edit: int = 0
    steps_without_productive_action: int = 0
    cumulative_ig_estimate: float = 0.0
    task_complexity_estimate: float = 0.0
    predicted_remaining_edits: float = 0.0

    # Redundancy detection features (5)
    lines_already_observed_fraction: float = 0.0
    identical_read_within_3: bool = False
    steps_since_lines_last_seen: int = 0
    bash_read_of_same_file: bool = False
    overlapping_reads_of_file: int = 0

    # Derived features (7)
    position_normalized: float = 0.0
    is_python_file: bool = False
    is_test_file: bool = False
    file_path_matches_problem_keywords: bool = False
    reasoning_length_normalized: float = 0.0
    was_mentioned_in_prior_search: bool = False
    num_files_already_edited: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        values = []
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool):
                values.append(1.0 if v else 0.0)
            else:
                values.append(float(v))
        return np.array(values, dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def feature_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def num_features(cls) -> int:
        return len(fields(cls))


class Embedder:
    """Wraps model2vec for fast cosine similarity computation."""

    def __init__(self, model_name: str = "minishlab/potion-base-2M"):
        self._model = None
        self._dim = 64
        self._cache: dict[str, np.ndarray] = {}

        if not HAS_MODEL2VEC:
            return
        try:
            self._model = StaticModel.from_pretrained(model_name)
        except Exception:
            pass

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding, with caching."""
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._model is None:
            emb = np.random.randn(self._dim).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
        else:
            emb = self._model.encode([text[:2000]])[0]

        self._cache[cache_key] = emb
        return emb

    def cosine(self, text_a: str, text_b: str) -> float:
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        dot = float(np.dot(emb_a, emb_b))
        norm = float(np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        return dot / max(norm, 1e-8)


class FeatureExtractor:
    """Extract features for a read event given trace context.

    Usage:
        extractor = FeatureExtractor(embedder=Embedder())
        features = extractor.extract(trace, step_idx)
    """

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()

    def extract(self, trace: dict, step_idx: int) -> ReadFeatures:
        """Extract features for the read at step_idx in the trace."""
        steps = trace["steps"]
        current_step = steps[step_idx]
        task_desc = trace.get("problem_statement", trace.get("task_id", ""))
        file_path = current_step.get("arguments", {}).get("file_path", "")
        total_steps = len(steps)

        # Build session state up to this point
        state = self._build_state(steps[:step_idx], file_path)

        features = ReadFeatures()

        # Content features
        features.task_file_cosine = self.embedder.cosine(task_desc, file_path)
        file_content = current_step.get("observation", "")
        if file_content:
            content_preview = file_content[:500]
            features.task_path_cosine = self.embedder.cosine(task_desc, content_preview)
        else:
            features.task_path_cosine = features.task_file_cosine * 0.8

        # Session state
        features.times_read_this_file = state["times_read"].get(file_path, 0)
        features.steps_since_last_edit = (
            step_idx - state["last_edit_step"] if state["last_edit_step"] >= 0 else step_idx
        )
        features.session_progress = step_idx / max(total_steps - 1, 1)
        features.file_changed_since_last_read = state["file_changed"].get(file_path, False)

        # History
        features.files_already_read_count = len(state["files_read"])
        features.prior_reads_utility_score = state["utility_running_avg"]

        # Risk
        features.file_in_traceback = self._file_in_traceback(file_path, task_desc)
        features.file_in_import_graph = file_path in state["import_graph_files"]
        features.first_read_of_likely_edit_target = state["times_read"].get(
            file_path, 0
        ) == 0 and self._is_likely_edit_target(file_path, task_desc)
        file_line_count = file_content.count("\n") + 1 if file_content else 0
        features.file_is_small = file_line_count < 50

        # Availability
        features.has_prior_observation = file_path in state["observed_hashes"]
        features.has_recent_edit = state["last_edit_file"] == file_path and state["last_edit_step"] == step_idx - 1
        current_hash = hashlib.sha256(file_content.encode()).hexdigest()[:16] if file_content else ""
        features.hash_unchanged_since_last_read = (
            state["observed_hashes"].get(file_path, "") == current_hash and current_hash != ""
        )

        # Task context (inference-safe: no ground-truth patch)
        features.estimated_edit_target = self._estimate_edit_target(file_path, task_desc, state)
        features.tests_failing = state["tests_failing"]
        features.agent_in_editing_phase = state["has_edited"]

        # Cost
        features.file_line_count = file_line_count
        features.estimated_full_tokens = len(file_content) // 4 + 1 if file_content else 0
        features.estimated_span_tokens = min(features.estimated_full_tokens, 1500)

        # Exploration budget
        budget_initial = self._estimate_initial_budget(trace)
        exploration_reads_so_far = sum(
            1 for f in state["files_read"] if not state["files_edited"] or f not in state["files_edited"]
        )
        features.budget_remaining = max(0, budget_initial - exploration_reads_so_far)
        features.budget_fraction_used = exploration_reads_so_far / max(budget_initial, 1)

        edits_so_far = len(state["files_edited"])
        reads_so_far = len(state["files_read"])
        features.explore_exploit_ratio = reads_so_far / max(edits_so_far, 1)
        features.reads_since_last_edit = (
            step_idx - state["last_edit_step"] if state["last_edit_step"] >= 0 else reads_so_far
        )
        features.steps_without_productive_action = state["steps_without_productive"]
        features.cumulative_ig_estimate = state["cumulative_ig"]
        features.task_complexity_estimate = self._task_complexity(trace)
        features.predicted_remaining_edits = max(0, features.task_complexity_estimate - edits_so_far)

        # Redundancy detection features
        current_start = current_step.get("arguments", {}).get("start_line", 1)
        current_end = current_step.get("arguments", {}).get("end_line", 0)
        if current_end == 0 and file_content:
            current_end = file_content.count("\n") + 1

        prior_ranges = state["line_ranges_read"].get(file_path, [])
        if prior_ranges and current_end > 0:
            # Compute what fraction of requested lines were already observed
            requested_lines = set(range(current_start, current_end + 1))
            observed_lines = set()
            for start, end, _ in prior_ranges:
                observed_lines.update(range(start, end + 1))
            overlap = requested_lines & observed_lines
            features.lines_already_observed_fraction = len(overlap) / max(len(requested_lines), 1)

            # Steps since these lines were last seen
            last_seen_step = -1
            for start, end, s_idx in prior_ranges:
                range_lines = set(range(start, end + 1))
                if range_lines & requested_lines:
                    last_seen_step = max(last_seen_step, s_idx)
            features.steps_since_lines_last_seen = step_idx - last_seen_step if last_seen_step >= 0 else step_idx

            # Count how many prior reads overlap with this one
            features.overlapping_reads_of_file = sum(
                1 for start, end, _ in prior_ranges if set(range(start, end + 1)) & requested_lines
            )
        else:
            features.lines_already_observed_fraction = 0.0
            features.steps_since_lines_last_seen = step_idx
            features.overlapping_reads_of_file = 0

        # Identical read within last 3 steps
        read_key = (file_path, current_start, current_end)
        prior_exact = state["exact_reads"].get(read_key, [])
        features.identical_read_within_3 = any(step_idx - s_idx <= 3 for s_idx in prior_exact)

        # Bash-based read of same file (agent bypassing read_file)
        file_basename = Path(file_path).name
        features.bash_read_of_same_file = (
            file_path in state["bash_read_files"]
            or file_basename in state["bash_read_files"]
            or any(file_path.endswith(bf.lstrip("./")) for bf in state["bash_read_files"] if bf)
        )

        # Derived features
        features.position_normalized = features.session_progress
        features.is_python_file = file_path.endswith(".py")
        features.is_test_file = "test" in file_path.lower()
        features.file_path_matches_problem_keywords = features.task_file_cosine > 0.3
        reasoning = current_step.get("reasoning_before", current_step.get("reasoning", ""))
        features.reasoning_length_normalized = len(reasoning) / 300.0 if reasoning else 0.0
        features.was_mentioned_in_prior_search = file_path in state["search_mentioned_files"]
        features.num_files_already_edited = len(state["files_edited"])

        return features

    def extract_batch(self, trace: dict) -> list[tuple[int, ReadFeatures]]:
        """Extract features for all read events in a trace."""
        results = []
        for i, step in enumerate(trace["steps"]):
            if step.get("tool") in ("read_file", "view_file", "str_replace_editor"):
                if step.get("tool") == "str_replace_editor":
                    cmd = step.get("arguments", {}).get("command", "")
                    if cmd != "view":
                        continue
                results.append((i, self.extract(trace, i)))
        return results

    def _build_state(self, prior_steps: list[dict], current_file: str) -> dict:
        """Build accumulated session state from steps before current."""
        state = {
            "files_read": set(),
            "files_edited": set(),
            "times_read": {},
            "observed_hashes": {},
            "file_changed": {},
            "last_edit_step": -1,
            "last_edit_file": "",
            "has_edited": False,
            "tests_failing": False,
            "import_graph_files": set(),
            "search_mentioned_files": set(),
            "utility_running_avg": 0.5,
            "steps_without_productive": 0,
            "cumulative_ig": 0.0,
            # Line-range tracking: file → [(start, end, step_idx)]
            "line_ranges_read": {},
            # Exact read tracking: (file, start, end) → [step_indices]
            "exact_reads": {},
            # Bash-based file reads: files read via open() in bash commands
            "bash_read_files": set(),
        }

        utility_scores = []
        last_productive_step = -1

        for i, step in enumerate(prior_steps):
            tool = step.get("tool", "")
            args = step.get("arguments", {})
            obs = step.get("observation", "")
            file_path = args.get("file_path", "")

            if tool in ("read_file", "view_file"):
                state["files_read"].add(file_path)
                state["times_read"][file_path] = state["times_read"].get(file_path, 0) + 1

                obs_hash = hashlib.sha256(obs.encode()).hexdigest()[:16] if obs else ""
                prev_hash = state["observed_hashes"].get(file_path, "")
                state["file_changed"][file_path] = prev_hash != "" and prev_hash != obs_hash
                state["observed_hashes"][file_path] = obs_hash

                # Track line ranges
                start_line = args.get("start_line", 1)
                end_line = args.get("end_line", 0)
                if end_line == 0 and obs:
                    end_line = obs.count("\n") + 1
                if file_path not in state["line_ranges_read"]:
                    state["line_ranges_read"][file_path] = []
                state["line_ranges_read"][file_path].append((start_line, end_line, i))

                # Track exact reads
                read_key = (file_path, start_line, end_line)
                if read_key not in state["exact_reads"]:
                    state["exact_reads"][read_key] = []
                state["exact_reads"][read_key].append(i)

            elif tool in ("edit_file", "str_replace_editor", "write_file"):
                cmd = args.get("command", "")
                if tool == "str_replace_editor" and cmd == "view":
                    state["files_read"].add(file_path)
                    state["times_read"][file_path] = state["times_read"].get(file_path, 0) + 1
                else:
                    state["files_edited"].add(file_path)
                    state["last_edit_step"] = i
                    state["last_edit_file"] = file_path
                    state["has_edited"] = True
                    last_productive_step = i

                    # Track imports of edited files
                    if obs:
                        imports = re.findall(r"from\s+([\w.]+)\s+import|import\s+([\w.]+)", obs)
                        for groups in imports:
                            for imp in groups:
                                if imp:
                                    state["import_graph_files"].add(imp.replace(".", "/") + ".py")

            elif tool in ("bash", "run_command", "execute_bash"):
                cmd_text = args.get("command", "")
                # Detect bash-based file reads: open('file'), cat file, etc.
                bash_file_reads = re.findall(r"open\(['\"]([^'\"]+)['\"]", cmd_text)
                bash_file_reads += re.findall(r"\bcat\s+['\"]?([^\s'\"]+\.\w+)", cmd_text)
                state["bash_read_files"].update(bash_file_reads)

                if obs and ("FAILED" in obs or "Error" in obs or "Traceback" in obs):
                    state["tests_failing"] = True
                    last_productive_step = i
                elif obs and ("PASSED" in obs or "OK" in obs):
                    state["tests_failing"] = False
                    last_productive_step = i

            elif tool in ("grep", "find", "search", "find_file"):
                if obs:
                    file_mentions = re.findall(r"[\w/]+\.(?:py|js|ts|java|go|rs|rb)\b", obs)
                    state["search_mentioned_files"].update(file_mentions)
                last_productive_step = i

            # Utility tracking: did prior reads lead to edits?
            if tool in ("read_file", "view_file") and file_path:
                was_useful = file_path in state["files_edited"]
                utility_scores.append(1.0 if was_useful else 0.0)

            # Information gain estimate: new file = high IG, re-read unchanged = low IG
            if tool in ("read_file", "view_file"):
                if state["times_read"].get(file_path, 0) <= 1:
                    state["cumulative_ig"] += 1.0
                elif state["file_changed"].get(file_path, False):
                    state["cumulative_ig"] += 0.5
                else:
                    state["cumulative_ig"] += 0.1

        state["utility_running_avg"] = np.mean(utility_scores) if utility_scores else 0.5
        state["steps_without_productive"] = (
            len(prior_steps) - 1 - last_productive_step if last_productive_step >= 0 else len(prior_steps)
        )

        return state

    def _file_in_traceback(self, file_path: str, task_desc: str) -> bool:
        """Check if file appears in a traceback in the task description."""
        if "Traceback" not in task_desc and "Error" not in task_desc:
            return False
        basename = Path(file_path).name
        return basename in task_desc or file_path in task_desc

    def _is_likely_edit_target(self, file_path: str, task_desc: str) -> bool:
        """Heuristic: is this file likely to need editing?"""
        basename = Path(file_path).name
        stem = Path(file_path).stem
        task_lower = task_desc.lower()
        return (
            basename in task_desc
            or stem.replace("_", " ") in task_lower
            or stem.replace("_", "") in task_lower.replace(" ", "")
        )

    def _estimate_edit_target(self, file_path: str, task_desc: str, state: dict) -> bool:
        """Inference-safe estimate: is this file likely to be edited?"""
        if file_path in state["files_edited"]:
            return True
        if self._file_in_traceback(file_path, task_desc):
            return True
        if self._is_likely_edit_target(file_path, task_desc):
            return True
        if file_path in state["search_mentioned_files"]:
            basename = Path(file_path).stem
            task_lower = task_desc.lower()
            if basename.replace("_", " ") in task_lower or basename in task_lower:
                return True
        return False

    def _estimate_initial_budget(self, trace: dict) -> float:
        """Estimate exploration budget from task description (no GT patch)."""
        task_desc = trace.get("problem_statement", trace.get("task_id", ""))
        file_refs = set(re.findall(r"[\w/]+\.(?:py|js|ts|java|go|rs|rb)\b", task_desc))
        traceback_count = task_desc.count("Traceback")
        desc_length = len(task_desc)
        estimated_files = max(len(file_refs), 1 + traceback_count)
        if desc_length > 3000:
            estimated_files = max(estimated_files, 3)
        return max(5, estimated_files * 4)

    def _task_complexity(self, trace: dict) -> float:
        """Estimate task complexity from description signals (no GT patch)."""
        task_desc = trace.get("problem_statement", trace.get("task_id", ""))
        identifiers = set(re.findall(r"\b[A-Z][a-z]+[A-Z]\w+\b", task_desc))  # CamelCase
        identifiers.update(re.findall(r"\b[a-z]+_[a-z_]+\b", task_desc))  # snake_case
        file_refs = set(re.findall(r"[\w/]+\.(?:py|js|ts|java|go|rs|rb)\b", task_desc))
        traceback_count = task_desc.count("Traceback")
        desc_tokens = len(task_desc) // 4
        complexity = len(file_refs) + len(identifiers) / 10.0 + traceback_count + desc_tokens / 500.0
        return max(1.0, min(complexity, 20.0))


def extract_features_for_trace(
    trace: dict,
    embedder: Optional[Embedder] = None,
) -> list[dict]:
    """Extract features for all reads in a trace. Returns list of feature dicts."""
    extractor = FeatureExtractor(embedder=embedder)
    results = []
    for step_idx, features in extractor.extract_batch(trace):
        step = trace["steps"][step_idx]
        results.append(
            {
                "task_id": trace.get("task_id", ""),
                "step_idx": step_idx,
                "file_path": step.get("arguments", {}).get("file_path", ""),
                "features": features.to_dict(),
                "feature_vector": features.to_array().tolist(),
            }
        )
    return results
