"""Observation Action Generators — produce reduced-fidelity observations for gating.

Each action takes full file content and returns a reduced version.
These are used both at inference time (gating live reads) and for
counterfactual replay (generating tier labels).

Actions ordered by expected cost (ascending):
  block_with_pointer  < return_edit_receipt < return_summary
  < return_ast_outline < return_diff < return_relevant_spans
  < return_cached_view < allow_full
"""

import ast
import hashlib
import re
import textwrap
from dataclasses import dataclass, field
from difflib import unified_diff
from typing import Optional


@dataclass
class ReducedObservation:
    action: str
    content: str
    tokens_estimate: int
    original_tokens: int
    compression_ratio: float
    available_escalations: list[str] = field(default_factory=list)


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def block_with_pointer(file_path: str, file_content: str) -> ReducedObservation:
    """Most aggressive: block entirely, return only metadata."""
    line_count = file_content.count("\n") + 1
    ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""

    # Extract top-level definitions for context
    defs = []
    for line in file_content.splitlines()[:200]:
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "async def ")):
            defs.append(stripped.split("(")[0].split(":")[0])
    defs_preview = ", ".join(defs[:10])
    if len(defs) > 10:
        defs_preview += f" ... (+{len(defs) - 10} more)"

    msg = (
        f"[Observation compressed — this file's full content was predicted as low utility for your current task.]\n"
        f"File: {file_path} ({line_count} lines, .{ext})\n"
        f"Contains: {defs_preview or 'no top-level defs found'}\n"
        f"Hash: {content_hash(file_content)}\n"
        f"NOTE: You already have or can infer what you need. If you specifically need this file's content, "
        f"use a targeted view with line range (e.g. view_range=[1,50]) or search for specific symbols."
    )

    original_tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="block_with_pointer",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=[
            "read_full(path, reason='...')",
            "read_spans(path, symbols=[...])",
            "read_outline(path)",
        ],
    )


def return_edit_receipt(
    file_path: str,
    file_content: str,
    edit_old: str,
    edit_new: str,
    syntax_ok: bool = True,
) -> ReducedObservation:
    """Post-edit verification: show what changed, not full file."""
    line_count = file_content.count("\n") + 1
    diff_lines = list(
        unified_diff(
            edit_old.splitlines(keepends=True),
            edit_new.splitlines(keepends=True),
            fromfile="before",
            tofile="after",
            lineterm="",
            n=2,
        )
    )
    diff_text = "\n".join(diff_lines[:50])

    changed_lines = sum(1 for l in diff_lines if l.startswith("+") or l.startswith("-"))
    status = "OK" if syntax_ok else "SYNTAX ERROR"

    msg = (
        f"[Edit receipt — verification read blocked]\n"
        f"File: {file_path} ({line_count} lines)\n"
        f"Status: {status}\n"
        f"Changed {changed_lines} lines:\n"
        f"{diff_text}\n"
        f"Hash: {content_hash(file_content)}"
    )

    original_tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="return_edit_receipt",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=[
            "read_full(path, reason='need to verify surrounding context')",
        ],
    )


def return_summary(file_path: str, file_content: str) -> ReducedObservation:
    """Return a short structural summary: classes, functions, imports."""
    line_count = file_content.count("\n") + 1
    ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""

    classes = []
    functions = []
    imports = []

    if ext == "py":
        try:
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree)):
                        functions.append(node.name)
                    else:
                        functions.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except SyntaxError:
            classes = re.findall(r"^class\s+(\w+)", file_content, re.MULTILINE)
            functions = re.findall(r"^def\s+(\w+)", file_content, re.MULTILINE)
    else:
        classes = re.findall(r"^class\s+(\w+)", file_content, re.MULTILINE)
        functions = re.findall(r"^(?:def|function|func|fn)\s+(\w+)", file_content, re.MULTILINE)

    parts = [f"[Summary — reduced observation]\nFile: {file_path} ({line_count} lines)"]
    if classes:
        parts.append(f"Classes: {', '.join(classes[:20])}")
    if functions:
        parts.append(f"Functions: {', '.join(functions[:30])}")
    if imports:
        parts.append(f"Key imports: {', '.join(imports[:15])}")

    msg = "\n".join(parts)
    original_tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="return_summary",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=[
            "read_outline(path)",
            "read_spans(path, symbols=[...])",
            "read_full(path, reason='...')",
        ],
    )


def return_ast_outline(file_path: str, file_content: str) -> ReducedObservation:
    """Return function/class signatures with docstrings, no bodies."""
    ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
    line_count = file_content.count("\n") + 1

    if ext == "py":
        outline = _python_ast_outline(file_content)
    else:
        outline = _regex_outline(file_content, ext)

    if not outline:
        outline = _fallback_outline(file_content)

    header = f"[AST outline — reduced observation]\nFile: {file_path} ({line_count} lines)\n\n"
    msg = header + outline

    original_tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="return_ast_outline",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=[
            "read_spans(path, symbols=[...])",
            "read_full(path, reason='...')",
        ],
    )


def _python_ast_outline(source: str) -> str:
    """Extract Python class/function signatures with docstrings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _regex_outline(source, "py")

    lines = source.splitlines()
    parts = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            parts.append(_format_class_outline(node, lines))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            parts.append(_format_function_outline(node, lines))
        elif isinstance(node, ast.Import | ast.ImportFrom):
            parts.append(ast.get_source_segment(source, node) or "")

    return "\n\n".join(p for p in parts if p)


def _format_function_outline(node: ast.FunctionDef, lines: list[str]) -> str:
    """Format a function node as signature + docstring."""
    decorators = ""
    for dec in node.decorator_list:
        dec_line = lines[dec.lineno - 1].strip() if dec.lineno <= len(lines) else ""
        decorators += dec_line + "\n"

    sig_start = node.lineno - 1
    sig_end = sig_start
    for i in range(sig_start, min(sig_start + 5, len(lines))):
        if ":" in lines[i]:
            sig_end = i
            break
    sig = "\n".join(lines[sig_start : sig_end + 1])

    docstring = ast.get_docstring(node)
    doc_part = ""
    if docstring:
        first_line = docstring.split("\n")[0]
        doc_part = f'    """{first_line}"""'

    return f"{decorators}{sig}\n{doc_part}    ..." if doc_part else f"{decorators}{sig}\n    ..."


def _format_class_outline(node: ast.ClassDef, lines: list[str]) -> str:
    """Format a class node as signature + method signatures."""
    sig_line = lines[node.lineno - 1] if node.lineno <= len(lines) else f"class {node.name}:"

    docstring = ast.get_docstring(node)
    doc_part = ""
    if docstring:
        first_line = docstring.split("\n")[0]
        doc_part = f'    """{first_line}"""'

    methods = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
            method_sig = lines[item.lineno - 1].strip() if item.lineno <= len(lines) else ""
            methods.append(f"    {method_sig} ...")

    body = "\n".join(filter(None, [doc_part] + methods[:20]))
    return f"{sig_line}\n{body}"


def _regex_outline(content: str, ext: str) -> str:
    """Fallback: regex-based outline for non-Python files."""
    lines = content.splitlines()
    outline_lines = []

    patterns = {
        "js": r"^(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+\w+",
        "ts": r"^(?:export\s+)?(?:async\s+)?(?:function|class|interface|type|const|let)\s+\w+",
        "java": r"^(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|\w+\s+\w+\s*\()",
        "go": r"^(?:func|type|var|const)\s+",
        "rs": r"^(?:pub\s+)?(?:fn|struct|enum|trait|impl|mod|type)\s+",
        "rb": r"^(?:class|module|def)\s+",
    }
    pattern = patterns.get(ext, r"^(?:class|def|function|func|fn|pub|export)\s+")

    for i, line in enumerate(lines):
        if re.match(pattern, line.strip()):
            outline_lines.append(f"L{i + 1}: {line.rstrip()}")

    return "\n".join(outline_lines) if outline_lines else ""


def _fallback_outline(content: str) -> str:
    """Last resort: first 10 lines + any line starting with class/def/function."""
    lines = content.splitlines()
    parts = lines[:5]
    parts.append("...")
    for i, line in enumerate(lines[5:], start=6):
        stripped = line.strip()
        if stripped and (
            stripped.startswith(("class ", "def ", "function ", "func ", "fn "))
            or stripped.startswith(("import ", "from ", "#include", "use "))
        ):
            parts.append(f"L{i}: {line.rstrip()}")
    return "\n".join(parts[:40])


def return_diff(
    file_path: str,
    file_content: str,
    prior_content: str,
) -> ReducedObservation:
    """Return only the diff from previously observed content."""
    line_count = file_content.count("\n") + 1

    diff_lines = list(
        unified_diff(
            prior_content.splitlines(keepends=True),
            file_content.splitlines(keepends=True),
            fromfile=f"{file_path} (previously observed)",
            tofile=f"{file_path} (current)",
            lineterm="",
            n=3,
        )
    )

    if not diff_lines:
        msg = (
            f"[No changes since last observation]\n"
            f"File: {file_path} ({line_count} lines)\n"
            f"Hash: {content_hash(file_content)} (unchanged)"
        )
    else:
        diff_text = "\n".join(diff_lines[:100])
        changed = sum(1 for l in diff_lines if l.startswith("+") or l.startswith("-"))
        msg = (
            f"[Diff from prior observation — {changed} lines changed]\n"
            f"File: {file_path} ({line_count} lines)\n\n"
            f"{diff_text}"
        )

    original_tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="return_diff",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=[
            "read_full(path, reason='need full context around changes')",
            "read_spans(path, symbols=[...])",
        ],
    )


def return_cached_view(file_path: str, file_content: str) -> ReducedObservation:
    """File unchanged since last read — return pointer."""
    line_count = file_content.count("\n") + 1
    msg = (
        f"[Cached — file unchanged since last observation]\n"
        f"File: {file_path} ({line_count} lines)\n"
        f"Hash: {content_hash(file_content)}\n"
        f"Content is identical to your previous observation of this file."
    )
    original_tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="return_cached_view",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=[
            "read_full(path, reason='need to re-examine content')",
        ],
    )


def return_relevant_spans(
    file_path: str,
    file_content: str,
    task_description: str,
    max_tokens: int = 1500,
    embedder=None,
) -> ReducedObservation:
    """Return only spans most relevant to the task description."""
    line_count = file_content.count("\n") + 1
    ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""

    if ext == "py":
        spans = _extract_python_spans(file_content)
    else:
        spans = _extract_generic_spans(file_content)

    if not spans:
        return return_ast_outline(file_path, file_content)

    ranked = _rank_spans_by_relevance(spans, task_description, embedder)

    selected = []
    token_budget = max_tokens
    for span_text, score in ranked:
        span_tokens = estimate_tokens(span_text)
        if span_tokens <= token_budget:
            selected.append(span_text)
            token_budget -= span_tokens
        if token_budget <= 0:
            break

    not_shown = len(spans) - len(selected)
    content_parts = [
        f"[Relevant spans — {len(selected)}/{len(spans)} sections shown]\nFile: {file_path} ({line_count} lines)\n"
    ]
    content_parts.extend(selected)
    if not_shown > 0:
        content_parts.append(f"\n[{not_shown} additional sections not shown]")

    msg = "\n\n".join(content_parts)
    original_tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="return_relevant_spans",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=[
            "read_more(symbol='SymbolName')",
            "read_full(path, reason='...')",
        ],
    )


def allow_full(file_path: str, file_content: str) -> ReducedObservation:
    """Pass through — full observation delivered."""
    tokens = estimate_tokens(file_content)
    return ReducedObservation(
        action="allow_full",
        content=file_content,
        tokens_estimate=tokens,
        original_tokens=tokens,
        compression_ratio=1.0,
        available_escalations=[],
    )


def _extract_python_spans(source: str) -> list[str]:
    """Split Python source into per-function/class spans."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _extract_generic_spans(source)

    lines = source.splitlines()
    spans = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else start + 20
            span_text = "\n".join(lines[start:end])
            spans.append(span_text)

    if not spans:
        return _extract_generic_spans(source)
    return spans


def _extract_generic_spans(content: str) -> list[str]:
    """Split content into paragraph-like spans by blank lines."""
    lines = content.splitlines()
    spans = []
    current = []

    for line in lines:
        if line.strip() == "" and current:
            spans.append("\n".join(current))
            current = []
        else:
            current.append(line)

    if current:
        spans.append("\n".join(current))

    if len(spans) <= 2:
        chunk_size = 30
        spans = []
        for i in range(0, len(lines), chunk_size):
            spans.append("\n".join(lines[i : i + chunk_size]))

    return spans


def _rank_spans_by_relevance(
    spans: list[str],
    task_description: str,
    embedder=None,
) -> list[tuple[str, float]]:
    """Rank spans by relevance to task. Uses keyword overlap as fallback."""
    if embedder is not None:
        try:
            task_emb = embedder.encode([task_description])
            span_embs = embedder.encode(spans)
            import numpy as np

            scores = np.dot(span_embs, task_emb.T).flatten()
            ranked = sorted(zip(spans, scores.tolist()), key=lambda x: -x[1])
            return ranked
        except Exception:
            pass

    task_words = set(task_description.lower().split())
    task_words = {w for w in task_words if len(w) > 3}

    scored = []
    for span in spans:
        span_words = set(span.lower().split())
        overlap = len(task_words & span_words)
        scored.append((span, overlap))

    return sorted(scored, key=lambda x: -x[1])


# ============================================================================
# Bash/Build Output Renderers
# ============================================================================


def bash_exit_code_only(command: str, stdout: str, stderr: str, exit_code: int) -> ReducedObservation:
    """Most compressed: just command + exit code."""
    status = "success" if exit_code == 0 else "failed"
    msg = f"$ {command}\nExit code: {exit_code} ({status})"
    original_tokens = estimate_tokens(stdout + stderr)
    return ReducedObservation(
        action="bash_exit_code_only",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["bash_first_error", "bash_structured_summary", "bash_full"],
    )


def bash_first_error(command: str, stdout: str, stderr: str, exit_code: int) -> ReducedObservation:
    """Exit code + first error/traceback from stderr."""
    status = "success" if exit_code == 0 else "failed"
    error_lines = []
    if stderr:
        lines = stderr.splitlines()
        capture = False
        for line in lines:
            if re.search(r"(error|Error|ERROR|Traceback|FAILED|fatal)", line):
                capture = True
            if capture:
                error_lines.append(line)
            if len(error_lines) >= 20:
                break
    snippet = "\n".join(error_lines) if error_lines else "(no error output captured)"
    msg = f"$ {command}\nExit code: {exit_code} ({status})\n\n{snippet}"
    original_tokens = estimate_tokens(stdout + stderr)
    return ReducedObservation(
        action="bash_first_error",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["bash_structured_summary", "bash_full"],
    )


def bash_structured_summary(command: str, stdout: str, stderr: str, exit_code: int) -> ReducedObservation:
    """Group errors/warnings by type, show counts and first instance."""
    status = "success" if exit_code == 0 else "failed"
    combined = stderr + "\n" + stdout
    error_types: dict[str, list[str]] = {}
    warning_types: dict[str, list[str]] = {}

    for line in combined.splitlines():
        if re.search(r"(error|Error|ERROR|fatal|FATAL)", line):
            key = re.sub(r"(line \d+|:\d+|0x[0-9a-f]+)", "N", line.strip())[:80]
            error_types.setdefault(key, []).append(line.strip())
        elif re.search(r"(warning|Warning|WARN)", line):
            key = re.sub(r"(line \d+|:\d+|0x[0-9a-f]+)", "N", line.strip())[:80]
            warning_types.setdefault(key, []).append(line.strip())

    parts = [f"$ {command}", f"Exit code: {exit_code} ({status})", ""]
    if error_types:
        parts.append(f"Errors ({len(error_types)} types, {sum(len(v) for v in error_types.values())} total):")
        for pattern, instances in list(error_types.items())[:10]:
            parts.append(f"  [{len(instances)}x] {instances[0][:120]}")
    if warning_types:
        parts.append(f"Warnings ({len(warning_types)} types, {sum(len(v) for v in warning_types.values())} total):")
        for pattern, instances in list(warning_types.items())[:5]:
            parts.append(f"  [{len(instances)}x] {instances[0][:120]}")
    if not error_types and not warning_types:
        parts.append("No errors or warnings detected.")

    msg = "\n".join(parts)
    original_tokens = estimate_tokens(stdout + stderr)
    return ReducedObservation(
        action="bash_structured_summary",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["bash_full"],
    )


def bash_full(command: str, stdout: str, stderr: str, exit_code: int) -> ReducedObservation:
    """Pass-through: full output."""
    msg = f"$ {command}\nExit code: {exit_code}\n\n[stdout]\n{stdout}\n\n[stderr]\n{stderr}"
    tokens = estimate_tokens(msg)
    return ReducedObservation(
        action="bash_full",
        content=msg,
        tokens_estimate=tokens,
        original_tokens=tokens,
        compression_ratio=1.0,
        available_escalations=[],
    )


# ============================================================================
# Test Log Renderers
# ============================================================================

_TEST_FAIL_PATTERNS = [
    re.compile(r"FAILED\s+([\w/.:]+)"),
    re.compile(r"FAIL:\s+(\w+)"),
    re.compile(r"ERROR:\s+(\w+)"),
    re.compile(r"✗\s+(.+)"),
    re.compile(r"not ok\s+\d+\s+(.+)"),
]


def _extract_failing_test_names(output: str) -> list[str]:
    names = []
    for line in output.splitlines():
        for pattern in _TEST_FAIL_PATTERNS:
            m = pattern.search(line)
            if m:
                names.append(m.group(1).strip())
                break
    return list(dict.fromkeys(names))


def _extract_failure_tracebacks(output: str, max_blocks: int = 0) -> list[str]:
    """Extract failure/error tracebacks from test output."""
    blocks = []
    lines = output.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.search(r"(FAILED|ERRORS|FAILURES|Traceback|AssertionError|assert )", line):
            block = []
            j = i
            for j in range(max(0, i - 1), min(len(lines), i + 30)):
                block.append(lines[j])
                if j > i and lines[j].strip() == "" and len(block) > 3:
                    break
            blocks.append("\n".join(block))
            i = j + 1
            if max_blocks and len(blocks) >= max_blocks:
                break
        else:
            i += 1
    return blocks


def test_structured_summary(command: str, output: str, exit_code: int) -> ReducedObservation:
    """Parse test output into a compact structured summary (~100 tokens max).

    Extracts: pass/fail counts, failing test names, and first error line.
    This gives the agent everything it typically needs without the full output.
    """
    status = "PASSED" if exit_code == 0 else "FAILED"
    lines = output.splitlines()

    # Extract pass/fail counts
    passed = 0
    failed = 0
    errors = 0
    for line in lines[-20:]:
        m = re.search(r"(\d+)\s+passed", line)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+)\s+failed", line)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+)\s+error", line)
        if m:
            errors = int(m.group(1))

    # Extract failing test names
    failing_names = _extract_failing_test_names(output)

    # Extract first error/assertion line
    first_error_line = ""
    for line in lines:
        if re.search(r"(AssertionError|assert |TypeError|ValueError|KeyError|AttributeError|ImportError)", line):
            first_error_line = line.strip()[:200]
            break

    parts = [f"Tests: {status}", f"$ {command}"]
    if passed or failed or errors:
        parts.append(f"Results: {passed} passed, {failed} failed, {errors} errors")
    if failing_names:
        parts.append("Failing: " + ", ".join(failing_names[:10]))
    if first_error_line:
        parts.append(f"Error: {first_error_line}")

    msg = "\n".join(parts)
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="test_structured_summary",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["test_first_failure", "test_all_failures", "test_full"],
    )


def test_exit_code_only(command: str, output: str, exit_code: int) -> ReducedObservation:
    status = "PASSED" if exit_code == 0 else f"FAILED (exit code {exit_code})"
    msg = f"Tests: {status}\n$ {command}"
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="test_exit_code_only",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["test_structured_summary", "test_failing_names", "test_first_failure", "test_full"],
    )


def test_failing_names(command: str, output: str, exit_code: int) -> ReducedObservation:
    status = "PASSED" if exit_code == 0 else f"FAILED (exit code {exit_code})"
    names = _extract_failing_test_names(output)
    names_str = "\n".join(f"  - {n}" for n in names) if names else "  (none extracted)"
    msg = f"Tests: {status}\n$ {command}\n\nFailing tests:\n{names_str}"
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="test_failing_names",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["test_first_failure", "test_all_failures", "test_full"],
    )


def test_first_failure(command: str, output: str, exit_code: int) -> ReducedObservation:
    status = "PASSED" if exit_code == 0 else f"FAILED (exit code {exit_code})"
    blocks = _extract_failure_tracebacks(output, max_blocks=1)
    traceback_str = blocks[0] if blocks else "(no traceback captured)"
    msg = f"Tests: {status}\n$ {command}\n\nFirst failure:\n{traceback_str}"
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="test_first_failure",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["test_all_failures", "test_full"],
    )


def test_all_failures(command: str, output: str, exit_code: int) -> ReducedObservation:
    status = "PASSED" if exit_code == 0 else f"FAILED (exit code {exit_code})"
    blocks = _extract_failure_tracebacks(output)
    if blocks:
        collapsed = "\n\n---\n\n".join(blocks[:20])
    else:
        collapsed = "(no tracebacks captured)"
    msg = f"Tests: {status}\n$ {command}\n\nAll failures ({len(blocks)}):\n{collapsed}"
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="test_all_failures",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["test_full"],
    )


def test_full(command: str, output: str, exit_code: int) -> ReducedObservation:
    msg = f"$ {command}\nExit code: {exit_code}\n\n{output}"
    tokens = estimate_tokens(msg)
    return ReducedObservation(
        action="test_full",
        content=msg,
        tokens_estimate=tokens,
        original_tokens=tokens,
        compression_ratio=1.0,
        available_escalations=[],
    )


# ============================================================================
# Search/Grep Renderers
# ============================================================================


def search_count_only(command: str, output: str) -> ReducedObservation:
    lines = [l for l in output.splitlines() if l.strip()]
    count = len(lines)
    msg = f"$ {command}\n{'No matches' if count == 0 else f'{count} matches found'}"
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="search_count_only",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["search_filenames_only", "search_top_matches", "search_full"],
    )


def search_filenames_only(command: str, output: str) -> ReducedObservation:
    filenames = set()
    for line in output.splitlines():
        if ":" in line:
            filenames.add(line.split(":")[0])
        elif line.strip():
            filenames.add(line.strip())
    sorted_files = sorted(filenames)
    files_str = "\n".join(f"  {f}" for f in sorted_files) if sorted_files else "  (none)"
    msg = f"$ {command}\n{len(sorted_files)} files with matches:\n{files_str}"
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="search_filenames_only",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["search_top_matches", "search_full"],
    )


def search_top_matches(command: str, output: str, max_results: int = 10) -> ReducedObservation:
    lines = output.splitlines()
    selected = lines[:max_results]
    shown = "\n".join(selected)
    remaining = len(lines) - len(selected)
    msg = f"$ {command}\nTop {len(selected)} matches (of {len(lines)} total):\n{shown}"
    if remaining > 0:
        msg += f"\n\n[{remaining} more matches not shown]"
    original_tokens = estimate_tokens(output)
    return ReducedObservation(
        action="search_top_matches",
        content=msg,
        tokens_estimate=estimate_tokens(msg),
        original_tokens=original_tokens,
        compression_ratio=estimate_tokens(msg) / max(original_tokens, 1),
        available_escalations=["search_full"],
    )


def search_full(command: str, output: str) -> ReducedObservation:
    msg = f"$ {command}\n\n{output}"
    tokens = estimate_tokens(msg)
    return ReducedObservation(
        action="search_full",
        content=msg,
        tokens_estimate=tokens,
        original_tokens=tokens,
        compression_ratio=1.0,
        available_escalations=[],
    )


# ============================================================================
# Dispatch: select action by name
# ============================================================================

ACTION_REGISTRY = {
    # File read actions
    "allow_full": allow_full,
    "return_relevant_spans": return_relevant_spans,
    "return_ast_outline": return_ast_outline,
    "return_summary": return_summary,
    "return_diff": return_diff,
    "return_cached_view": return_cached_view,
    "return_edit_receipt": return_edit_receipt,
    "block_with_pointer": block_with_pointer,
    # Bash/build actions
    "bash_exit_code_only": bash_exit_code_only,
    "bash_first_error": bash_first_error,
    "bash_structured_summary": bash_structured_summary,
    "bash_full": bash_full,
    # Test log actions
    "test_exit_code_only": test_exit_code_only,
    "test_structured_summary": test_structured_summary,
    "test_failing_names": test_failing_names,
    "test_first_failure": test_first_failure,
    "test_all_failures": test_all_failures,
    "test_full": test_full,
    # Search/grep actions
    "search_count_only": search_count_only,
    "search_filenames_only": search_filenames_only,
    "search_top_matches": search_top_matches,
    "search_full": search_full,
}

ACTIONS_BY_COST = [
    "block_with_pointer",
    "return_edit_receipt",
    "return_cached_view",
    "return_summary",
    "return_ast_outline",
    "return_diff",
    "return_relevant_spans",
    "allow_full",
]

BASH_ACTIONS_BY_COST = [
    "bash_exit_code_only",
    "bash_first_error",
    "bash_structured_summary",
    "bash_full",
]

TEST_ACTIONS_BY_COST = [
    "test_exit_code_only",
    "test_structured_summary",
    "test_failing_names",
    "test_first_failure",
    "test_all_failures",
    "test_full",
]

SEARCH_ACTIONS_BY_COST = [
    "search_count_only",
    "search_filenames_only",
    "search_top_matches",
    "search_full",
]

_TOOL_TYPE_ACTIONS = {
    "file_read": ACTIONS_BY_COST,
    "bash": BASH_ACTIONS_BY_COST,
    "test": TEST_ACTIONS_BY_COST,
    "search": SEARCH_ACTIONS_BY_COST,
}


def get_tool_actions(tool_type: str) -> list[str]:
    """Return the ordered action list for a given tool type."""
    if tool_type not in _TOOL_TYPE_ACTIONS:
        raise ValueError(f"Unknown tool_type: {tool_type}. Expected one of {list(_TOOL_TYPE_ACTIONS.keys())}")
    return _TOOL_TYPE_ACTIONS[tool_type]


def get_available_actions(
    has_prior_observation: bool,
    has_recent_edit: bool,
    hash_unchanged: bool,
) -> list[str]:
    """Return actions available given current state (availability mask)."""
    available = ["allow_full", "return_summary", "return_ast_outline", "return_relevant_spans", "block_with_pointer"]

    if has_recent_edit:
        available.append("return_edit_receipt")
    if has_prior_observation:
        available.append("return_diff")
    if has_prior_observation and hash_unchanged:
        available.append("return_cached_view")

    return sorted(available, key=lambda a: ACTIONS_BY_COST.index(a))


def generate_observation(
    action: str,
    file_path: str,
    file_content: str,
    task_description: str = "",
    prior_content: Optional[str] = None,
    edit_old: str = "",
    edit_new: str = "",
    embedder=None,
) -> ReducedObservation:
    """Dispatch to the appropriate file-read action generator."""
    if action == "allow_full":
        return allow_full(file_path, file_content)
    elif action == "block_with_pointer":
        return block_with_pointer(file_path, file_content)
    elif action == "return_summary":
        return return_summary(file_path, file_content)
    elif action == "return_ast_outline":
        return return_ast_outline(file_path, file_content)
    elif action == "return_cached_view":
        return return_cached_view(file_path, file_content)
    elif action == "return_edit_receipt":
        return return_edit_receipt(file_path, file_content, edit_old, edit_new)
    elif action == "return_diff":
        if prior_content is None:
            return return_ast_outline(file_path, file_content)
        return return_diff(file_path, file_content, prior_content)
    elif action == "return_relevant_spans":
        return return_relevant_spans(
            file_path,
            file_content,
            task_description,
            embedder=embedder,
        )
    else:
        return allow_full(file_path, file_content)


def generate_tool_observation(tool_type: str, action: str, **kwargs) -> ReducedObservation:
    """General dispatcher for any tool type and action."""
    if tool_type == "file_read":
        return generate_observation(action=action, **kwargs)
    elif tool_type == "bash":
        fn_map = {
            "bash_exit_code_only": bash_exit_code_only,
            "bash_first_error": bash_first_error,
            "bash_structured_summary": bash_structured_summary,
            "bash_full": bash_full,
        }
        if action not in fn_map:
            return bash_full(**kwargs)
        return fn_map[action](**kwargs)
    elif tool_type == "test":
        fn_map = {
            "test_exit_code_only": test_exit_code_only,
            "test_structured_summary": test_structured_summary,
            "test_failing_names": test_failing_names,
            "test_first_failure": test_first_failure,
            "test_all_failures": test_all_failures,
            "test_full": test_full,
        }
        if action not in fn_map:
            return test_full(**kwargs)
        return fn_map[action](**kwargs)
    elif tool_type == "search":
        fn_map = {
            "search_count_only": search_count_only,
            "search_filenames_only": search_filenames_only,
            "search_top_matches": search_top_matches,
            "search_full": search_full,
        }
        if action not in fn_map:
            return search_full(**kwargs)
        return fn_map[action](**kwargs)
    else:
        raise ValueError(f"Unknown tool_type: {tool_type}")
