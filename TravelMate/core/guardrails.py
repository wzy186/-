"""Guardrails: input and output safety checks for the agent."""

import re
from typing import Callable


class GuardrailResult:
    def __init__(self, passed: bool, reason: str = "", severity: str = "low"):
        self.passed = passed
        self.reason = reason
        self.severity = severity  # low, medium, high, critical


# ── Input Guardrails ──

def check_prompt_injection(text: str) -> GuardrailResult:
    """Detect common prompt injection patterns."""
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now\s+",
        r"system\s*:\s*",
        r"new\s+instructions?\s*:",
        r"override\s+(safety|security|guardrails)",
        r"jailbreak",
        r"DAN\s+mode",
        r"pretend\s+you\s+(are|can)",
        r"act\s+as\s+if\s+you\s+have\s+no",
        r"reveal\s+(your|the)\s+(system|prompt|instructions)",
    ]
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return GuardrailResult(False, f"Possible prompt injection detected", "high")
    return GuardrailResult(True)


def check_content_safety(text: str) -> GuardrailResult:
    """Block clearly harmful content requests."""
    harmful = [
        r"how\s+to\s+(make|build|create)\s+(bomb|weapon|explosive)",
        r"harm\s+(someone|people|others)",
        r"kill\s+(someone|yourself)",
        r"suicide\s+method",
        r"illegal\s+(drug|activity)",
    ]
    for p in harmful:
        if re.search(p, text, re.IGNORECASE):
            return GuardrailResult(False, "Harmful content detected", "critical")
    return GuardrailResult(True)


def check_pii_leak(text: str) -> GuardrailResult:
    """Warn if user seems to be sharing sensitive PII."""
    import re as _re
    # Credit card pattern
    if _re.search(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', text):
        return GuardrailResult(True, "⚠️ 检测到疑似银行卡号，请勿在对话中分享敏感信息", "medium")
    # ID card
    if _re.search(r'\b\d{17}[\dXx]\b', text):
        return GuardrailResult(True, "⚠️ 检测到疑似身份证号，请勿在对话中分享敏感信息", "medium")
    return GuardrailResult(True)


# ── Output Guardrails ──

def check_output_safety(text: str) -> GuardrailResult:
    """Ensure output doesn't contain harmful content."""
    harmful = ["如何制造炸弹", "如何制作毒品", "自杀方法"]
    for h in harmful:
        if h in text:
            return GuardrailResult(False, "Output contains harmful content", "critical")
    return GuardrailResult(True)


def check_output_format(text: str, expected: str = "text") -> GuardrailResult:
    """Basic format validation."""
    if expected == "json":
        try:
            import json
            json.loads(text)
            return GuardrailResult(True)
        except json.JSONDecodeError:
            return GuardrailResult(True, "Output is not valid JSON but may be acceptable", "low")
    return GuardrailResult(True)


# ── Guardrail Runner ──

INPUT_GUARDRAILS: list[Callable[[str], GuardrailResult]] = [
    check_prompt_injection,
    check_content_safety,
    check_pii_leak,
]

OUTPUT_GUARDRAILS: list[Callable[[str], GuardrailResult]] = [
    check_output_safety,
]


def run_input_guardrails(text: str) -> list[GuardrailResult]:
    """Run all input guardrails. Returns results."""
    results = []
    for g in INPUT_GUARDRAILS:
        results.append(g(text))
    return results


def run_output_guardrails(text: str) -> list[GuardrailResult]:
    """Run all output guardrails."""
    results = []
    for g in OUTPUT_GUARDRAILS:
        results.append(g(text))
    return results


def has_critical_failure(results: list[GuardrailResult]) -> GuardrailResult | None:
    """Return the first critical failure, or None."""
    for r in results:
        if not r.passed and r.severity in ("high", "critical"):
            return r
    return None


def has_warnings(results: list[GuardrailResult]) -> list[str]:
    """Return warning messages from guardrail results."""
    return [r.reason for r in results if r.passed and r.reason]
