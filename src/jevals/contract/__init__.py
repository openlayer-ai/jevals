"""Contract evals: an agent's system prompt as rules, and which questions cover them.

jevals contract rules prompt.md -o contract.yaml
# your coding agent writes evals whose questions cite rule ids: covers: [r-3f2a9c]
jevals contract check contract.yaml evals/
"""

from ._coverage import (
    Citation,
    CoverageReport,
    Doubt,
    PromptDiff,
    acheck_coverage,
    check_coverage,
)
from ._rules import (
    Contract,
    Extraction,
    Rule,
    aextract_rules,
    extract_rules,
    load_contract,
    normalize_rule,
    prompt_from_trace,
    prompt_sha256,
    rule_id,
    split_rules,
)

__all__ = [
    "Citation",
    "Contract",
    "CoverageReport",
    "Doubt",
    "PromptDiff",
    "acheck_coverage",
    "check_coverage",
    "Extraction",
    "Rule",
    "aextract_rules",
    "extract_rules",
    "load_contract",
    "normalize_rule",
    "prompt_from_trace",
    "prompt_sha256",
    "rule_id",
    "split_rules",
]
