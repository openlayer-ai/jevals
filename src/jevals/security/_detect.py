"""Deterministic detectors: PII / PHI entities and secrets.

Presidio is used when installed (`pip install "jevals[pii]"`). Otherwise a
regex-and-checksum fallback covers the common entity types. Either way the
result is a list of {type, text, start, end, score}.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

Entity = dict[str, Any]

# --------------------------------------------------------------------------- checksums


def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits if c.isdigit()]
    if len(d) < 12:
        return False
    total = 0
    for i, x in enumerate(reversed(d)):
        if i % 2 == 1:
            x *= 2
            if x > 9:
                x -= 9
        total += x
    return total % 10 == 0


def npi_ok(digits: str) -> bool:
    """US National Provider Identifier: 10 digits, Luhn with prefix 80840."""
    if not re.fullmatch(r"\d{10}", digits):
        return False
    return luhn_ok("80840" + digits)


def cpf_ok(digits: str) -> bool:
    d = re.sub(r"\D", "", digits)
    if len(d) != 11 or d == d[0] * 11:
        return False
    for n in (9, 10):
        s = sum(int(d[i]) * ((n + 1) - i) for i in range(n))
        chk = (s * 10) % 11 % 10
        if chk != int(d[n]):
            return False
    return True


# --------------------------------------------------------------------------- regex fallback

_PII_PATTERNS: list[tuple[str, re.Pattern[str], Any]] = [
    ("EMAIL_ADDRESS", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), None),
    ("US_SSN", re.compile(r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), None),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), lambda m: luhn_ok(m)),
    ("PHONE_NUMBER", re.compile(r"(?<!\d)(?:\+?1[ .-]?)?\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}(?!\d)"), None),
    (
        "IP_ADDRESS",
        re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
        None,
    ),
    ("IBAN_CODE", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}\b"), None),
    ("US_PASSPORT", re.compile(r"\b(?:passport(?:\s*(?:no|number|#))?[:\s]+)([A-Z]?\d{8,9})\b", re.I), None),
    (
        "DATE_OF_BIRTH",
        re.compile(
            r"\b(?:DOB|date of birth|born)[:\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})", re.I
        ),
        None,
    ),
    ("BR_CPF", re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), lambda m: cpf_ok(m)),
]

_PHI_PATTERNS: list[tuple[str, re.Pattern[str], Any]] = [
    ("US_NPI", re.compile(r"\b(?:NPI[:#\s]*)?(\d{10})\b"), lambda m: npi_ok(re.sub(r"\D", "", m)[-10:])),
    (
        "MEDICAL_RECORD_NUMBER",
        re.compile(
            r"\b(?:MRN|medical record(?: number| no\.?)?|patient (?:id|number))[:#\s]+([A-Z0-9-]{5,14})\b",
            re.I,
        ),
        None,
    ),
    (
        "HEALTH_PLAN_ID",
        re.compile(
            r"\b(?:member|policy|subscriber|insurance|health plan) (?:id|number|no\.?)[:#\s]+([A-Z0-9-]{6,16})\b",
            re.I,
        ),
        None,
    ),
    (
        "ICD10_CODE",
        re.compile(r"\b[A-TV-Z][0-9][0-9AB]\.?[0-9A-TV-Z]{0,4}\b(?=.{0,40}(?:diagnos|code|icd))", re.I),
        None,
    ),
    ("DEA_NUMBER", re.compile(r"\b[ABCDEFGHJKLMPRSTUX][A-Z9]\d{7}\b"), None),
]

_MEDICAL_TERMS = re.compile(
    r"\b(diagnos\w*|prescri\w*|patient|symptom\w*|treatment|therapy|dosage|mg\b|medication|clinic\w*|hospital\w*|"
    r"surgery|oncolog\w*|hiv|cancer|diabet\w*|depress\w*|anxiety|pregnan\w*|hypertension|asthma|mental health|"
    r"blood (?:test|pressure|type)|lab results?|discharge|admitted|physician|doctor|nurse|icd-?10|cpt|rx\b)",
    re.I,
)

_SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("AWS_ACCESS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    (
        "AWS_SECRET_KEY",
        re.compile(r"(?i)aws(?:.{0,20})?(?:secret|access)?(?:.{0,10})?['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
    ),
    (
        "GITHUB_TOKEN",
        re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b|\bgithub_pat_[A-Za-z0-9_]{60,}\b"),
    ),
    ("ANTHROPIC_KEY", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("OPENAI_KEY", re.compile(r"\bsk-(?!ant-)(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}\b")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    ("STRIPE_KEY", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}\b")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY(?: BLOCK)?-----")),
    (
        "GENERIC_SECRET",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd|pwd)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-/+=]{16,})['\"]?"
        ),
    ),
    (
        "DATABASE_URL",
        re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s:@/]+:[^\s@/]+@[^\s/]+"),
    ),
]


def _scan(text: str, patterns: list[tuple[str, re.Pattern[str], Any]]) -> list[Entity]:
    out: list[Entity] = []
    for etype, pat, check in patterns:
        for m in pat.finditer(text):
            val = m.group(1) if m.groups() and m.group(1) else m.group(0)
            if check and not check(val):
                continue
            out.append(
                {
                    "type": etype,
                    "text": val,
                    "start": m.start(),
                    "end": m.end(),
                    "score": 0.85 if check else 0.7,
                }
            )
    return out


def _dedupe(ents: list[Entity]) -> list[Entity]:
    ents.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    out: list[Entity] = []
    for e in ents:
        if out and e["start"] < out[-1]["end"]:
            if e["score"] > out[-1]["score"]:
                out[-1] = e
            continue
        out.append(e)
    return out


# --------------------------------------------------------------------------- presidio (optional)


@lru_cache(maxsize=1)
def _presidio():
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore

        return AnalyzerEngine()
    except Exception:  # noqa: BLE001  (not installed, or spaCy model missing)
        return None


def presidio_available() -> bool:
    return _presidio() is not None


def _presidio_scan(text: str, entities: list[str] | None, threshold: float) -> list[Entity] | None:
    eng = _presidio()
    if eng is None:
        return None
    res = eng.analyze(text=text, language="en", entities=entities, score_threshold=threshold)
    return [
        {
            "type": r.entity_type,
            "text": text[r.start : r.end],
            "start": r.start,
            "end": r.end,
            "score": float(r.score),
        }
        for r in res
    ]


# --------------------------------------------------------------------------- public


def detect_pii(text: str, threshold: float = 0.5, use_presidio: bool = True) -> list[Entity]:
    if not text:
        return []
    ents = _presidio_scan(text, None, threshold) if use_presidio else None
    if ents is None:
        ents = _scan(text, _PII_PATTERNS)
    else:  # presidio lacks BR CPF and checksum-validated cards in some configs; add ours
        ents += _scan(text, [p for p in _PII_PATTERNS if p[0] in ("BR_CPF",)])
    return _dedupe(ents)


def detect_phi(
    text: str, threshold: float = 0.5, use_presidio: bool = True
) -> tuple[list[Entity], list[str]]:
    """PHI = identifiers + health context. Returns (entities, medical_terms_found)."""
    if not text:
        return [], []
    ents = detect_pii(text, threshold, use_presidio) + _scan(text, _PHI_PATTERNS)
    terms = sorted({m.group(0).lower() for m in _MEDICAL_TERMS.finditer(text)})
    return _dedupe(ents), terms


# Patterns whose shape alone is decisive (fixed prefix + length). Anything else,
# and anything that looks like a placeholder, is scored low so the model is
# asked whether it is a real credential.
_STRUCTURED_SECRETS = {
    "AWS_ACCESS_KEY",
    "GITHUB_TOKEN",
    "ANTHROPIC_KEY",
    "OPENAI_KEY",
    "SLACK_TOKEN",
    "STRIPE_KEY",
    "GOOGLE_API_KEY",
    "JWT",
    "PRIVATE_KEY",
}
_PLACEHOLDER = re.compile(
    r"(?i)xxx|\.\.\.|<[^>]*>|\{\{|\$\{|\byour[_-]?|example|placeholder|changeme|dummy|sample|"
    r"\b(?:USER(?:NAME)?|PASS(?:WORD)?|PWD|SECRET|TOKEN|API[_-]?KEY|HOST|PORT|DB(?:NAME)?)\b"
)


def _looks_placeholder(val: str) -> bool:
    return bool(_PLACEHOLDER.search(val))


def detect_secrets(text: str) -> list[Entity]:
    if not text:
        return []
    out: list[Entity] = []
    for etype, pat in _SECRET_PATTERNS:
        for m in pat.finditer(text):
            val = m.group(1) if m.groups() and m.group(1) else m.group(0)
            if etype in _STRUCTURED_SECRETS and not _looks_placeholder(val):
                score = 0.9
            elif etype == "GENERIC_SECRET" or _looks_placeholder(val):
                score = 0.5
            else:
                score = 0.7
            out.append({"type": etype, "text": val, "start": m.start(), "end": m.end(), "score": score})
    return _dedupe(out)


def redact(text: str, entities: list[Entity], style: str = "type") -> str:
    """Replace entity spans. style: 'type' -> <EMAIL_ADDRESS>, 'mask' -> ****, 'partial' -> keep last 4."""
    out = text
    for e in sorted(entities, key=lambda e: -e["start"]):
        span = out[e["start"] : e["end"]]
        if style == "mask":
            rep = "*" * len(span)
        elif style == "partial":
            rep = "*" * max(0, len(span) - 4) + span[-4:]
        else:
            rep = f"<{e['type']}>"
        out = out[: e["start"]] + rep + out[e["end"] :]
    return out
