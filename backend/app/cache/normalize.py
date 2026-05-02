import hashlib
import re

_FILLERS = ("帮我", "请", "麻烦", "能不能", "可以", "一下", "告诉我")
_PUNCTUATION_RE = re.compile(r"[，。！？、,.!?]+")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = _PUNCTUATION_RE.sub(" ", cleaned)
    for filler in _FILLERS:
        cleaned = cleaned.replace(filler, "")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def make_llm_cache_key(query: str, team_id: str, year_month: str) -> str:
    normalized = normalize_query(query)
    raw = f"{team_id}:{year_month}:{normalized}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"llm:resp:{digest}"
