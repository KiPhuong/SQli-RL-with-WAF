"""
sql_prefix_validator.py

A self-contained Python module that provides a deterministic, fast-ish validator and
suggestion engine for incrementally-generated SQL payloads (suitable for RL agents).

Features:
- Loads a vocabulary from /mnt/data/keywords.txt (one token per line).
- Uses sqlglot for parsing (no DB connections).
- Detects whether a SQL string is:
    * a complete, syntactically valid statement
    * a potentially-extendable prefix (incomplete but plausibly extendable)
- Suggests next tokens from the loaded vocabulary in a context-aware deterministic order.
- Robust to sqlglot TokenError (a known tokenizer bug for some prefixes).
- Caches parse attempts for speed.

Requirements:
    pip install sqlglot

Usage:
    import and construct SQLPrefixValidator, then call methods:
      v = SQLPrefixValidator()
      v.is_complete_query(sql_text)
      v.is_potential_prefix(sql_text)
      v.suggest_next_tokens(sql_text, top_k=20)
"""

from typing import List, Tuple, Optional, Iterable
from functools import lru_cache
import re
import os

from sqlglot import parse_one
from sqlglot.errors import ParseError, TokenError

# ---------- Configuration / Heuristics ----------
DEFAULT_VOCAB_PATH = "keywords.txt"

# small set of trailing keywords that typically indicate continuation
TRAILING_KEYWORDS = {
    "SELECT",
    "FROM",
    "WHERE",
    "GROUP",
    "ORDER",
    "BY",
    "HAVING",
    "LIMIT",
    "OFFSET",
    "UNION",
    "UNION ALL",
    "INSERT",
    "VALUES",
    "INTO",
    "UPDATE",
    "SET",
    "DELETE",
    "JOIN",
    "ON",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "WITH",
    "OVER",
}

# Common small patches to try when testing if a prefix can be extended.
COMMON_PATCHES = (
    " SELECT 1",
    " SELECT 1;",
    " FROM DUAL",
    ";",
    " )",
    "'",   # close single quote
    '"',   # close double quote
    " -- comment",
    " /* comment */",
)

# regex helpers
_RE_LAST_WORD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*\s*)$", re.IGNORECASE)
_RE_SINGLE_QUOTE = re.compile(r"(?<!\\)'")
_RE_DOUBLE_QUOTE = re.compile(r'(?<!\\)"')


# ---------- Utility functions ----------
def _strip_and_norm(sql: str) -> str:
    return (sql or "").strip()


def _remove_quoted_substrings(sql: str) -> str:
    """
    Remove substrings enclosed in single or double quotes (naive).
    Used for simple parentheses counting ignoring quoted text.
    """
    # replace quoted substrings with empty string to avoid counting internal parens
    return re.sub(r"'(?:\\'|[^'])*'|\"(?:\\\"|[^\"])*\"", "", sql)


def _ends_with_one_of(sql_upper: str, keywords: Iterable[str]) -> bool:
    for kw in sorted(keywords, key=len, reverse=True):
        if sql_upper.endswith(kw):
            return True
    return False


# ---------- Caching parse attempts ----------
@lru_cache(maxsize=16384)
def _try_parse_once(candidate: str) -> bool:
    """
    Try to parse candidate SQL with sqlglot.parse_one.
    Catches ParseError and TokenError and returns False if parsing fails.
    Cached for speed.
    """
    if not candidate or not candidate.strip():
        return False
    try:
        parse_one(candidate)
        return True
    except (ParseError, TokenError):
        return False


# ---------- Main class ----------
class SQLPrefixValidator:
    """
    Class that validates SQL prefixes and suggests next tokens.

    Public API:
      - is_potential_prefix(sql_text: str) -> bool
      - is_complete_query(sql_text: str) -> bool
      - suggest_next_tokens(sql_text: str, top_k: int = 20, return_scores: bool = False) -> List[str] or List[(str,float)]
    """

    def __init__(self, base_query: str = "SELECT * FROM TABLE_NAME WHERE id = -1",
                 vocab_path: str = DEFAULT_VOCAB_PATH,
                 patches: Optional[Iterable[str]] = None):
        """
        Initialize validator. Loads vocabulary from vocab_path (one token per line).
        """
        self.base_query = base_query or ""
        self.vocab_path = vocab_path
        self.vocab = self._load_vocab(vocab_path)
        # keep original order for deterministic tie-breaking
        self.vocab_index = {tok: i for i, tok in enumerate(self.vocab)}
        self.patches = tuple(patches) if patches is not None else COMMON_PATCHES

    @staticmethod
    def _load_vocab(path: str) -> List[str]:
        """
        Load tokens from a file. Each non-empty line is a token. If file not found,
        returns a small fallback list.
        """
        try:
            if not path or not os.path.exists(path):
                # fallback minimal vocab
                return [
                    " ", ";", "--", "/*", "*/", "(", ")", ",",
                    "UNION", "ALL", "SELECT", "FROM", "WHERE", "ORDER", "BY",
                    "AND", "OR", "NOT", "SLEEP(", "MD5(", "COUNT(", "LIMIT", "'"
                ]
            toks = []
            with open(path, "r", encoding="utf8") as f:
                for line in f:
                    token = line.rstrip("\n\r")
                    if token is None:
                        continue
                    token = token.strip()
                    if token == "":
                        continue
                    toks.append(token)
            # ensure deterministic order and unique
            seen = set()
            uniq = []
            for t in toks:
                if t not in seen:
                    seen.add(t)
                    uniq.append(t)
            return uniq
        except Exception:
            # On any error fallback to small hardcoded vocab to avoid runtime crash
            return [
                " ", ";", "--", "/*", "*/", "(", ")", ",",
                "UNION", "ALL", "SELECT", "FROM", "WHERE", "ORDER", "BY",
                "AND", "OR", "NOT", "SLEEP(", "MD5(", "COUNT(", "LIMIT", "'"
            ]

    # ----------------- completeness / prefix checks -----------------
    def is_complete_query(self, sql_text: str) -> bool:
        """
        True if sql_text is a complete syntactically valid SQL statement according to sqlglot.
        """
        s = _strip_and_norm(sql_text)
        if not s:
            return False
        # try parsing directly (cached)
        return _try_parse_once(s)

    def _has_unclosed_parentheses(self, sql_text: str) -> bool:
        s = _strip_and_norm(sql_text)
        if not s:
            return False
        no_quotes = _remove_quoted_substrings(s)
        return no_quotes.count("(") > no_quotes.count(")")

    def _has_unclosed_quote(self, sql_text: str) -> bool:
        """
        Simple heuristic: odd number of single or double quotes -> open quote present.
        """
        s = sql_text or ""
        single_count = len(re.findall(_RE_SINGLE_QUOTE, s))
        double_count = len(re.findall(_RE_DOUBLE_QUOTE, s))
        return (single_count % 2 == 1) or (double_count % 2 == 1)

    def _ends_with_trailing_keyword(self, sql_text: str) -> bool:
        s = _strip_and_norm(sql_text).upper()
        if not s:
            return False
        # check multi-word keywords too via TRAILING_KEYWORDS set
        if _ends_with_one_of(s, TRAILING_KEYWORDS):
            return True
        # also check last token (simple)
        m = _RE_LAST_WORD.search(s)
        if m:
            last = m.group(1).strip().upper()
            if last in TRAILING_KEYWORDS:
                return True
        return False

    def _is_prefix_of_known_keyword(self, sql_text: str) -> bool:
        """
        If the last token is a prefix of any vocabulary token or trailing keyword, return True.
        """
        s = _strip_and_norm(sql_text)
        if not s:
            return False
        m = _RE_LAST_WORD.search(s)
        if not m:
            return False
        last = m.group(1).strip().upper()
        # candidate keywords: vocab + internal TRAILING_KEYWORDS
        candidates = set([k.upper() for k in TRAILING_KEYWORDS])
        for v in self.vocab:
            if not v:
                continue
            candidates.add(v.upper())
        for cand in candidates:
            if cand != last and cand.startswith(last):
                return True
        return False

    @lru_cache(maxsize=8192)
    def _try_patches(self, sql_text: str) -> bool:
        """
        Try appending small patches (like ' SELECT 1', "'", etc.) to sql_text and parse.
        Cached for speed.
        """
        s = _strip_and_norm(sql_text)
        if not s:
            return False
        for p in self.patches:
            candidate = s + p
            if _try_parse_once(candidate):
                return True
        return False

    def is_potential_prefix(self, sql_text: str, try_patching: bool = True) -> bool:
        """
        Return True if sql_text (possibly incomplete) could be extended into a syntactically valid SQL statement.

        Heuristics used (conservative):
          - already complete -> True
          - unclosed parentheses -> True
          - unclosed single/double quote -> True
          - ends with trailing keyword (e.g., 'UNION', 'WHERE') -> True
          - last token is prefix of known vocab/keywords -> True
          - trailing comma or operator -> True
          - optionally try small patches and attempt parsing -> True
        """
        s = _strip_and_norm(sql_text)
        if not s:
            return False

        # 1) Completed successfully
        if self.is_complete_query(s):
            return True

        # 2) heuristic checks
        if self._has_unclosed_parentheses(s):
            return True
        if self._has_unclosed_quote(s):
            return True
        if self._ends_with_trailing_keyword(s):
            return True
        if self._is_prefix_of_known_keyword(s):
            return True
        # trailing comma or operator (e.g., ends with '+' '-' '*' '/' %)
        if s.endswith(",") or re.search(r"[+\-*/%]\s*$", s):
            return True

        # 3) try light patching (costlier)
        if try_patching and self._try_patches(s):
            return True

        return False

    def is_potential_extension(self, suffix: str, try_patching: bool = True) -> bool:
        """
        Convenience: check whether base_query + ' ' + suffix is complete/potential-prefix.
        Useful for RL where agent appends tokens to a fixed base query.
        """
        sfx = (suffix or "").strip()
        if not sfx:
            # no-op extension: check base
            return self.is_potential_prefix(self.base_query, try_patching=try_patching)
        candidate = (self.base_query + " " + sfx).strip()
        if self.is_complete_query(candidate):
            return True
        # if suffix itself looks extendable (like 'UNION SELECT') and candidate can be patched, return True
        if self.is_potential_prefix(sfx, try_patching=False):
            if try_patching and self._try_patches(candidate):
                return True
            # if suffix ends with trailing keyword it's likely extendable
            if self._ends_with_trailing_keyword(sfx):
                return True
        # try patching the whole candidate
        if try_patching and self._try_patches(candidate):
            return True
        return False

    # ----------------- suggestion engine -----------------
    def _score_candidate(self, sql_text: str, token: str) -> float:
        """
        Heuristic scoring rules (higher is more plausible). Deterministic.
        Score range roughly 0.0 - 1.0.
        """
        s = _strip_and_norm(sql_text)
        t = token or ""
        score = 0.0

        # 1) exact context matches: e.g., if s endswith 'UNION' and token is 'ALL' or ' SELECT'
        s_up = s.upper()
        t_up = t.upper()

        if s_up.endswith("UNION") and (t_up == "ALL" or t_up.startswith(" SELECT") or t_up.startswith("SELECT") or t_up == " "):
            score += 0.5

        if s_up.endswith("WHERE") and (t_up == " " or t_up.startswith("NOT") or t_up.startswith("EXISTS") or t_up.startswith("SELECT") or t_up.startswith("'")):
            score += 0.4

        # punctuation preferences
        if t in (" ", ",", ";", "--", "/*", "*/", ")", "("):
            score += 0.2

        # function-like tokens (ending with '(') are plausible after SELECT or WHERE or commas
        if t.endswith("(") and re.search(r"(SELECT|,|\bAS\b|\bFROM\b)$", s_up):
            score += 0.25

        # tokens that directly make the candidate parseable with a small patch get a bonus
        candidate = s + t
        # Attempt quick parse of candidate and candidate + small patch; cached via _try_parse_once/_try_patches
        if _try_parse_once(candidate):
            score += 0.9
        else:
            # try patching candidate
            if self._try_patches(candidate):
                score += 0.6

        # if token is prefix of a known keyword and current last token is a prefix -> plausible
        if self._is_prefix_of_known_keyword(s + t):
            score += 0.15

        # short tokens penalty (we prefer meaningful tokens but allow punctuation)
        if len(t.strip()) == 0:
            score += 0.05

        # clamp to 0..1
        if score > 1.0:
            score = 1.0
        if score < 0.0:
            score = 0.0
        return float(score)

    def suggest_next_tokens(self, sql_text: str, top_k: int = 20, return_scores: bool = False) -> List:
        """
        Suggest up to top_k candidate tokens from loaded vocabulary that are plausible continuations.
        Deterministic: same input -> same ordered list.

        If return_scores is False: returns List[str]
        If return_scores is True: returns List[Tuple[str, float]] (token, score)

        Implementation notes:
          - For speed, we pre-filter vocab by simple heuristics (prefix matching, punctuation,
            or being a common continuation token based on trailing words).
          - We compute a deterministic score per candidate and sort by (-score, vocab_index).
        """
        s = sql_text or ""
        s_stripped = _strip_and_norm(s)
        s_up = s_stripped.upper()

        candidates = []

        # fast path: if inside a string literal, prefer closing the quote or useful attack fragments
        inside_single = (len(re.findall(_RE_SINGLE_QUOTE, s)) % 2 == 1)
        inside_double = (len(re.findall(_RE_DOUBLE_QUOTE, s)) % 2 == 1)
        if inside_single or inside_double:
            # prefer the matching quote token
            quote_token = "'" if inside_single else '"'
            # ensure quote present in vocab; otherwise include it
            if quote_token in self.vocab:
                candidates.append(quote_token)
            else:
                candidates.append(quote_token)
            # also propose common payload fragments if present in vocab
            for tok in self.vocab:
                if tok.startswith("'") or tok.startswith('"'):
                    candidates.append(tok)
            # deduplicate preserving order
            uniq = []
            seen = set()
            for t in candidates:
                if t not in seen:
                    uniq.append(t)
                    seen.add(t)
            candidates = uniq

        # otherwise build a set of plausible tokens using simple filters
        if not candidates:
            # If text ends with alphabetic prefix, prefer tokens that start with that prefix (case-insensitive)
            m = _RE_LAST_WORD.search(s_stripped)
            last_token = m.group(1).strip() if m else ""
            last_up = last_token.upper() if last_token else ""

            # build a deterministic candidate list: preserve original vocab order
            for tok in self.vocab:
                tok_stripped = tok or ""
                tok_up = tok_stripped.upper()

                # always include punctuation tokens
                if tok_stripped in {",", ";", " ", ")", "(", "--", "/*", "*/"}:
                    candidates.append(tok_stripped)
                    continue

                # if last token equals a keyword that expects certain continuations, favor those
                if last_up == "UNION":
                    if tok_up == "ALL" or tok_up.startswith("SELECT") or tok_up == " ":
                        candidates.append(tok_stripped)
                        continue

                if last_up == "WHERE" and (tok_up.startswith("NOT") or tok_up.startswith("EXISTS") or tok_up.startswith("SELECT") or tok_up.startswith("'")):
                    candidates.append(tok_stripped)
                    continue

                # prefix matching: if last token is a prefix of vocab token, include
                if last_up and tok_up.startswith(last_up):
                    candidates.append(tok_stripped)
                    continue

                # otherwise include tokens that are short (punctuation) or SQL keywords (heuristic)
                if len(tok_stripped) <= 3 and tok_stripped.isalpha():
                    candidates.append(tok_stripped)
                    continue

                # include functions and keywords heuristically
                if tok_stripped.endswith("(") or tok_up in TRAILING_KEYWORDS or tok_up.startswith("SELECT") or tok_up.startswith("FROM") or tok_up.startswith("WHERE"):
                    candidates.append(tok_stripped)
                    continue

            # fallback: include entire vocab (ensures deterministic behavior)
            if not candidates:
                candidates = list(self.vocab)

        # Deduplicate while preserving first-seen order
        seen = set()
        deduped = []
        for t in candidates:
            if t not in seen:
                deduped.append(t)
                seen.add(t)
        candidates = deduped

        # Score each candidate deterministically. Keep only those with non-zero-ish score to reduce noise.
        scored = []
        for t in candidates:
            sc = self._score_candidate(s_stripped, t)
            # include tokens with minimal score threshold to avoid noise; keep 0 too if needed
            if sc > 0.0:
                scored.append((t, sc))
        # If nothing scored > 0, include first N candidates with tiny scores
        if not scored:
            scored = [(t, 0.0) for t in candidates[:top_k * 5]]

        # Sort by score desc, then by original vocab index for determinism
        def sort_key(item):
            token, score = item
            idx = self.vocab_index.get(token, 10_000_000)  # unknown tokens go last deterministically
            # negative score so higher scores come first
            return (-score, idx)

        scored.sort(key=sort_key)

        # limit to top_k
        final = scored[:top_k]

        if return_scores:
            return final
        # else return list of tokens
        return [t for t, _ in final]


# ----------------- Demo / CLI -----------------
if __name__ == "__main__":
    import pprint

    validator = SQLPrefixValidator()

    examples = [
        "UNION",
        "UNION SELECT",
        "UNION SELECT 1",
        "SELECT * FROM",
        "SELECT * FROM t WHERE name = '",
        "SLE",  # prefix of SLEEP(
        validator.base_query,
        validator.base_query + " UNION SELECT username FROM users --",
    ]

    print("Loaded vocab (first 60 shown):")
    pprint.pprint(validator.vocab[:60])
    print("=" * 80)

    for ex in examples:
        print(f"INPUT: {repr(ex)}")
        try:
            complete = validator.is_complete_query(ex)
        except Exception as e:
            complete = f"ERROR: {e}"
        try:
            potential = validator.is_potential_prefix(ex)
        except Exception as e:
            potential = f"ERROR: {e}"
        try:
            extension = validator.is_potential_extension(ex)
        except Exception as e:
            extension = f"ERROR: {e}"
        suggestions = validator.suggest_next_tokens(ex, top_k=10, return_scores=True)
        print("  is_complete_query: ", complete)
        print("  is_potential_prefix:", potential)
        print("  is_potential_extension(base + suffix):", extension)
        print("  suggestions (token, score):")
        for tok, sc in suggestions:
            print(f"    {repr(tok):20} -> {sc:.3f}")
        print("-" * 80)
