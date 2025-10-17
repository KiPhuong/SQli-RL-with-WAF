"""
SQL Prefix Validator for Reinforcement Learning
A validator for incomplete SQL queries that can suggest next tokens for RL agents.
"""

import re
import os
from typing import List, Tuple, Optional, Set
import sqlglot
from sqlglot import parse_one, ParseError


class SQLPrefixValidator:
    """
    Validates incomplete SQL queries and suggests next tokens for RL agents
    generating SQL payloads incrementally.
    """

    def __init__(self, base_query: str = "SELECT * FROM TABLE_NAME WHERE id = -1", vocab_file: str = "keywords.txt"):
        """
        Initialize the validator with a base query and vocabulary file.

        Args:
            base_query: The starting query that the RL agent builds upon
            vocab_file: Path to the vocabulary file containing SQL keywords and tokens
        """
        self.base_query = base_query

        # Load vocabulary from file
        self.vocab = self._load_vocabulary(vocab_file)

        # Separate vocabulary into categories for easier access
        self.keywords = set()
        self.functions = set()
        self.operators = set()
        self.punctuation = set()
        self.injection_fragments = set()

        self._categorize_vocabulary()

        # Common SQL injection payload fragments (additional to vocab file)
        self.injection_fragments.update({
            "' OR '1'='1", "' OR 1=1", "' AND '1'='1", "' AND 1=1",
            "' UNION SELECT", "'; DROP TABLE", "'; INSERT INTO",
            "' OR SLEEP(5)", "' AND SLEEP(5)", "admin'--", "' OR TRUE--",
            "' /*", "*/ OR '1'='1", "') OR ('1'='1"
        })

        # Patterns for incomplete constructs - more comprehensive coverage
        self.incomplete_patterns = [
            r'\bUNION\s*$', r'\bUNION\s+ALL\s*$', r'\bSELECT\s*$', r'\bFROM\s*$',
            r'\bWHERE\s*$', r'\bGROUP\s*$', r'\bGROUP\s+BY\s*$', r'\bORDER\s*$',
            r'\bORDER\s+BY\s*$', r'\bHAVING\s*$', r'\bLIMIT\s*$', r'\bOFFSET\s*$',
            r'\bAND\s*$', r'\bOR\s*$', r'\bNOT\s*$', r'\bIN\s*$', r'\bLIKE\s*$',
            r'\bBETWEEN\s*$', r'\bIS\s*$', r'\bCASE\s*$', r'\bWHEN\s*$',
            r'\bTHEN\s*$', r'\bELSE\s*$', r'=\s*$', r'<\s*$', r'>\s*$', r',\s*$',
            r'\(\s*$', r"'\s*$", r'--\s*$', r'/\*\s*$', r'\bJOIN\s*$',
            r'\bINNER\s+JOIN\s*$', r'\bLEFT\s+JOIN\s*$', r'\bRIGHT\s+JOIN\s*$',
            r'\bFULL\s+OUTER\s+JOIN\s*$', r'\bCROSS\s+JOIN\s*$',
            # Additional patterns for SQL injection scenarios
            r'\bTABLE\s*$', r'\bCOLLATE\s*$', r'\bUNIoN\s*$', r'\bUNIoN\s*\(\s*$',
            r'\d+\s*$', r"'\d+\s*$", r"'\d+>\s*$", r"'\d+>\d+\s*$", r"'\d+>\d+=\s*$",
            r"'\d+>\d+=\d+\s*$", r'"\s*$', r'",\s*$', r'",\d+\s*$',
            # Patterns for incomplete comparison operators
            r'>\s*$', r'>=\s*$', r'<>\s*$', r'!=\s*$',
            # Patterns for incomplete string literals with operators
            r"'[^']*>\s*$", r"'[^']*=\s*$", r"'[^']*<\s*$",
            # Patterns for incomplete UNION constructs
            r'\bUNIoN\s*\(\s*\(\s*$', r'\bUNIoN\s*\(\s*\(\s*\d+\s*$',
            # Patterns for incomplete quoted expressions
            r"'[^']*#\s*$", r'"[^"]*#\s*$',
            # Patterns for TABLE with parentheses and dots (common in SQL injection)
            r'\bTABLE\s*\)\s*$', r'\bTABLE\s*\)\s*\.\s*$',
            # More flexible patterns for keywords followed by punctuation
            r'\b[A-Z_]+\s*\)\s*$', r'\b[A-Z_]+\s*\)\s*\.\s*$',
            # Patterns for UNION with forward slash (SQL injection technique)
            r'\bUNION\s*/\s*$', r'\bUNiON\s*/\s*$', r'\buNiOn\s*/\s*$',
            # Patterns for UNION followed by keywords
            r'\bUNION\s*/\s*UNION\s*$', r'\bUNiON\s*/\s*UNiON\s*$',
            r'\bUNION\s*/\s*UNION\s+DISTINCT\s*$', r'\bUNiON\s*/\s*UNiON\s+DISTINCT\s*$',
            # Patterns for strings with forward slash followed by keywords
            r"'[^']*/'[A-Z_]+\s*$", r"'[^']*/[A-Z_]+\s*$"
        ]

    def _load_vocabulary(self, vocab_file: str) -> Set[str]:
        """
        Load vocabulary from file.

        Args:
            vocab_file: Path to the vocabulary file

        Returns:
            Set[str]: Set of vocabulary tokens
        """
        vocab = set()

        # Try to load from file
        if os.path.exists(vocab_file):
            try:
                with open(vocab_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        token = line.strip()
                        if token and not token.startswith('#'):  # Skip empty lines and comments
                            vocab.add(token.upper())
            except Exception as e:
                print(f"Warning: Could not load vocabulary from {vocab_file}: {e}")
                vocab = self._get_fallback_vocabulary()
        else:
            print(f"Warning: Vocabulary file {vocab_file} not found, using fallback vocabulary")
            vocab = self._get_fallback_vocabulary()

        return vocab

    def _get_fallback_vocabulary(self) -> Set[str]:
        """
        Get fallback vocabulary if file loading fails.

        Returns:
            Set[str]: Basic SQL vocabulary set
        """
        return {
            'SELECT', 'DISTINCT', 'ALL', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING',
            'ORDER', 'LIMIT', 'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT', 'WITH',
            'AS', 'ON', 'USING', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER',
            'CROSS', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIKE', 'BETWEEN', 'IS',
            'NULL', 'TRUE', 'FALSE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'LENGTH', 'SUBSTRING', 'CONCAT',
            'UPPER', 'LOWER', 'TRIM', 'COALESCE', 'ISNULL', 'SLEEP', 'BENCHMARK',
            'VERSION', 'DATABASE', 'USER', 'CURRENT_USER', 'MD5', 'SHA1', 'HEX',
            '=', '!=', '<>', '<', '>', '<=', '>=', '+', '-', '*', '/', '%', '||',
            '(', ')', ',', ';', '.', "'", '"', '--', '/*', '*/', '#',
            # Additional keywords for injection scenarios
            'COLLATE', 'BINARY', 'ASCII', 'UNICODE', 'TABLE', 'UNIoN', 'sELECt'
        }

    def _categorize_vocabulary(self):
        """
        Categorize loaded vocabulary into different types for easier access.
        """
        # SQL Keywords
        sql_keywords = {
            'SELECT', 'DISTINCT', 'ALL', 'FROM', 'WHERE', 'GROUP', 'BY', 'HAVING',
            'ORDER', 'LIMIT', 'OFFSET', 'UNION', 'INTERSECT', 'EXCEPT', 'WITH',
            'AS', 'ON', 'USING', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'FULL', 'OUTER',
            'CROSS', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'LIKE', 'BETWEEN', 'IS',
            'NULL', 'TRUE', 'FALSE', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP', 'GRANT',
            'REVOKE', 'COMMIT', 'ROLLBACK', 'BEGIN', 'TRANSACTION', 'INDEX',
            'TABLE', 'VIEW', 'DATABASE', 'SCHEMA', 'COLUMN', 'PRIMARY', 'FOREIGN',
            'KEY', 'CONSTRAINT', 'UNIQUE', 'CHECK', 'DEFAULT', 'REFERENCES',
            # Additional SQL keywords for injection scenarios
            'COLLATE', 'BINARY', 'ASCII', 'UNICODE', 'NOCASE', 'RTRIM', 'LOCALIZED',
            'UNIoN', 'sELECt', 'fRoM', 'wHeRe', 'aNd', 'oR', 'nOt'  # Case variations
        }

        # SQL Functions
        sql_functions = {
            'COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'LENGTH', 'SUBSTRING', 'CONCAT',
            'UPPER', 'LOWER', 'TRIM', 'COALESCE', 'ISNULL', 'SLEEP', 'BENCHMARK',
            'VERSION', 'CURRENT_USER', 'SESSION_USER', 'SYSTEM_USER', 'MD5', 'SHA1', 'HEX',
            'EXTRACTVALUE', 'UPDATEXML', 'XMLTYPE', 'CAST', 'CONVERT', 'CHAR', 'ASCII'
        }

        # Operators
        operators = {'=', '!=', '<>', '<', '>', '<=', '>=', '+', '-', '*', '/', '%', '||'}

        # Punctuation
        punctuation = {'(', ')', ',', ';', '.', "'", '"', '--', '/*', '*/', '#'}

        # Categorize vocabulary
        for token in self.vocab:
            if token in sql_keywords:
                self.keywords.add(token)
            elif token in sql_functions:
                self.functions.add(token)
            elif token in operators:
                self.operators.add(token)
            elif token in punctuation:
                self.punctuation.add(token)
            else:
                # Add to keywords by default for unknown tokens
                self.keywords.add(token)

    def is_potential_prefix(self, sql_text: str) -> bool:
        """
        Return True if sql_text could be extended into a syntactically valid SQL statement.

        Args:
            sql_text: The SQL text to check (may be incomplete)

        Returns:
            bool: True if the text is a valid prefix that can be extended
        """
        if not sql_text or not sql_text.strip():
            return True  # Empty string can always be extended

        sql_text = sql_text.strip()

        # Use the structural check for a fast path decision
        structural_verdict = self._structural_is_potential(sql_text)
        if structural_verdict is not None:
            return structural_verdict

        # Fallback to pattern-based and parser-based checks
        if self._has_invalid_syntax_patterns(sql_text):
            return False

        # Check for obvious incomplete patterns that are valid prefixes
        for pattern in self.incomplete_patterns:
            if re.search(pattern, sql_text, re.IGNORECASE):
                return True

        # Check for unclosed string literals (valid prefix)
        if self._has_unclosed_string(sql_text):
            return True

        # Check for unmatched parentheses (could be valid prefix)
        if self._has_unmatched_parens(sql_text):
            # But check if it's a reasonable unmatched paren situation
            if not self._is_reasonable_unmatched_parens(sql_text):
                return False
            return True

        # Try parsing with sqlglot
        try:
            parsed = sqlglot.parse_one(sql_text, error_level=None)
            if parsed is not None:
                return True
        except (ParseError, Exception):
            pass

        # Try common completions to see if it's a valid prefix
        return self._test_prefix_completions(sql_text)

    def is_complete_query(self, sql_text: str) -> bool:
        """
        Return True if sql_text is a complete, syntactically valid SQL statement.

        Args:
            sql_text: The SQL text to check

        Returns:
            bool: True if the text is complete and valid
        """
        if not sql_text or not sql_text.strip():
            return False

        sql_text = sql_text.strip()

        # Check for obvious incomplete patterns
        for pattern in self.incomplete_patterns:
            if re.search(pattern, sql_text, re.IGNORECASE):
                return False

        # Check for unclosed constructs
        if (self._has_unclosed_string(sql_text) or
            self._has_unmatched_parens(sql_text)):
            return False

        # Try parsing with sqlglot
        try:
            parsed = sqlglot.parse_one(sql_text, error_level=None)
            return parsed is not None
        except (ParseError, Exception):
            return False

    def suggest_next_tokens(self, sql_text: str, top_k: int = 20) -> List[str]:
        """
        Return candidate tokens that are reasonable continuations of sql_text.

        Args:
            sql_text: The current SQL text
            top_k: Maximum number of suggestions to return

        Returns:
            List[str]: Ordered list of suggested next tokens
        """
        if not sql_text:
            sql_text = ""

        sql_text = sql_text.strip()
        suggestions = []

        # Analyze context and generate context-specific suggestions
        context = self._analyze_context(sql_text)

        if context == 'empty':
            suggestions.extend(['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE'])
        elif context == 'after_union':
            suggestions.extend([' ALL', ' SELECT', ' DISTINCT', ' ('])
        elif context == 'after_select':
            suggestions.extend([' *', ' DISTINCT', ' COUNT(*)', ' 1', ' username', ' password'])
        elif context == 'after_from':
            suggestions.extend([' users', ' information_schema.tables', ' information_schema.columns', ' ('])
        elif context == 'after_where':
            suggestions.extend([' 1=1', ' id=1', ' username=', " '", ' (', ' NOT'])
        elif context == 'after_operator':
            suggestions.extend([' 1', " 'admin'", ' (SELECT', ' NULL', ' TRUE'])
        elif context == 'after_comma':
            suggestions.extend([' *', ' username', ' password', ' id', ' 1'])
        elif context == 'in_string':
            suggestions.extend(["'", "' OR '1'='1", "' UNION SELECT", "' AND 1=1--", "' OR 1=1#"])
        elif context == 'after_comment_start':
            suggestions.extend([' comment */', ' */'])
        elif context == 'after_join':
            suggestions.extend([' users', ' information_schema.tables', ' ('])
        else:
            # General suggestions based on current state
            suggestions.extend(self._get_general_suggestions(sql_text))

        # Add vocabulary-based suggestions
        suggestions.extend(self._get_vocab_suggestions(sql_text))

        # Add common SQL continuations from vocabulary
        suggestions.extend([' AND', ' OR', ' UNION', ' ORDER BY', ' GROUP BY', ' LIMIT'])
        suggestions.extend([' --', ' /*', ' */', ';', ')'])

        # Add function calls from vocabulary
        function_suggestions = [f' {func}(' for func in self.functions if func in ['SLEEP', 'BENCHMARK', 'VERSION', 'DATABASE']]
        suggestions.extend(function_suggestions)

        # Add injection-specific suggestions based on context
        if "'" in sql_text:
            suggestions.extend(list(self.injection_fragments))

        # Add whitespace variations for better token diversity
        whitespace_variants = []
        for suggestion in suggestions[:]:
            if not suggestion.startswith(' ') and suggestion not in [';', ')', ',']:
                whitespace_variants.append(' ' + suggestion)
        suggestions.extend(whitespace_variants)

        # Score and rank suggestions
        scored_suggestions = self._score_suggestions(sql_text, suggestions)

        # Remove duplicates while preserving order and limit results
        seen = set()
        unique_suggestions = []
        for suggestion, score in scored_suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)

        return unique_suggestions[:top_k]

    def _structural_is_potential(self, sql_text: str) -> Optional[bool]:
        """
        Fast structural check using SQL clause-aware FSM.
        Returns True/False when confident, None when undecided.
        """
        if not sql_text:
            return True

        # SQL clause states for FSM
        class SQLState:
            INITIAL = 'initial'
            SELECT_CLAUSE = 'select'
            FROM_CLAUSE = 'from'
            WHERE_CLAUSE = 'where'
            GROUP_BY_CLAUSE = 'group_by'
            HAVING_CLAUSE = 'having'
            ORDER_BY_CLAUSE = 'order_by'
            UNION_CLAUSE = 'union'
            JOIN_CLAUSE = 'join'
            IN_EXPRESSION = 'in_expr'
            IN_FUNCTION = 'in_func'
            IN_SUBQUERY = 'in_subquery'

        state = SQLState.INITIAL
        paren_depth = 0
        in_string = False
        in_comment = False
        comment_type = None  # 'line' or 'block'
        tokens = self._tokenize_sql(sql_text)

        if not tokens:
            return True

        i = 0
        while i < len(tokens):
            token = tokens[i]
            token_upper = token.upper()

            # Handle string/comment state
            if in_string:
                if token == "'" and (i == 0 or tokens[i-1] != '\\'):
                    in_string = False
                i += 1
                continue

            if in_comment:
                if comment_type == 'line' and token == '\n':
                    in_comment = False
                elif comment_type == 'block' and token == '*/':
                    in_comment = False
                i += 1
                continue

            # Handle comment/string start
            if token == "'":
                in_string = True
                i += 1
                continue
            elif token == '--' or token == '#':
                in_comment = True
                comment_type = 'line'
                i += 1
                continue
            elif token == '/*':
                # Special check for UNION/* pattern - this is invalid
                if i > 0 and tokens[i-1].upper() == 'UNION':
                    return False
                in_comment = True
                comment_type = 'block'
                i += 1
                continue

            # Handle parentheses
            if token == '(':
                paren_depth += 1
                if state in [SQLState.SELECT_CLAUSE, SQLState.WHERE_CLAUSE]:
                    state = SQLState.IN_SUBQUERY if paren_depth == 1 else state
            elif token == ')':
                if paren_depth == 0:
                    return False  # Unmatched closing paren
                paren_depth -= 1
                if paren_depth == 0 and state == SQLState.IN_SUBQUERY:
                    state = SQLState.SELECT_CLAUSE  # Return to outer query

            # FSM state transitions based on keywords
            elif token_upper in self.keywords:
                new_state = self._get_next_state(state, token_upper)
                if new_state is False:
                    return False  # Invalid transition
                elif new_state is not None:
                    state = new_state

            i += 1

        # Final state analysis
        return self._analyze_final_state(state, sql_text, in_string, in_comment, paren_depth)

    def _tokenize_sql(self, sql_text: str) -> List[str]:
        """Simple SQL tokenizer that preserves important structure."""
        tokens = []
        i = 0
        n = len(sql_text)

        while i < n:
            ch = sql_text[i]

            # Skip whitespace but preserve newlines for comment handling
            if ch.isspace():
                if ch == '\n':
                    tokens.append('\n')
                i += 1
                continue

            # Multi-character operators and comments
            if i < n - 1:
                two_char = sql_text[i:i+2]
                if two_char in ['--', '/*', '*/', '<=', '>=', '<>', '!=']:
                    tokens.append(two_char)
                    i += 2
                    continue

            # Single character tokens
            if ch in "(),'\"=<>+-*/;#.":
                tokens.append(ch)
                i += 1
                continue

            # Word tokens (identifiers, keywords, numbers)
            if ch.isalnum() or ch == '_':
                j = i
                while j < n and (sql_text[j].isalnum() or sql_text[j] == '_'):
                    j += 1
                tokens.append(sql_text[i:j])
                i = j
                continue

            # Skip unknown characters
            i += 1

        return tokens

    def _get_next_state(self, current_state, keyword):
        """Determine next FSM state based on current state and keyword."""
        # Define valid state transitions
        transitions = {
            'initial': {
                'SELECT': 'select', 'WITH': 'select', 'INSERT': 'select',
                'UPDATE': 'select', 'DELETE': 'select'
            },
            'select': {
                'FROM': 'from', 'WHERE': 'where', 'GROUP': 'group_by',
                'ORDER': 'order_by', 'UNION': 'union', 'HAVING': 'having'
            },
            'from': {
                'WHERE': 'where', 'GROUP': 'group_by', 'ORDER': 'order_by',
                'UNION': 'union', 'JOIN': 'join', 'INNER': 'join',
                'LEFT': 'join', 'RIGHT': 'join', 'FULL': 'join'
            },
            'where': {
                'GROUP': 'group_by', 'ORDER': 'order_by', 'UNION': 'union',
                'AND': 'where', 'OR': 'where'
            },
            'group_by': {
                'HAVING': 'having', 'ORDER': 'order_by', 'UNION': 'union'
            },
            'having': {
                'ORDER': 'order_by', 'UNION': 'union'
            },
            'order_by': {
                'UNION': 'union'
            },
            'union': {
                'SELECT': 'select', 'ALL': 'union'
            },
            'join': {
                'ON': 'where', 'WHERE': 'where', 'JOIN': 'join',
                'INNER': 'join', 'LEFT': 'join', 'RIGHT': 'join'
            }
        }

        return transitions.get(current_state, {}).get(keyword)

    def _analyze_final_state(self, state, sql_text, in_string, in_comment, paren_depth):
        """Analyze if the final state represents a valid prefix."""
        # Always allow if we're in an open construct
        if in_string or in_comment or paren_depth > 0:
            return True

        sql_upper = sql_text.upper().strip()

        # Reject obvious invalid patterns
        if '..' in sql_text:
            return False
        if re.search(r'UNION\s*/\s*\*', sql_upper) and not re.search(r'\*/\s*$', sql_upper):
            return False
        if sql_upper.endswith('UNION/*') or sql_upper.endswith('UNION /*'):
            return False
        if re.search(r'\bFROM\s+FROM\b', sql_upper):
            return False
        if re.search(r'\bWHERE\s+WHERE\b', sql_upper) and not re.search(r'\bSELECT\b', sql_upper):
            return False

        # Check for trailing operators that expect continuation
        if re.search(r'(AND|OR|=|<|>|LIKE|IN|BETWEEN|IS)\s*$', sql_upper):
            return True

        # Allow incomplete clauses
        if re.search(r'(ORDER|GROUP|UNION|JOIN)\s*$', sql_upper):
            return True

        # Allow after keywords that expect more
        if state in ['select', 'from', 'where', 'union', 'join']:
            return True

        # Default to allowing (conservative approach)
        return None

    def _get_vocab_suggestions(self, sql_text: str) -> List[str]:
        """
        Get suggestions based on loaded vocabulary.

        Args:
            sql_text: Current SQL text

        Returns:
            List[str]: Vocabulary-based suggestions
        """
        suggestions = []

        # Get the last token to check for prefix matching
        last_token = self._get_last_token(sql_text)

        # Find vocabulary tokens that start with the last token (prefix matching)
        if last_token:
            for token in self.vocab:
                if token.upper().startswith(last_token.upper()) and token.upper() != last_token.upper():
                    # Add the remaining part of the token
                    remaining = token[len(last_token):]
                    suggestions.append(remaining)

        # Add random vocabulary tokens for diversity
        vocab_list = list(self.vocab)
        import random
        random.shuffle(vocab_list)
        suggestions.extend([f' {token}' for token in vocab_list[:10]])

        return suggestions

    def _get_last_token(self, sql_text: str) -> str:
        """
        Extract the last token from SQL text.

        Args:
            sql_text: SQL text

        Returns:
            str: Last token or empty string
        """
        if not sql_text:
            return ""

        # Split by common delimiters
        tokens = re.split(r'[\s\(\),;\'"]', sql_text)

        # Get the last non-empty token
        for token in reversed(tokens):
            if token.strip():
                return token.strip()

        return ""

    def _is_prefix_of_known_keyword(self, token: str) -> bool:
        """
        Check if token is a prefix of any known keyword in vocabulary.

        Args:
            token: Token to check

        Returns:
            bool: True if token is a prefix of a known keyword
        """
        if not token:
            return False

        token_upper = token.upper()

        for vocab_token in self.vocab:
            if vocab_token.startswith(token_upper) and len(vocab_token) > len(token_upper):
                return True

        return False

    def _score_suggestions(self, sql_text: str, suggestions: List[str]) -> List[tuple]:
        """
        Score and rank suggestions based on context and vocabulary.

        Args:
            sql_text: Current SQL text
            suggestions: List of suggestions to score

        Returns:
            List[tuple]: List of (suggestion, score) tuples sorted by score
        """
        scored = []

        for suggestion in suggestions:
            score = self._score_candidate(sql_text, suggestion)
            scored.append((suggestion, score))

        # Sort by score (higher is better)
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _score_candidate(self, sql_text: str, candidate: str) -> float:
        """
        Score a candidate suggestion.

        Args:
            sql_text: Current SQL text
            candidate: Candidate suggestion

        Returns:
            float: Score for the candidate (higher is better)
        """
        score = 0.0

        # Base score for vocabulary tokens
        candidate_clean = candidate.strip().upper()
        if candidate_clean in self.vocab:
            score += 10.0

        # Bonus for keywords
        if candidate_clean in self.keywords:
            score += 5.0

        # Bonus for functions
        if candidate_clean in self.functions:
            score += 3.0

        # Bonus for operators
        if candidate_clean in self.operators:
            score += 2.0

        # Bonus for injection fragments
        if candidate in self.injection_fragments:
            score += 8.0

        # Context-based scoring
        context = self._analyze_context(sql_text)

        if context == 'after_select' and candidate_clean in ['*', 'DISTINCT', 'COUNT']:
            score += 15.0
        elif context == 'after_from' and candidate_clean in ['USERS', 'INFORMATION_SCHEMA']:
            score += 15.0
        elif context == 'after_where' and candidate_clean in ['1=1', 'TRUE', 'FALSE']:
            score += 15.0
        elif context == 'after_union' and candidate_clean in ['SELECT', 'ALL']:
            score += 15.0

        # Penalty for very long suggestions
        if len(candidate) > 50:
            score -= 5.0

        return score

    def _has_unclosed_string(self, sql_text: str) -> bool:
        """Check if there's an unclosed string literal in valid context."""
        # Count quotes, but be more permissive for SQL injection scenarios
        single_quotes = sql_text.count("'")
        double_quotes = sql_text.count('"')

        # If we have unclosed quotes, check if it's in a reasonable context
        has_unclosed_single = (single_quotes % 2 != 0)
        has_unclosed_double = (double_quotes % 2 != 0)

        if not (has_unclosed_single or has_unclosed_double):
            return False

        # For SQL injection scenarios, be more permissive with unclosed strings
        # Check if the unclosed string contains potential injection patterns
        if has_unclosed_single:
            last_quote_pos = sql_text.rfind("'")
            if last_quote_pos >= 0:
                after_quote = sql_text[last_quote_pos + 1:].strip()
                # Allow common injection patterns after quotes
                injection_patterns = [
                    r'^\d+', r'^>', r'^=', r'^<', r'^!', r'^#', r'^\s*$',
                    r'^OR\b', r'^AND\b', r'^UNION\b', r'^SELECT\b'
                ]
                for pattern in injection_patterns:
                    if re.match(pattern, after_quote, re.IGNORECASE):
                        return True

        if has_unclosed_double:
            last_quote_pos = sql_text.rfind('"')
            if last_quote_pos >= 0:
                after_quote = sql_text[last_quote_pos + 1:].strip()
                # Similar patterns for double quotes
                injection_patterns = [
                    r'^\d+', r'^>', r'^=', r'^<', r'^!', r'^#', r'^\s*$',
                    r'^OR\b', r'^AND\b', r'^UNION\b', r'^SELECT\b'
                ]
                for pattern in injection_patterns:
                    if re.match(pattern, after_quote, re.IGNORECASE):
                        return True

        return has_unclosed_single or has_unclosed_double

    def _has_unmatched_parens(self, sql_text: str) -> bool:
        """Check if there are unmatched parentheses."""
        open_count = sql_text.count('(')
        close_count = sql_text.count(')')
        return open_count != close_count

    def _test_prefix_completions(self, sql_text: str) -> bool:
        """Test if the text is a valid prefix by trying completions."""
        completions = [
            f"{sql_text} 1",
            f"{sql_text} SELECT 1",
            f"{sql_text} users",
            f"{sql_text} *",
            f"{sql_text}'",
            f"{sql_text})"
        ]

        for completion in completions:
            try:
                parsed = sqlglot.parse_one(completion, error_level=None)
                if parsed is not None:
                    return True
            except (ParseError, Exception):
                continue

        return False

    def _analyze_context(self, sql_text: str) -> str:
        """Analyze the current context to determine appropriate suggestions."""
        if not sql_text:
            return 'empty'

        sql_upper = sql_text.upper().strip()

        # Check for specific contexts in order of specificity
        # Handle UNION with forward slash patterns (SQL injection)
        if re.search(r'\bUNION\s*/\s*$', sql_upper):
            return 'after_union'
        elif re.search(r'\bUNION\s*/\s*UNION\s*$', sql_upper):
            return 'after_union'
        elif re.search(r'\bUNION\s*/\s*UNION\s+DISTINCT\s*$', sql_upper):
            return 'after_select'
        elif re.search(r'\bUNION\s*$', sql_upper):
            return 'after_union'
        elif re.search(r'\bUNION\s+ALL\s*$', sql_upper):
            return 'after_union'
        elif re.search(r'\bDISTINCT\s*$', sql_upper):
            return 'after_select'
        elif re.search(r'\bSELECT\s*$', sql_upper):
            return 'after_select'
        elif re.search(r'\bFROM\s*$', sql_upper):
            return 'after_from'
        elif re.search(r'\bWHERE\s*$', sql_upper):
            return 'after_where'
        elif re.search(r'\bJOIN\s*$', sql_upper):
            return 'after_join'
        elif re.search(r'\b(INNER|LEFT|RIGHT|FULL|CROSS)\s+JOIN\s*$', sql_upper):
            return 'after_join'
        elif re.search(r'[=<>!]+\s*$', sql_upper):
            return 'after_operator'
        elif re.search(r',\s*$', sql_upper):
            return 'after_comma'
        elif self._has_unclosed_string(sql_text):
            return 'in_string'
        elif re.search(r'/\*\s*$', sql_upper):
            return 'after_comment_start'
        elif re.search(r'\bAND\s*$', sql_upper):
            return 'after_where'
        elif re.search(r'\bOR\s*$', sql_upper):
            return 'after_where'
        # Handle string patterns with forward slash
        elif re.search(r"'[^']*/'[A-Z_]*\s*$", sql_upper):
            return 'in_string'
        else:
            return 'general'

    def _has_invalid_syntax_patterns(self, sql_text: str) -> bool:
        """Check for syntax patterns that are definitely invalid and cannot be extended."""
        sql_upper = sql_text.upper()

        # Check keyword sequence validity first
        if not self._is_valid_keyword_sequence(sql_text):
            return True

        # Disallow unclosed block comments
        if '/*' in sql_text and '*/' not in sql_text:
            return True

        # UNION followed by * without space is invalid
        if re.search(r'\bUNION\*', sql_upper):
            return True

        # Multiple consecutive operators without operands
        if re.search(r'[+\-]{3,}', sql_text):  # 3+ consecutive +/- operators
            return True

        # Invalid operator combinations - */ without proper comment context
        if re.search(r'\*/', sql_text):
            # Allow */ only if it's closing a /* comment
            if not re.search(r'/\*.*\*/', sql_text, re.DOTALL):
                return True

        # Double dots anywhere are suspicious (e.g., ..ON)
        if re.search(r'\.\.', sql_text):
            return True

        # Repeated forward slashes after empty string
        if re.search(r"''\s*//", sql_text):
            return True

        # Forward slash connecting alphanumeric tokens (except UNION/)
        if re.search(r'[A-Z0-9]\s*/\s*[A-Z0-9]', sql_upper):
            if not re.search(r'\bUNION\s*/', sql_upper):
                return True

        # Invalid parentheses sequences
        if re.search(r'\)\s*\(\s*\)\s*\(', sql_text):  # )()( pattern
            return True

        # Function calls with invalid syntax
        if re.search(r'\b[A-Z_]+\s*\(\s*[A-Z_]+\s*\(\s*[A-Z_]+\s*\(\s*[A-Z_]+\s*\(\s*[A-Z_]+', sql_text, re.IGNORECASE):
            return True

        # Numbers immediately followed by quoted identifiers without delimiter
        if re.search(r'[0-9]"\s*[A-Z_]', sql_upper):
            return True

        # ORDER BY followed by number then quote is invalid
        if re.search(r'\bORDER\s+BY\s+[^,)]*"\s*[A-Z_]', sql_upper):
            return True

        # Multiple unclosed quotes of different types
        single_quotes = sql_text.count("'") - sql_text.count("\\'")
        double_quotes = sql_text.count('"') - sql_text.count('\\"')
        if (single_quotes % 2 == 1) and (double_quotes % 2 == 1):  # Both unclosed
            return True

        return False

    def _is_valid_keyword_sequence(self, sql_text: str) -> bool:
        """Check if keywords are properly spaced."""
        sql_upper = sql_text.upper()
        known_keywords = self.keywords.union(self.functions)

        # Look for sequences of 3+ keywords concatenated without delimiters
        # This catches patterns like ONXMLTYPEACTIONCONTINUE
        for keyword1 in known_keywords:
            if len(keyword1) >= 3:  # Only check reasonably long keywords
                for keyword2 in known_keywords:
                    if len(keyword2) >= 3:
                        for keyword3 in known_keywords:
                            if len(keyword3) >= 3:
                                # Check if all three keywords appear concatenated
                                concatenated = keyword1 + keyword2 + keyword3
                                if concatenated in sql_upper:
                                    # Make sure they're not properly spaced in the original
                                    spaced_version = f"{keyword1} {keyword2} {keyword3}"
                                    if spaced_version not in sql_upper:
                                        return False

        # Check for specific problematic two-keyword patterns
        problematic_pairs = [
            ('CALL', 'UNDO'), ('CALL', 'DISTINCT'), ('CALL', 'SELECT'),
            ('UNDO', 'DISTINCT'), ('UNDO', 'SELECT'),
            ('DISTINCT', 'SELECT'),
            ('ON', 'XMLTYPE'), ('XMLTYPE', 'ACTION'), ('ACTION', 'CONTINUE')
        ]

        for keyword1, keyword2 in problematic_pairs:
            concatenated = keyword1 + keyword2
            if concatenated in sql_upper:
                # Check if they appear spaced in the original
                spaced_version = f"{keyword1} {keyword2}"
                if spaced_version not in sql_upper:
                    return False

        return True

    def _is_reasonable_unmatched_parens(self, sql_text: str) -> bool:
        """Check if unmatched parentheses are in a reasonable context."""
        # Count opening and closing parens
        open_count = sql_text.count('(')
        close_count = sql_text.count(')')

        # If more closing than opening, it's definitely invalid
        if close_count > open_count:
            return False

        # If way too many unmatched opening parens, likely invalid
        if open_count - close_count > 3:
            return False

        # Check if the unmatched parens are in reasonable contexts
        # Look for function calls or subqueries
        if re.search(r'\b[A-Z_]+\s*\(\s*$', sql_text, re.IGNORECASE):  # function(
            return True
        if re.search(r'\(\s*SELECT\b', sql_text, re.IGNORECASE):  # (SELECT
            return True
        if re.search(r'\bIN\s*\(\s*$', sql_text, re.IGNORECASE):  # IN (
            return True

        # If unmatched parens but no clear function/subquery context, be suspicious
        return open_count - close_count <= 1

    def _get_general_suggestions(self, sql_text: str) -> List[str]:
        """Get general suggestions based on current query state."""
        suggestions = []
        sql_upper = sql_text.upper()

        # Add keywords that make sense in current context
        if 'SELECT' in sql_upper and 'FROM' not in sql_upper:
            suggestions.append(' FROM')
        if 'FROM' in sql_upper and 'WHERE' not in sql_upper:
            suggestions.append(' WHERE')
        if 'WHERE' in sql_upper and 'ORDER' not in sql_upper:
            suggestions.extend([' ORDER BY', ' GROUP BY'])
        if 'SELECT' in sql_upper and 'UNION' not in sql_upper:
            suggestions.append(' UNION')

        return suggestions


if __name__ == "__main__":
    # Test/demo block
    validator = SQLPrefixValidator()

    test_cases = [
        # Base query
        validator.base_query,
        # Incomplete prefix
       "UNioN",
        "UNioN ",
        "UNioN ALL",
        "UNioN ALL(",
        "UNioN ALL(LEFT",
        "UNioN ALL(LEFT ",
        "UNioN ALL(LEFT DECLARE",
        "UNioN ALL(LEFT DECLARE ",
        "UNioN ALL(LEFT DECLARE ROUTINE",
        "UNioN ALL(LEFT DECLARE ROUTINEANY",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 ",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE ",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 ",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM ",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1'",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1')",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1');",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1');DO",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1');DO ",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1');DO (",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1');DO ( ",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1');DO ( /",
        "UNioN ALL(LEFT DECLARE ROUTINEANY(3 )LIKE 0 FROM 1');DO ( //"
    ]

    print("SQL Prefix Validator Demo")
    print("=" * 50)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: '{test_case}'")
        print("-" * 40)

        is_potential = validator.is_potential_prefix(test_case)
        is_complete = validator.is_complete_query(test_case)
        suggestions = validator.suggest_next_tokens(test_case, top_k=10)

        print(f"Is potential prefix: {is_potential}")
        print(f"Is complete query: {is_complete}")
        print(f"Next token suggestions: {suggestions}")