#!/usr/bin/env python3
"""
Test suite for the structural SQL prefix validator.
Tests various edge cases and problematic patterns.
"""

from sql_prefix_validator import SQLPrefixValidator

def test_structural_validator():
    """Test the structural validator with various SQL prefixes."""
    validator = SQLPrefixValidator()
    
    # Test cases: (sql_prefix, expected_result, description)
    test_cases = [
        # Basic valid prefixes
        ("", True, "Empty string"),
        ("SELECT", True, "Simple SELECT"),
        ("SELECT *", True, "SELECT with asterisk"),
        ("SELECT * FROM", True, "SELECT FROM incomplete"),
        ("SELECT * FROM users WHERE", True, "WHERE clause incomplete"),
        ("SELECT * FROM users WHERE id =", True, "Incomplete comparison"),
        
        # Valid incomplete constructs
        ("SELECT * FROM users WHERE id = '", True, "Unclosed string"),
        ("SELECT * FROM (", True, "Unclosed parenthesis"),
        ("SELECT /* comment", True, "Unclosed block comment"),
        ("SELECT -- comment", True, "Line comment"),
        ("UNION", True, "UNION without SELECT"),
        ("ORDER", True, "ORDER without BY"),
        ("GROUP", True, "GROUP without BY"),
        
        # Invalid constructs that cannot be extended
        ("SELECT * FROM users)", False, "Unmatched closing paren"),
        ("..", False, "Double dots"),
        ("SELECT..FROM", False, "Double dots in query"),
        ("UNION/*", False, "UNION with unclosed comment"),
        ("LIKE ''//", False, "Invalid slash after string"),
        
        # Complex valid prefixes
        ("SELECT * FROM users WHERE id IN (", True, "IN with open paren"),
        ("SELECT COUNT(*) FROM users GROUP BY", True, "GROUP BY incomplete"),
        ("SELECT * FROM users ORDER BY id", True, "Complete ORDER BY"),
        ("SELECT * FROM users UNION", True, "UNION incomplete"),
        ("SELECT * FROM users UNION SELECT", True, "UNION SELECT incomplete"),
        
        # Edge cases with operators
        ("SELECT * WHERE id =", True, "Trailing equals"),
        ("SELECT * WHERE id <", True, "Trailing less than"),
        ("SELECT * WHERE id LIKE", True, "Trailing LIKE"),
        ("SELECT * WHERE id BETWEEN", True, "Trailing BETWEEN"),
        ("SELECT * WHERE id IS", True, "Trailing IS"),
        ("SELECT * WHERE id AND", True, "Trailing AND"),
        ("SELECT * WHERE id OR", True, "Trailing OR"),
        
        # Function calls
        ("SELECT VERSION(", True, "Function call incomplete"),
        ("SELECT SLEEP(", True, "SLEEP function incomplete"),
        ("SELECT SUBSTRING(", True, "SUBSTRING function incomplete"),
        
        # Injection patterns that should be valid prefixes
        ("' OR '1'='", True, "SQL injection pattern"),
        ("' UNION SELECT", True, "UNION injection"),
        ("' AND 1=1--", True, "Comment injection"),
        ("1' OR '1'='1", True, "Classic injection"),
        
        # Mixed quotes and comments
        ("SELECT 'test", True, "Unclosed single quote"),
        ('SELECT "test', True, "Unclosed double quote"),
        ("SELECT /* test */ 'unclosed", True, "Comment then unclosed string"),
        
        # Subqueries
        ("SELECT * FROM (SELECT", True, "Nested SELECT"),
        ("SELECT * FROM (SELECT * FROM users", True, "Incomplete subquery"),
        ("SELECT * FROM (SELECT * FROM users)", True, "Complete subquery"),
        
        # JOIN patterns
        ("SELECT * FROM users JOIN", True, "Incomplete JOIN"),
        ("SELECT * FROM users LEFT JOIN", True, "Incomplete LEFT JOIN"),
        ("SELECT * FROM users JOIN other ON", True, "JOIN with incomplete ON"),
        
        # Numbers and identifiers
        ("SELECT 1", True, "Number literal"),
        ("SELECT 1,", True, "Number with comma"),
        ("SELECT user_id", True, "Identifier with underscore"),
        ("SELECT `quoted_id`", True, "Backtick quoted identifier"),
        
        # Error cases that should be rejected
        ("SELECT FROM FROM", False, "Double FROM"),
        ("WHERE WHERE", False, "Double WHERE without SELECT"),
        (")", False, "Standalone closing paren"),
        ("*/", False, "Closing comment without opening"),
    ]
    
    print("Testing structural SQL prefix validator...")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for sql_prefix, expected, description in test_cases:
        try:
            result = validator.is_potential_prefix(sql_prefix)
            status = "PASS" if result == expected else "FAIL"

            if result == expected:
                passed += 1
            else:
                failed += 1
                # Debug the failing case
                if sql_prefix == "UNION/*":
                    print(f"DEBUG: UNION/* case - structural result: {validator._structural_is_potential(sql_prefix)}")

            print(f"{status:4} | {result:5} | {expected:5} | {description:30} | '{sql_prefix}'")

        except Exception as e:
            failed += 1
            print(f"ERROR | Exception occurred for '{sql_prefix}': {e}")
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    return failed == 0

if __name__ == "__main__":
    success = test_structural_validator()
    exit(0 if success else 1)
