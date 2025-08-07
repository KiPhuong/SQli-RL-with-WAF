"""
Logging utilities for the SQL injection RL framework
"""

import logging
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class CustomFormatter(logging.Formatter):
    """
    Custom log formatter with color support and structured formatting.
    """
    
    # Color codes for different log levels
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, use_colors: bool = True, include_timestamp: bool = True):
        """
        Initialize the custom formatter.
        
        Args:
            use_colors: Whether to use color codes
            include_timestamp: Whether to include timestamp
        """
        self.use_colors = use_colors
        self.include_timestamp = include_timestamp
        
        # Define format string
        if include_timestamp:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        else:
            fmt = '%(name)s - %(levelname)s - %(message)s'
        
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """
        Format the log record with colors and structure.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message
        """
        # Store original format
        original_format = self._style._fmt
        
        # Apply colors if enabled
        if self.use_colors and record.levelname in self.COLORS:
            color = self.COLORS[record.levelname]
            reset = self.COLORS['RESET']
            
            # Color the level name
            colored_levelname = f"{color}{record.levelname}{reset}"
            record.levelname = colored_levelname
        
        # Format the record
        formatted = super().format(record)
        
        # Restore original format
        self._style._fmt = original_format
        
        return formatted


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    
    def format(self, record):
        """
        Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log message
        """
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['extra'] = record.extra_data
        
        return json.dumps(log_entry)


class SQLInjectionLogger:
    """
    Specialized logger for SQL injection testing activities.
    """
    
    def __init__(self, name: str = 'sqli_rl', log_dir: str = 'logs',
                 log_level: str = 'INFO', max_file_size: int = 10485760,
                 backup_count: int = 5):
        """
        Initialize the SQL injection logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Logging level
            max_file_size: Maximum size for log files in bytes
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize loggers
        self.main_logger = self._setup_main_logger()
        self.payload_logger = self._setup_payload_logger()
        self.performance_logger = self._setup_performance_logger()
        self.security_logger = self._setup_security_logger()
    
    def _setup_main_logger(self) -> logging.Logger:
        """
        Setup the main application logger.
        
        Returns:
            Configured main logger
        """
        logger = logging.getLogger(f"{self.name}.main")
        logger.setLevel(self.log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_formatter = CustomFormatter(use_colors=True)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_path = os.path.join(self.log_dir, 'application.log')
        file_handler = RotatingFileHandler(
            file_path, 
            maxBytes=self.max_file_size, 
            backupCount=self.backup_count
        )
        file_handler.setLevel(self.log_level)
        file_formatter = CustomFormatter(use_colors=False)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_payload_logger(self) -> logging.Logger:
        """
        Setup logger for payload testing activities.
        
        Returns:
            Configured payload logger
        """
        logger = logging.getLogger(f"{self.name}.payloads")
        logger.setLevel(logging.DEBUG)  # Always debug level for payloads
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # JSON file handler for structured payload logs
        file_path = os.path.join(self.log_dir, 'payloads.json')
        file_handler = RotatingFileHandler(
            file_path, 
            maxBytes=self.max_file_size, 
            backupCount=self.backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_performance_logger(self) -> logging.Logger:
        """
        Setup logger for performance metrics.
        
        Returns:
            Configured performance logger
        """
        logger = logging.getLogger(f"{self.name}.performance")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Time-based rotating file handler
        file_path = os.path.join(self.log_dir, 'performance.log')
        file_handler = TimedRotatingFileHandler(
            file_path, 
            when='midnight', 
            interval=1, 
            backupCount=30
        )
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s,%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _setup_security_logger(self) -> logging.Logger:
        """
        Setup logger for security events and findings.
        
        Returns:
            Configured security logger
        """
        logger = logging.getLogger(f"{self.name}.security")
        logger.setLevel(logging.WARNING)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # JSON file handler for security events
        file_path = os.path.join(self.log_dir, 'security_events.json')
        file_handler = RotatingFileHandler(
            file_path, 
            maxBytes=self.max_file_size, 
            backupCount=self.backup_count
        )
        file_handler.setLevel(logging.WARNING)
        json_formatter = JsonFormatter()
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def log_payload_attempt(self, payload: str, response: Dict[str, Any], 
                           analysis: Dict[str, Any], action: int, reward: float):
        """
        Log a payload attempt with full details.
        
        Args:
            payload: SQL injection payload
            response: HTTP response information
            analysis: Response analysis results
            action: Action taken by agent
            reward: Reward received
        """
        extra_data = {
            'event_type': 'payload_attempt',
            'payload': payload,
            'action': action,
            'reward': reward,
            'response': {
                'status_code': response.get('status_code'),
                'response_time': response.get('response_time'),
                'content_length': response.get('content_length', 0),
                'url': response.get('url')
            },
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create log record with extra data
        record = logging.LogRecord(
            name=self.payload_logger.name,
            level=logging.INFO,
            pathname='',
            lineno=0,
            msg='Payload attempt',
            args=(),
            exc_info=None
        )
        record.extra_data = extra_data
        
        self.payload_logger.handle(record)
    
    def log_injection_success(self, payload: str, injection_type: str, 
                             severity: str, data_extracted: Optional[str] = None):
        """
        Log successful SQL injection discovery.
        
        Args:
            payload: Successful payload
            injection_type: Type of injection (union, blind, error, etc.)
            severity: Severity level
            data_extracted: Any data extracted from the injection
        """
        extra_data = {
            'event_type': 'injection_success',
            'payload': payload,
            'injection_type': injection_type,
            'severity': severity,
            'data_extracted': data_extracted,
            'timestamp': datetime.now().isoformat()
        }
        
        record = logging.LogRecord(
            name=self.security_logger.name,
            level=logging.CRITICAL,
            pathname='',
            lineno=0,
            msg=f'SQL Injection Found: {injection_type}',
            args=(),
            exc_info=None
        )
        record.extra_data = extra_data
        
        self.security_logger.handle(record)
        
        # Also log to main logger
        self.main_logger.critical(
            f"SQL Injection discovered! Type: {injection_type}, "
            f"Severity: {severity}, Payload: {payload}"
        )
    
    def log_waf_detection(self, waf_type: str, confidence: float, 
                         detection_method: str):
        """
        Log WAF detection event.
        
        Args:
            waf_type: Type of WAF detected
            confidence: Detection confidence score
            detection_method: Method used for detection
        """
        extra_data = {
            'event_type': 'waf_detection',
            'waf_type': waf_type,
            'confidence': confidence,
            'detection_method': detection_method,
            'timestamp': datetime.now().isoformat()
        }
        
        record = logging.LogRecord(
            name=self.security_logger.name,
            level=logging.WARNING,
            pathname='',
            lineno=0,
            msg=f'WAF Detected: {waf_type}',
            args=(),
            exc_info=None
        )
        record.extra_data = extra_data
        
        self.security_logger.handle(record)
        
        self.main_logger.warning(
            f"WAF detected: {waf_type} (confidence: {confidence:.2f})"
        )
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """
        Log performance metrics.
        
        Args:
            metrics: Performance metrics dictionary
        """
        # Format metrics as CSV-like string for easy parsing
        metric_parts = []
        for key, value in metrics.items():
            metric_parts.append(f"{key}={value}")
        
        metrics_str = ','.join(metric_parts)
        self.performance_logger.info(metrics_str)
    
    def log_agent_decision(self, state: Dict[str, Any], action: int, 
                          q_values: List[float], exploration_method: str):
        """
        Log agent decision-making process.
        
        Args:
            state: Current state
            action: Selected action
            q_values: Q-values for all actions
            exploration_method: Exploration method used
        """
        extra_data = {
            'event_type': 'agent_decision',
            'state_summary': {
                'attempt_count': state.get('attempt_count', 0),
                'waf_detected': state.get('waf_detected', False),
                'injection_detected': state.get('injection_detected', False)
            },
            'action_selected': action,
            'q_values': q_values,
            'exploration_method': exploration_method,
            'timestamp': datetime.now().isoformat()
        }
        
        record = logging.LogRecord(
            name=self.main_logger.name,
            level=logging.DEBUG,
            pathname='',
            lineno=0,
            msg=f'Agent selected action {action}',
            args=(),
            exc_info=None
        )
        record.extra_data = extra_data
        
        self.main_logger.handle(record)
    
    def log_training_episode(self, episode: int, total_reward: float, 
                           steps: int, success: bool, exploration_rate: float):
        """
        Log training episode completion.
        
        Args:
            episode: Episode number
            total_reward: Total reward for episode
            steps: Number of steps taken
            success: Whether episode was successful
            exploration_rate: Current exploration rate
        """
        self.main_logger.info(
            f"Episode {episode} completed: "
            f"Reward={total_reward:.2f}, Steps={steps}, "
            f"Success={success}, Exploration={exploration_rate:.3f}"
        )
        
        # Log performance metrics
        metrics = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps,
            'success': int(success),
            'exploration_rate': exploration_rate
        }
        self.log_performance_metrics(metrics)
    
    def log_error(self, error_type: str, error_message: str, 
                 context: Optional[Dict[str, Any]] = None):
        """
        Log error with context.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context information
        """
        extra_data = {
            'event_type': 'error',
            'error_type': error_type,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        record = logging.LogRecord(
            name=self.main_logger.name,
            level=logging.ERROR,
            pathname='',
            lineno=0,
            msg=f'{error_type}: {error_message}',
            args=(),
            exc_info=None
        )
        record.extra_data = extra_data
        
        self.main_logger.handle(record)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about logged events.
        
        Returns:
            Log statistics
        """
        stats = {
            'log_files': {},
            'total_events': 0,
            'event_types': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Check log files
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.log') or filename.endswith('.json'):
                file_path = os.path.join(self.log_dir, filename)
                file_stats = os.stat(file_path)
                stats['log_files'][filename] = {
                    'size_bytes': file_stats.st_size,
                    'modified': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
                }
        
        return stats
    
    def export_logs(self, start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None, 
                   event_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Export logs within time range and event types.
        
        Args:
            start_time: Start time for export
            end_time: End time for export
            event_types: List of event types to include
            
        Returns:
            List of log entries
        """
        exported_logs = []
        
        # Read JSON log files (payloads and security events)
        json_files = ['payloads.json', 'security_events.json']
        
        for filename in json_files:
            file_path = os.path.join(self.log_dir, filename)
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        for line in f:
                            try:
                                log_entry = json.loads(line.strip())
                                
                                # Check time range
                                log_time = datetime.fromisoformat(
                                    log_entry.get('timestamp', '').replace('Z', '+00:00')
                                )
                                
                                if start_time and log_time < start_time:
                                    continue
                                if end_time and log_time > end_time:
                                    continue
                                
                                # Check event types
                                if event_types:
                                    event_type = log_entry.get('extra', {}).get('event_type')
                                    if event_type not in event_types:
                                        continue
                                
                                exported_logs.append(log_entry)
                                
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    continue
        
        return exported_logs
    
    def cleanup_old_logs(self, days_to_keep: int = 30):
        """
        Clean up log files older than specified days.
        
        Args:
            days_to_keep: Number of days to keep logs
        """
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        for filename in os.listdir(self.log_dir):
            file_path = os.path.join(self.log_dir, filename)
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_time:
                    try:
                        os.remove(file_path)
                        self.main_logger.info(f"Removed old log file: {filename}")
                    except Exception as e:
                        self.main_logger.error(f"Failed to remove {filename}: {e}")
    
    def set_log_level(self, level: str):
        """
        Set logging level for all loggers.
        
        Args:
            level: New logging level
        """
        new_level = getattr(logging, level.upper())
        self.main_logger.setLevel(new_level)
        
        # Update console handler level
        for handler in self.main_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler):
                handler.setLevel(new_level)


# Global logger instance
_global_logger = None


def get_logger(name: str = 'sqli_rl') -> SQLInjectionLogger:
    """
    Get or create global logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        Global logger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = SQLInjectionLogger(name)
    return _global_logger


def setup_logging(log_dir: str = 'logs', log_level: str = 'INFO') -> SQLInjectionLogger:
    """
    Setup logging for the application.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    _global_logger = SQLInjectionLogger(log_dir=log_dir, log_level=log_level)
    return _global_logger
