"""
Custom exceptions for Market Monitor.

Exception Hierarchy:
    MarketMonitorError (Base)
    ├── ConfigError
    │   ├── ConfigNotFoundError
    │   ├── ConfigValidationError
    │   └── MissingRequiredKeyError
    ├── DataLoadError
    │   ├── FileNotFoundError (built-in)
    │   ├── InvalidFormatError
    │   └── MissingColumnError
    └── AnalysisError
        ├── InsufficientDataError
        ├── NetworkConstructionError
        └── ComputationError
"""

from typing import Optional, List


class MarketMonitorError(Exception):
    """Base exception for all Market Monitor errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


# =============================================================================
# Config Errors
# =============================================================================

class ConfigError(MarketMonitorError):
    """Configuration related errors."""
    pass


class ConfigNotFoundError(ConfigError):
    """Config file not found."""

    def __init__(self, path: str):
        super().__init__(
            f"Configuration file not found: {path}",
            "Please provide a valid config file path or use default configuration."
        )
        self.path = path


class ConfigValidationError(ConfigError):
    """Config validation failed."""

    def __init__(self, errors: List[str]):
        self.errors = errors
        error_list = "\n  - ".join(errors)
        super().__init__(
            f"Configuration validation failed with {len(errors)} error(s)",
            f"Errors:\n  - {error_list}"
        )


class MissingRequiredKeyError(ConfigError):
    """Required config key is missing."""

    def __init__(self, key: str, section: Optional[str] = None):
        self.key = key
        self.section = section
        location = f"in section '{section}'" if section else "in config"
        super().__init__(
            f"Required key '{key}' is missing {location}",
            "Please add the required key to your configuration file."
        )


# =============================================================================
# Data Load Errors
# =============================================================================

class DataLoadError(MarketMonitorError):
    """Data loading related errors."""
    pass


class InvalidFormatError(DataLoadError):
    """Data file has invalid format."""

    def __init__(self, filepath: str, expected_format: str, actual_issue: str):
        self.filepath = filepath
        self.expected_format = expected_format
        self.actual_issue = actual_issue
        super().__init__(
            f"Invalid data format in {filepath}",
            f"Expected: {expected_format}\nIssue: {actual_issue}"
        )


class MissingColumnError(DataLoadError):
    """Required column is missing from data."""

    def __init__(self, column: str, available_columns: Optional[List[str]] = None):
        self.column = column
        self.available_columns = available_columns
        details = None
        if available_columns:
            available = ", ".join(available_columns[:10])
            if len(available_columns) > 10:
                available += f"... ({len(available_columns)} total)"
            details = f"Available columns: {available}"
        super().__init__(
            f"Required column '{column}' not found in data",
            details
        )


# =============================================================================
# Analysis Errors
# =============================================================================

class AnalysisError(MarketMonitorError):
    """Analysis related errors."""
    pass


class InsufficientDataError(AnalysisError):
    """Not enough data for analysis."""

    def __init__(self, required: int, actual: int, analysis_type: str = "analysis"):
        self.required = required
        self.actual = actual
        self.analysis_type = analysis_type
        super().__init__(
            f"Insufficient data for {analysis_type}",
            f"Required: {required} observations, Actual: {actual} observations"
        )


class NetworkConstructionError(AnalysisError):
    """Failed to construct network."""

    def __init__(self, reason: str, node_count: Optional[int] = None):
        self.reason = reason
        self.node_count = node_count
        details = f"Node count: {node_count}" if node_count else None
        super().__init__(
            f"Failed to construct network: {reason}",
            details
        )


class ComputationError(AnalysisError):
    """Computation failed."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"Computation failed during {operation}",
            reason
        )


# =============================================================================
# Utility Functions
# =============================================================================

def format_exception_chain(exc: Exception) -> str:
    """Format exception with cause chain for logging."""
    messages = [str(exc)]
    current = exc.__cause__
    while current:
        messages.append(f"  Caused by: {current}")
        current = current.__cause__
    return "\n".join(messages)
