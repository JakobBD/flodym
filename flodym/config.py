from pydantic import BaseModel, confloat
from typing import Optional
from enum import Enum
import logging



class ErrorBehavior(str, Enum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"
    IGNORE = "ignore"

def handle_error(behavior: ErrorBehavior, message: str=None, error: Exception = None):

    # ensures valid ErrorBehavior
    behavior = ErrorBehavior(behavior)

    if message is None == error is None:
        raise ValueError("Exactly one of message or error must be provided.")

    if behavior == ErrorBehavior.ERROR:
        raise error or ValueError(message)
    elif behavior == ErrorBehavior.WARN:
        logging.warning(message or str(error))
    elif behavior == ErrorBehavior.INFO:
        logging.info(message or str(error))
    elif behavior == ErrorBehavior.IGNORE:
        pass


class Config(BaseModel, validate_assignment=True):
    """
    Configuration class for Flodym.
    """

    check_mass_balance_processes: bool = True
    """Whether to check mass balance after each `process.compute()` call."""
    check_mass_balance_stocks: bool = True
    """Whether to check mass balance after each `stock.compute()` call."""

    error_behavior_mass_balance: ErrorBehavior = ErrorBehavior.ERROR
    """How to handle an error if mass balance is not satisfied."""
    error_behavior_process_underdetermined: ErrorBehavior = ErrorBehavior.ERROR
    """How to handle an error if a process is underdetermined.
    Does not apply to recursive mode and `MFASystem.compute_all_possible()`.
    """
    error_behavior_check_flows: ErrorBehavior = ErrorBehavior.WARN
    """How to handle an error if a flow has negative or NaN value."""

    tolerance: Optional[confloat(ge=0)] = None
    """Absolute tolerance used for mass balance and other checks.
    If None, a default tolerance based on float precision is applied.
    """

config = Config()
