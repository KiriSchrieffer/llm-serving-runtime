class RuntimeErrorBase(Exception):
    """Base exception for runtime-specific failures."""


class BackendUnavailableError(RuntimeErrorBase):
    """Raised when a backend cannot serve a request."""


class SchedulerError(RuntimeErrorBase):
    """Raised when request scheduling fails."""

