"""Typed error hierarchy.

Rule: no bare `except Exception: pass` on core paths. Every failure that can
reach a client must map to one of these classes so the API can emit a stable
machine-readable `code`.
"""

from __future__ import annotations

from typing import Any


class LVCError(Exception):
    """Base class for every runtime error. Carries a stable wire code."""

    code = "internal_error"
    http_status = 500
    retryable = False

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        self.context = context

    def to_wire(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "retryable": self.retryable,
                "context": self.context,
            }
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} code={self.code} message={self.message!r}>"


class ConfigurationError(LVCError):
    code = "configuration_error"
    http_status = 400


class ProviderUnavailable(LVCError):
    code = "provider_unavailable"
    http_status = 503
    retryable = True


class ProviderLoadError(LVCError):
    code = "provider_load_error"
    http_status = 503
    retryable = True


class ModelMissing(LVCError):
    code = "model_missing"
    http_status = 404


class OutOfMemory(LVCError):
    code = "out_of_memory"
    http_status = 507


class DeviceUnavailable(LVCError):
    code = "device_unavailable"
    http_status = 503
    retryable = True


class AudioDeviceError(LVCError):
    code = "audio_device_error"
    http_status = 503
    retryable = True


class APITimeout(LVCError):
    code = "api_timeout"
    http_status = 504
    retryable = True


class AuthenticationError(LVCError):
    code = "authentication_error"
    http_status = 401


class CancelledTurn(LVCError):
    """Raised when a turn is interrupted (barge-in or explicit cancel)."""

    code = "cancelled_turn"
    http_status = 409


class ValidationFailed(LVCError):
    code = "validation_failed"
    http_status = 422


class NotFound(LVCError):
    code = "not_found"
    http_status = 404


ERROR_CODES: tuple[str, ...] = tuple(
    sorted({cls.code for cls in LVCError.__subclasses__()}) + ["internal_error"]
)
