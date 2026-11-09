class LLMServiceError(Exception):
    """Base error for failures talking to the LLM provider."""


class LLMConfigError(LLMServiceError):
    """The LLM provider settings are missing or invalid."""


class LLMTimeoutError(LLMServiceError):
    """The LLM provider did not respond in time."""


class LLMUnavailableError(LLMServiceError):
    """The LLM provider could not be reached or returned an error status."""


class LLMInvalidResponseError(LLMServiceError):
    """The LLM provider responded, but not with a usable analysis."""
