"""Exception utilities for the production AI/LLM pipeline."""

import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class PipelineException(Exception):
    """Base exception for the AI/LLM pipeline.

    This exception preserves the original exception, extracts traceback details,
    and formats a compact logger-friendly message.
    """

    def __init__(
        self,
        message: str,
        original_exception: Optional[BaseException] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.original_exception = original_exception
        self.context = context or {}
        self.file_name, self.line_number = self._extract_location()
        self.full_traceback = self._build_full_traceback()
        self.logger_message = self._build_logger_message()

    def _extract_location(self) -> Tuple[Optional[str], Optional[int]]:
        if self.original_exception is not None and self.original_exception.__traceback__ is not None:
            extracted = traceback.extract_tb(self.original_exception.__traceback__)
            if extracted:
                last_frame = extracted[-1]
                return last_frame.filename, last_frame.lineno

        extracted = traceback.extract_stack()[:-1]
        if extracted:
            last_frame = extracted[-1]
            return last_frame.filename, last_frame.lineno

        return None, None

    def _build_full_traceback(self) -> str:
        if self.original_exception is not None:
            return "".join(
                traceback.format_exception(
                    type(self.original_exception),
                    self.original_exception,
                    self.original_exception.__traceback__,
                )
            )

        current_exc = traceback.TracebackException(type(self), self, self.__traceback__)
        return "".join(current_exc.format())

    def _build_logger_message(self) -> str:
        location = "unknown"
        if self.file_name and self.line_number is not None:
            location = f"{Path(self.file_name).name}:{self.line_number}"

        parts = [f"{location}", self.message]

        if self.original_exception is not None:
            original_name = type(self.original_exception).__name__
            parts.append(f"caused_by={original_name}")
            parts.append(f"original={self.original_exception!r}")

        if self.context:
            context_pairs = [f"{key}={value!r}" for key, value in self.context.items()]
            parts.append(f"context={{ {', '.join(context_pairs)} }}")

        return " | ".join(parts)

    def __str__(self) -> str:
        return self.logger_message

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(message={self.message!r}, "
            f"original_exception={self.original_exception!r}, "
            f"context={self.context!r}, "
            f"file_name={self.file_name!r}, "
            f"line_number={self.line_number!r})"
        )

    def full_trace(self) -> str:
        """Return the full traceback string for logging or debugging."""
        return self.full_traceback


