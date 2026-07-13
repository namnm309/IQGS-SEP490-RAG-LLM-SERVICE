"""Lỗi có stage — BE/FE đọc được bước pipeline ingest bị fail."""


class IngestStageError(Exception):
    def __init__(self, stage: str, message: str, exception_type: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.exception_type = exception_type or "IngestStageError"

    @classmethod
    def wrap(cls, stage: str, exc: BaseException) -> "IngestStageError":
        return cls(stage, str(exc), type(exc).__name__)
