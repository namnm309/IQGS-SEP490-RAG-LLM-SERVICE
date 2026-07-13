"""Parse và validate JD từ file hoặc text — dùng cho S3-01."""
from __future__ import annotations

import tempfile
from pathlib import Path

from config.settings import Settings
from helpers.jd_validator import validate_jd_text
from models.internal_schemas import ParseJdResponse, ValidateJdRequest, ValidateJdResponse
from services.document_parser import DocumentParser
from services.rag_error_helpers import build_rag_error_detail


class JdParseService:
    def __init__(self, parser: DocumentParser, settings: Settings):
        self._parser = parser
        self._settings = settings

    def _validation_config(self) -> dict:
        return {
            "jd_min_chars": self._settings.jd_min_chars,
            "jd_max_chars": self._settings.jd_max_chars,
            "jd_min_words": self._settings.jd_min_words,
            "jd_max_words": self._settings.jd_max_words,
        }

    def _stats_dict(self, stats) -> dict:
        return {
            "charCount": stats.char_count,
            "wordCount": stats.word_count,
            "meaningfulLines": stats.meaningful_lines,
            "signalGroupsMatched": stats.signal_groups_matched,
        }

    def _validation_failure_response(
        self, file_name: str, validation_result
    ) -> ParseJdResponse:
        return ParseJdResponse(
            success=False,
            file_name=file_name,
            error="JD không hợp lệ",
            detail=validation_result.errors[0] if validation_result.errors else "JD không hợp lệ",
            stage="JD_VALIDATION",
            exception_type="JdValidationError",
            errors=list(validation_result.errors),
            stats=self._stats_dict(validation_result.stats),
        )

    def validate_text(self, request: ValidateJdRequest) -> ValidateJdResponse:
        file_name = request.file_name or "JD"
        validation = validate_jd_text(
            request.job_description,
            file_name,
            self._validation_config(),
        )
        if not validation.valid:
            return ValidateJdResponse(
                success=False,
                error="JD không hợp lệ",
                detail=validation.errors[0] if validation.errors else "JD không hợp lệ",
                stage="JD_VALIDATION",
                exception_type="JdValidationError",
                errors=list(validation.errors),
                warnings=list(validation.warnings),
                stats=self._stats_dict(validation.stats),
            )

        return ValidateJdResponse(
            success=True,
            job_description=request.job_description.strip(),
            file_name=file_name,
            warnings=list(validation.warnings),
            stats=self._stats_dict(validation.stats),
        )

    def parse_upload(self, file_bytes: bytes, file_name: str) -> tuple[ParseJdResponse | None, dict | None]:
        """
        Trả (success_response, error_detail).
        error_detail dùng với HTTPException khi cần HTTP 422.
        """
        if not file_name:
            detail = build_rag_error_detail(
                error="File không hợp lệ",
                detail="Thiếu tên file.",
                stage="JD_PARSE",
                exception_type="MissingFileName",
                errors=["Thiếu tên file."],
            )
            return None, detail

        if not self._parser.supported_extension(file_name):
            detail = build_rag_error_detail(
                error="Định dạng file không hỗ trợ",
                detail=f"Chỉ hỗ trợ: .pdf, .docx, .txt. File: {file_name}",
                stage="JD_PARSE",
                exception_type="UnsupportedFileType",
                errors=[f"Định dạng file '{Path(file_name).suffix}' không hỗ trợ."],
            )
            return None, detail

        suffix = Path(file_name).suffix.lower()
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = Path(tmp.name)

            try:
                text = self._parser.parse(tmp_path, file_name)
            finally:
                tmp_path.unlink(missing_ok=True)
        except ValueError as exc:
            detail = build_rag_error_detail(
                error="Không đọc được file JD",
                detail=str(exc),
                stage="JD_PARSE",
                exception_type="EmptyDocument",
                errors=[str(exc)],
            )
            return None, detail
        except Exception as exc:
            detail = build_rag_error_detail(
                error="Không đọc được file JD",
                detail=str(exc),
                stage="JD_PARSE",
                exception_type=type(exc).__name__,
                errors=[str(exc)],
            )
            return None, detail

        validation = validate_jd_text(text, file_name, self._validation_config())
        if not validation.valid:
            failure = self._validation_failure_response(file_name, validation)
            detail = build_rag_error_detail(
                error=failure.error or "JD không hợp lệ",
                detail=failure.detail,
                stage=failure.stage,
                exception_type=failure.exception_type,
                errors=failure.errors,
            )
            return failure, detail

        return ParseJdResponse(
            success=True,
            job_description=text.strip(),
            file_name=file_name,
            warnings=list(validation.warnings),
            stats=self._stats_dict(validation.stats),
        ), None
