"""Orchestration ingest: download → parse → chunk → embed → pgvector + callback."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException

from exceptions.ingest_errors import IngestStageError
from models.internal_schemas import IngestRequest, IngestResponse
from services.backend_callback_client import BackendCallbackClient
from services.chunking_service import ChunkingService
from services.document_classify import DocumentClassifyService
from services.document_downloader import DocumentDownloader
from services.document_parser import DocumentParser
from services.embedding_service import EmbeddingService
from services.jsonl_qa_chunker import JsonlQaChunker, QaChunkDraft
from vectorstores.base import ChunkRecord, VectorStore

logger = logging.getLogger(__name__)

VALID_SCOPES = frozenset({"SYSTEM", "HR"})


class RagIngestService:
    def __init__(
        self,
        vector_store: VectorStore,
        downloader: DocumentDownloader,
        parser: DocumentParser,
        chunking: ChunkingService,
        embedding: EmbeddingService,
        callback_client: BackendCallbackClient,
        jsonl_chunker: JsonlQaChunker | None = None,
        classifier: DocumentClassifyService | None = None,
    ):
        self._store = vector_store
        self._downloader = downloader
        self._parser = parser
        self._chunking = chunking
        self._embedding = embedding
        self._callback = callback_client
        self._jsonl_chunker = jsonl_chunker or JsonlQaChunker()
        self._classifier = classifier

    def execute_ingest(self, request: IngestRequest) -> None:
        """Pipeline + callback — dùng cho background task, không raise HTTP."""
        scope = request.scope.upper()
        try:
            self._validate_scope_owner(scope, request.owner_id)
        except HTTPException as exc:
            self._notify_failed(request.document_id, exc.detail)
            return

        self._callback.update_document_status(request.document_id, "PROCESSING")

        try:
            chunk_count = self._run_pipeline(request, scope)
            self._callback.update_document_status(
                request.document_id,
                "COMPLETED",
                chunk_count=chunk_count,
            )
        except IngestStageError as exc:
            logger.warning(
                "Ingest stage %s failed for %s: %s",
                exc.stage,
                request.document_id,
                exc.message,
            )
            self._notify_failed(request.document_id, f"[{exc.stage}] {exc.message}")
        except ValueError as exc:
            wrapped = IngestStageError("VALIDATE", str(exc), "ValueError")
            self._notify_failed(request.document_id, wrapped.message)
        except Exception as exc:
            logger.exception("Ingest pipeline failed %s", request.document_id)
            wrapped = IngestStageError.wrap("PIPELINE", exc)
            self._notify_failed(request.document_id, wrapped.message)

    def ingest(self, request: IngestRequest) -> IngestResponse:
        scope = request.scope.upper()
        self._validate_scope_owner(scope, request.owner_id)

        self._callback.update_document_status(request.document_id, "PROCESSING")

        try:
            chunk_count = self._run_pipeline(request, scope)
            self._callback.update_document_status(
                request.document_id,
                "COMPLETED",
                chunk_count=chunk_count,
            )
            return IngestResponse(
                document_id=request.document_id,
                status="COMPLETED",
                chunk_count=chunk_count,
            )
        except HTTPException as exc:
            self._notify_failed(request.document_id, exc.detail)
            raise
        except IngestStageError as exc:
            logger.warning(
                "Ingest stage %s failed for %s: %s",
                exc.stage,
                request.document_id,
                exc.message,
            )
            self._notify_failed(request.document_id, f"[{exc.stage}] {exc.message}")
            status = 400 if exc.stage == "VALIDATE" else 500
            raise self._to_http_exception(exc, status_code=status) from exc
        except ValueError as exc:
            wrapped = IngestStageError("VALIDATE", str(exc), "ValueError")
            self._notify_failed(request.document_id, wrapped.message)
            raise self._to_http_exception(wrapped, status_code=400) from exc
        except Exception as exc:
            logger.exception("Ingest pipeline failed %s", request.document_id)
            wrapped = IngestStageError.wrap("PIPELINE", exc)
            self._notify_failed(request.document_id, wrapped.message)
            raise self._to_http_exception(wrapped, status_code=500) from exc

    def _run_pipeline(self, request: IngestRequest, scope: str) -> int:
        try:
            with self._downloader.download_to_tempfile(
                request.blob_read_url, request.file_name
            ) as temp_path:
                if not self._parser.supported_extension(request.file_name):
                    raise IngestStageError(
                        "VALIDATE",
                        f"Định dạng file không hỗ trợ: {request.file_name}",
                        "UnsupportedFileType",
                    )

                owner_id = request.owner_id if scope == "HR" else None
                # SCRUM-442/455: section + metadata competency/roadmap vào chunk
                metadata_base = {
                    "fileName": request.file_name,
                    "sourceTitle": request.source_title,
                    "section": request.section,
                    "sourceUrl": request.source_url,
                    "year": request.year,
                    "documentType": request.document_type or request.section,
                    "sourceVersion": request.source_version,
                    "roleKey": request.role_key,
                    "technology": request.technology,
                    "level": request.level,
                    "skill": request.skill,
                    "topic": request.topic,
                }
                metadata_base = {k: v for k, v in metadata_base.items() if v is not None}

                is_jsonl = Path(request.file_name).suffix.lower() == ".jsonl"
                if is_jsonl:
                    # SCRUM-448: 1 record Q/A = 1 chunk (không split theo ký tự)
                    try:
                        drafts = self._jsonl_chunker.build_chunks(
                            temp_path, request.file_name
                        )
                    except ValueError as exc:
                        raise IngestStageError("PARSE", str(exc), "ValueError") from exc
                    except Exception as exc:
                        raise IngestStageError.wrap("PARSE", exc) from exc

                    if not drafts:
                        raise IngestStageError(
                            "CHUNK",
                            "Không tạo được chunk từ nội dung tài liệu",
                            "EmptyChunks",
                        )

                    # SCRUM-466 L2: classify sample trước embed
                    sample = "\n".join(d.content for d in drafts[:20])
                    self._ensure_it_kb(sample)

                    return self._embed_and_upsert(
                        request=request,
                        scope=scope,
                        owner_id=owner_id,
                        metadata_base=metadata_base,
                        drafts=drafts,
                    )

                try:
                    raw_text = self._parser.parse(temp_path, request.file_name)
                except Exception as exc:
                    raise IngestStageError.wrap("PARSE", exc) from exc

                # SCRUM-466 L2: classify excerpt trước chunk/embed
                self._ensure_it_kb(raw_text)

                try:
                    text_chunks = self._chunking.split(raw_text)
                except Exception as exc:
                    raise IngestStageError.wrap("CHUNK", exc) from exc

                if not text_chunks:
                    raise IngestStageError(
                        "CHUNK",
                        "Không tạo được chunk từ nội dung tài liệu",
                        "EmptyChunks",
                    )

                drafts = [
                    QaChunkDraft(content=text, record_index=i, repo=None)
                    for i, text in enumerate(text_chunks)
                ]
                return self._embed_and_upsert(
                    request=request,
                    scope=scope,
                    owner_id=owner_id,
                    metadata_base=metadata_base,
                    drafts=drafts,
                )
        except IngestStageError:
            raise
        except Exception as exc:
            raise IngestStageError.wrap("BLOB_DOWNLOAD", exc) from exc

    def _ensure_it_kb(self, text: str) -> None:
        """SCRUM-466 L2: classify KB — reject non-IT trước khi embed."""
        if self._classifier is None:
            logger.warning("KB classify skipped — classifier not configured")
            return
        result = self._classifier.classify(text, "kb", run_l1=True, max_chars=8000)
        if result.success:
            return
        # L1 fail hoặc classify reject → CLASSIFY stage (status Failed, không embed)
        if result.exception_type == "LlmJsonError" or (
            result.exception_type
            and result.exception_type not in {"NotItDocument", "NotItDomainL1", "EmptyDocument"}
        ):
            # Infra/LLM fail — vẫn fail-closed theo plan
            raise IngestStageError(
                "CLASSIFY",
                result.detail or result.error or "Không phân loại được tài liệu",
                result.exception_type or "ClassifyFailed",
            )
        raise IngestStageError(
            "CLASSIFY",
            result.detail or result.error or "Tài liệu không thuộc IT",
            result.exception_type or "NotItDocument",
        )

    def _embed_and_upsert(
        self,
        request: IngestRequest,
        scope: str,
        owner_id: str | None,
        metadata_base: dict,
        drafts: list[QaChunkDraft],
    ) -> int:
        batch_size = self._embedding.batch_size
        record_batches: list[list[ChunkRecord]] = []

        for batch_start in range(0, len(drafts), batch_size):
            batch_drafts = drafts[batch_start : batch_start + batch_size]
            batch_texts = [d.content for d in batch_drafts]
            try:
                embeddings = self._embedding.embed_texts(batch_texts)
            except Exception as exc:
                raise IngestStageError.wrap("EMBEDDING", exc) from exc

            batch_records = []
            for draft, embedding in zip(batch_drafts, embeddings):
                meta = {
                    **metadata_base,
                    "chunkIndex": draft.record_index,
                    **(draft.extra_metadata or {}),
                }
                if draft.record_kind:
                    meta["recordKind"] = draft.record_kind
                elif draft.repo is not None:
                    # SCRUM-448: metadata Q/A dataset
                    meta["recordKind"] = "qa"
                    meta["repo"] = draft.repo
                elif Path(request.file_name).suffix.lower() == ".jsonl":
                    meta["recordKind"] = "qa"
                if draft.repo is not None:
                    meta["repo"] = draft.repo

                batch_records.append(
                    ChunkRecord(
                        chunk_index=draft.record_index,
                        content=draft.content,
                        embedding=embedding,
                        scope=scope,
                        owner_id=owner_id,
                        metadata=meta,
                    )
                )
            record_batches.append(batch_records)

        try:
            return self._store.upsert_chunks_batched(request.document_id, record_batches)
        except Exception as exc:
            raise IngestStageError.wrap("VECTOR_STORE", exc) from exc

    def _notify_failed(self, document_id: str, detail: object) -> None:
        message = detail if isinstance(detail, str) else str(detail)
        try:
            self._callback.update_document_status(
                document_id,
                "FAILED",
                error_message=message[:2000],
            )
        except Exception:
            logger.warning("Callback FAILED thất bại cho document %s", document_id)

    @staticmethod
    def _to_http_exception(exc: IngestStageError, status_code: int) -> HTTPException:
        return HTTPException(
            status_code=status_code,
            detail={
                "error": f"[{exc.stage}] {exc.message}",
                "detail": exc.message,
                "stage": exc.stage,
                "exceptionType": exc.exception_type,
            },
        )

    @staticmethod
    def _validate_scope_owner(scope: str, owner_id: str | None) -> None:
        if scope not in VALID_SCOPES:
            raise HTTPException(status_code=400, detail=f"Scope không hợp lệ: {scope}")
        if scope == "SYSTEM" and owner_id:
            raise HTTPException(
                status_code=400,
                detail="SYSTEM scope không được có ownerId",
            )
        if scope == "HR" and not owner_id:
            raise HTTPException(
                status_code=400,
                detail="HR scope bắt buộc ownerId",
            )

    def delete_document(self, document_id: str) -> int:
        return self._store.delete_document_chunks(document_id)
