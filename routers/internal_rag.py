"""Internal RAG endpoints — chỉ Backend gọi qua X-Internal-Api-Key."""
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from api.deps import (
    get_async_generation_service,
    get_async_ingest_service,
    get_cv_parse_service,
    get_evaluate_answer_service,
    get_evaluate_question_set_service,
    get_ingest_service,
    get_jd_analyze_service,
    get_jd_parse_service,
    get_recommend_configuration_service,
    get_plan_service,
    get_plan_refine_service,
    get_candidate_plan_service,
    get_candidate_question_service,
    get_practice_session_insight_service,
    get_question_assist_service,
    get_question_service,
    get_retrieval_service,
    get_settings_ref,
    reload_runtime_config,
)
from models.internal_schemas import (
    AnalyzeJdRequest,
    AnalyzeJdResponse,
    AsyncAcceptedResponse,
    BindOutlineSourcesRequest,
    BindOutlineSourcesResponse,
    DeleteDocumentResponse,
    EvaluateAnswerRequest,
    EvaluateAnswerResponse,
    EvaluateQuestionSetRequest,
    EvaluateQuestionSetResponse,
    GeneratePlanAsyncRequest,
    GeneratePlanRequest,
    GeneratePlanResponse,
    GenerateQuestionsFromPlanAsyncRequest,
    GenerateQuestionsFromPlanRequest,
    GenerateQuestionsFromPlanResponse,
    CandidateGeneratePlanRequest,
    CandidateGenerateQuestionsFromPlanRequest,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    IngestRequest,
    IngestResponse,
    ParseCvResponse,
    ParseJdResponse,
    PracticeSessionInsightRequest,
    PracticeSessionInsightResponse,
    QuestionAssistRequest,
    QuestionAssistResponse,
    RagModelItem,
    RagModelsListResponse,
    RefinePlanRequest,
    RefinePlanResponse,
    RecommendInterviewConfigurationRequest,
    RecommendInterviewConfigurationResponse,
    ReloadConfigResponse,
    RetrieveRequest,
    RetrieveResponse,
    RetrievedChunkDto,
    ValidateJdRequest,
    ValidateJdResponse,
)
from security.internal_api_key import verify_internal_api_key
from services.async_generation_service import AsyncGenerationService
from services.async_ingest_service import AsyncIngestService
from services.cv_parse_service import CvParseService
from services.evaluate_answer_service import EvaluateAnswerService
from services.evaluate_question_set_service import EvaluateQuestionSetService
from services.jd_analyze_service import JdAnalyzeService
from services.jd_parse_service import JdParseService
from services.plan_generation_service import PlanGenerationService
from services.plan_refine_service import PlanRefineService
from services.candidate_plan_generation_service import CandidatePlanGenerationService
from services.candidate_question_generation_service import CandidateQuestionGenerationService
from services.practice_session_insight_service import PracticeSessionInsightService
from services.question_assist_service import QuestionAssistService
from services.question_generation_service import QuestionGenerationService
from services.rag_ingest_service import RagIngestService
from services.recommend_interview_configuration_service import RecommendInterviewConfigurationService
from services.rag_error_helpers import build_rag_error_detail

router = APIRouter(
    prefix="/internal/rag",
    tags=["Internal RAG"],
    dependencies=[Depends(verify_internal_api_key)],
)


@router.post("/ingest", response_model=IngestResponse)
def ingest_document(
    request: IngestRequest,
    service: RagIngestService = Depends(get_ingest_service),
) -> IngestResponse:
    return service.ingest(request)


@router.post(
    "/ingest/async",
    response_model=AsyncAcceptedResponse,
    status_code=202,
    response_model_exclude_none=True,
)
def ingest_document_async(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    async_service: AsyncIngestService = Depends(get_async_ingest_service),
) -> AsyncAcceptedResponse:
    background_tasks.add_task(async_service.run_ingest, request)
    return AsyncAcceptedResponse(
        accepted=True,
        document_id=request.document_id,
        phase="INGEST",
    )


def _chunk_to_dto(c) -> RetrievedChunkDto:
    meta = c.metadata or {}
    return RetrievedChunkDto(
        document_id=c.document_id,
        chunk_index=c.chunk_index,
        content=c.content[:800] + ("…" if len(c.content) > 800 else ""),
        scope=c.scope,
        score=float(c.score or 0),
        file_name=meta.get("fileName") or meta.get("file_name"),
        section=meta.get("section"),
    )


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_chunks(
    request: RetrieveRequest,
    retrieval=Depends(get_retrieval_service),
) -> RetrieveResponse:
    """SCRUM-443/444: retrieve SYSTEM + HR (HR chỉ khi documentIds non-empty)."""
    import time

    start = time.time()
    if not request.job_description.strip():
        return RetrieveResponse(
            success=False,
            error="jobDescription là bắt buộc",
            processing_time_ms=(time.time() - start) * 1000,
        )
    if not request.owner_id.strip():
        return RetrieveResponse(
            success=False,
            error="ownerId là bắt buộc",
            processing_time_ms=(time.time() - start) * 1000,
        )

    try:
        system_chunks, hr_chunks = retrieval.retrieve_for_job(
            request.job_description,
            request.owner_id.strip(),
            top_k_system=request.top_k_system,
            top_k_hr=request.top_k_hr,
            document_ids=list(request.document_ids or []),
            query_extra=request.query_extra,
        )
    except Exception as exc:
        return RetrieveResponse(
            success=False,
            error=str(exc),
            processing_time_ms=(time.time() - start) * 1000,
        )

    return RetrieveResponse(
        success=True,
        system_chunks=[_chunk_to_dto(c) for c in system_chunks],
        hr_chunks=[_chunk_to_dto(c) for c in hr_chunks],
        processing_time_ms=(time.time() - start) * 1000,
    )


@router.post("/validate-jd", response_model=ValidateJdResponse)
def validate_jd(
    request: ValidateJdRequest,
    service: JdParseService = Depends(get_jd_parse_service),
) -> ValidateJdResponse:
    result = service.validate_text(request)
    if not result.success:
        raise HTTPException(
            status_code=422,
            detail=build_rag_error_detail(
                error=result.error or "JD không hợp lệ",
                detail=result.detail,
                stage=result.stage or "JD_VALIDATION",
                exception_type=result.exception_type or "JdValidationError",
                errors=result.errors,
            ),
        )
    return result


@router.post("/parse-jd", response_model=ParseJdResponse)
async def parse_jd(
    file: UploadFile = File(...),
    service: JdParseService = Depends(get_jd_parse_service),
) -> ParseJdResponse:
    content = await file.read()
    result, error_detail = service.parse_upload(content, file.filename or "")
    if error_detail:
        raise HTTPException(status_code=422, detail=error_detail)
    return result


@router.post("/analyze-jd", response_model=AnalyzeJdResponse, response_model_exclude_none=True)
def analyze_jd(
    request: AnalyzeJdRequest,
    service: JdAnalyzeService = Depends(get_jd_analyze_service),
) -> AnalyzeJdResponse:
    """SCRUM-416/432: classify IT job posting + extract metadata — null nếu không chắc."""
    result = service.analyze(request)
    if not result.success:
        # SCRUM-432: classify reject → 422; LLM/infra fail → 502
        status = 422 if (result.stage or "") == "JD_CLASSIFY" else 502
        raise HTTPException(
            status_code=status,
            detail=build_rag_error_detail(
                error=result.error or "Phân tích JD thất bại",
                detail=result.detail,
                stage=result.stage or "JD_ANALYZE",
                exception_type=result.exception_type or "JdAnalyzeError",
                errors=result.errors,
            ),
        )
    return result


@router.post(
    "/recommend-interview-configuration",
    response_model=RecommendInterviewConfigurationResponse,
    response_model_exclude_none=True,
)
def recommend_interview_configuration(
    request: RecommendInterviewConfigurationRequest,
    service: RecommendInterviewConfigurationService = Depends(get_recommend_configuration_service),
) -> RecommendInterviewConfigurationResponse:
    """Studio Phase 1: AI đề xuất focus areas + phân bổ câu hỏi từ JD + RAG (không ghi đè HR settings)."""
    result = service.recommend(request)
    if not result.success:
        raise HTTPException(
            status_code=502,
            detail=build_rag_error_detail(
                error=result.error or "Đề xuất cấu hình phỏng vấn thất bại",
                detail=result.detail,
                stage=result.stage or "RECOMMEND_CONFIGURATION",
                exception_type=result.exception_type or "RecommendConfigurationError",
                errors=result.errors,
            ),
        )
    return result


@router.post("/parse-cv", response_model=ParseCvResponse, response_model_exclude_none=True)
async def parse_cv(
    file: UploadFile = File(...),
    service: CvParseService = Depends(get_cv_parse_service),
) -> ParseCvResponse:
    """Trích kỹ năng + tóm tắt từ CV (PDF/DOCX/JPG/JPEG/PNG) bằng AI (SCRUM-300)."""
    content = await file.read()
    result, error_detail, status_code = service.parse_upload(content, file.filename or "")
    if error_detail:
        raise HTTPException(status_code=status_code, detail=error_detail)
    return result


@router.post(
    "/generate-plan",
    response_model=GeneratePlanResponse,
    response_model_exclude_none=True,
)
def generate_plan(
    request: GeneratePlanRequest,
    service: PlanGenerationService = Depends(get_plan_service),
) -> GeneratePlanResponse:
    return service.generate(request)


@router.post(
    "/refine-plan",
    response_model=RefinePlanResponse,
    response_model_exclude_none=True,
)
def refine_plan(
    request: RefinePlanRequest,
    service: PlanRefineService = Depends(get_plan_refine_service),
) -> RefinePlanResponse:
    """SCRUM-420: merge refine — trả PlanPatch delta, không full plan."""
    return service.refine(request)


@router.post(
    "/bind-outline-sources",
    response_model=BindOutlineSourcesResponse,
    response_model_exclude_none=True,
)
def bind_outline_sources(
    request: BindOutlineSourcesRequest,
    retrieval=Depends(get_retrieval_service),
) -> BindOutlineSourcesResponse:
    """SCRUM-426: khóa/rebind JD (+ Admin) citations trên outline slots."""
    import time

    from services.outline_source_binder import bind_sources_to_outline

    start = time.time()
    if not request.job_description.strip():
        return BindOutlineSourcesResponse(
            success=False,
            error="jobDescription là bắt buộc",
            processing_time_ms=(time.time() - start) * 1000,
        )
    if not request.outline:
        return BindOutlineSourcesResponse(
            success=True,
            outline=[],
            processing_time_ms=(time.time() - start) * 1000,
        )

    chunks = []
    try:
        query_extra = " | ".join(
            part
            for it in request.outline
            for part in [it.skill, it.focus_area]
            if part
        )[:1500]
        system_chunks, hr_chunks = retrieval.retrieve_for_job(
            request.job_description,
            request.owner_id.strip(),
            query_extra=query_extra or None,
            document_ids=request.document_ids or None,
            top_k_system=8,
        )
        chunks = list(system_chunks) + list(hr_chunks)
    except Exception:
        chunks = []

    locked = bind_sources_to_outline(
        list(request.outline),
        job_description=request.job_description,
        chunks=chunks,
        force_rebind=request.force_rebind,
    )
    return BindOutlineSourcesResponse(
        success=True,
        outline=locked,
        processing_time_ms=(time.time() - start) * 1000,
    )


@router.post(
    "/generate-plan/async",
    response_model=AsyncAcceptedResponse,
    status_code=202,
    response_model_exclude_none=True,
)
def generate_plan_async(
    request: GeneratePlanAsyncRequest,
    background_tasks: BackgroundTasks,
    async_service: AsyncGenerationService = Depends(get_async_generation_service),
) -> AsyncAcceptedResponse:
    plan_request = GeneratePlanRequest.model_validate(
        request.model_dump(by_alias=True, exclude={"jobId", "job_id"})
    )
    background_tasks.add_task(async_service.run_plan, request.job_id, plan_request)
    return AsyncAcceptedResponse(accepted=True, job_id=request.job_id, phase="PLAN")


@router.post(
    "/generate-questions-from-plan",
    response_model=GenerateQuestionsFromPlanResponse,
    response_model_exclude_none=True,
)
def generate_questions_from_plan(
    request: GenerateQuestionsFromPlanRequest,
    service: QuestionGenerationService = Depends(get_question_service),
) -> GenerateQuestionsFromPlanResponse:
    return service.generate_from_plan(request)


@router.post(
    "/candidate/generate-plan",
    response_model=GeneratePlanResponse,
    response_model_exclude_none=True,
)
def candidate_generate_plan(
    request: CandidateGeneratePlanRequest,
    service: CandidatePlanGenerationService = Depends(get_candidate_plan_service),
) -> GeneratePlanResponse:
    return service.generate(request)


@router.post(
    "/candidate/generate-questions-from-plan",
    response_model=GenerateQuestionsFromPlanResponse,
    response_model_exclude_none=True,
)
def candidate_generate_questions_from_plan(
    request: CandidateGenerateQuestionsFromPlanRequest,
    service: CandidateQuestionGenerationService = Depends(get_candidate_question_service),
) -> GenerateQuestionsFromPlanResponse:
    return service.generate_from_plan(request)


@router.post(
    "/generate-questions-from-plan/async",
    response_model=AsyncAcceptedResponse,
    status_code=202,
    response_model_exclude_none=True,
)
def generate_questions_from_plan_async(
    request: GenerateQuestionsFromPlanAsyncRequest,
    background_tasks: BackgroundTasks,
    async_service: AsyncGenerationService = Depends(get_async_generation_service),
) -> AsyncAcceptedResponse:
    questions_request = GenerateQuestionsFromPlanRequest.model_validate(
        request.model_dump(by_alias=True, exclude={"jobId", "job_id"})
    )
    background_tasks.add_task(
        async_service.run_questions_from_plan, request.job_id, questions_request
    )
    return AsyncAcceptedResponse(accepted=True, job_id=request.job_id, phase="QUESTIONS")


@router.post(
    "/generate-questions",
    response_model=GenerateQuestionsResponse,
    response_model_exclude_none=True,
)
def generate_questions(
    request: GenerateQuestionsRequest,
    service: QuestionGenerationService = Depends(get_question_service),
) -> GenerateQuestionsResponse:
    return service.generate(request)


@router.post(
    "/question-assist",
    response_model=QuestionAssistResponse,
    response_model_exclude_none=True,
)
def question_assist(
    request: QuestionAssistRequest,
    service: QuestionAssistService = Depends(get_question_assist_service),
) -> QuestionAssistResponse:
    return service.assist(request)


@router.post(
    "/evaluate-answer",
    response_model=EvaluateAnswerResponse,
    response_model_exclude_none=True,
)
def evaluate_answer(
    request: EvaluateAnswerRequest,
    service: EvaluateAnswerService = Depends(get_evaluate_answer_service),
) -> EvaluateAnswerResponse:
    """Chấm điểm câu trả lời Candidate theo rubric (SCRUM-281)."""
    return service.evaluate(request)


@router.post(
    "/evaluate-question-set",
    response_model=EvaluateQuestionSetResponse,
    response_model_exclude_none=True,
)
def evaluate_question_set(
    request: EvaluateQuestionSetRequest,
    service: EvaluateQuestionSetService = Depends(get_evaluate_question_set_service),
) -> EvaluateQuestionSetResponse:
    """Đánh giá bộ câu hỏi so với JD — chỉ nhận xét lời, không điểm số."""
    return service.evaluate(request)


@router.post(
    "/practice-session-insight",
    response_model=PracticeSessionInsightResponse,
    response_model_exclude_none=True,
)
def practice_session_insight(
    request: PracticeSessionInsightRequest,
    service: PracticeSessionInsightService = Depends(get_practice_session_insight_service),
) -> PracticeSessionInsightResponse:
    """Sinh AI Insight tổng quan + skillsToImprove song ngữ (SCRUM-305)."""
    return service.generate(request)


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document_chunks(
    document_id: str,
    service: RagIngestService = Depends(get_ingest_service),
) -> DeleteDocumentResponse:
    try:
        deleted_count = service.delete_document(document_id)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=build_rag_error_detail(
                error="Xóa chunks thất bại",
                detail=str(exc),
                stage="CHUNK_DELETE",
                exception_type=type(exc).__name__,
            ),
        ) from exc

    if deleted_count > 0:
        message = f"Đã xóa {deleted_count} chunk(s) thành công."
    else:
        message = "Không có chunk nào để xóa (document_id không tồn tại hoặc đã xóa trước đó)."

    return DeleteDocumentResponse(
        success=True,
        document_id=document_id,
        deleted_count=deleted_count,
        message=message,
    )


@router.get("/models", response_model=RagModelsListResponse)
def list_ollama_models() -> RagModelsListResponse:
    """Proxy GET Ollama /api/tags — danh sách model local (+ cloud đã pull). SCRUM-378."""
    import httpx

    settings = get_settings_ref()
    base = settings.ollama_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    tags_url = f"{base}/api/tags"

    try:
        response = httpx.get(tags_url, timeout=min(30, settings.request_timeout_seconds))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return RagModelsListResponse(
            models=[],
            error_message=f"Không kết nối được Ollama tại {tags_url}: {exc}",
        )

    models: list[RagModelItem] = []
    for item in payload.get("models") or []:
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        models.append(
            RagModelItem(
                name=name,
                size=item.get("size"),
                digest=item.get("digest"),
                is_cloud="cloud" in name.lower(),
            )
        )
    return RagModelsListResponse(models=models)


@router.post("/reload-config", response_model=ReloadConfigResponse)
def reload_config() -> ReloadConfigResponse:
    """Force reload runtime settings từ DB — gọi sau Admin PUT. SCRUM-378."""
    try:
        result = reload_runtime_config()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=build_rag_error_detail(
                error="Reload config thất bại",
                detail=str(exc),
                stage="RELOAD_CONFIG",
                exception_type=type(exc).__name__,
            ),
        ) from exc

    return ReloadConfigResponse(
        success=True,
        applied=bool(result.get("applied")),
        connection_changed=bool(result.get("connectionChanged")),
        chat_model=result.get("chatModel"),
        ollama_base_url=result.get("ollamaBaseUrl"),
        temperature=result.get("temperature"),
        top_k_system=result.get("topKSystem"),
        top_k_hr=result.get("topKHr"),
    )
