"""Internal RAG endpoints — chỉ Backend gọi qua X-Internal-Api-Key."""
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

from api.deps import (
    get_async_generation_service,
    get_async_ingest_service,
    get_evaluate_answer_service,
    get_ingest_service,
    get_jd_parse_service,
    get_plan_service,
    get_question_assist_service,
    get_question_service,
)
from models.internal_schemas import (
    AsyncAcceptedResponse,
    DeleteDocumentResponse,
    EvaluateAnswerRequest,
    EvaluateAnswerResponse,
    GeneratePlanAsyncRequest,
    GeneratePlanRequest,
    GeneratePlanResponse,
    GenerateQuestionsFromPlanAsyncRequest,
    GenerateQuestionsFromPlanRequest,
    GenerateQuestionsFromPlanResponse,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    IngestRequest,
    IngestResponse,
    ParseJdResponse,
    QuestionAssistRequest,
    QuestionAssistResponse,
    ValidateJdRequest,
    ValidateJdResponse,
)
from security.internal_api_key import verify_internal_api_key
from services.async_generation_service import AsyncGenerationService
from services.async_ingest_service import AsyncIngestService
from services.evaluate_answer_service import EvaluateAnswerService
from services.jd_parse_service import JdParseService
from services.plan_generation_service import PlanGenerationService
from services.question_assist_service import QuestionAssistService
from services.question_generation_service import QuestionGenerationService
from services.rag_ingest_service import RagIngestService
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
