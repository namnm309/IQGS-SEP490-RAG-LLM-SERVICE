"""Pydantic schemas cho internal RAG API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ScopeType = Literal["SYSTEM", "HR"]
DocumentStatusType = Literal["COMPLETED", "FAILED", "PROCESSING", "QUEUED"]
DifficultyType = Literal["easy", "medium", "hard"]
ExperienceLevelType = Literal["intern", "junior", "mid", "senior", "lead"]
QuestionType = Literal[
    "technical",
    "behavioral",
    "situational",
    "system-design",
    "problem-solving",
]

VALID_QUESTION_TYPES = frozenset(
    {
        "technical",
        "behavioral",
        "situational",
        "system-design",
        "problem-solving",
    }
)


VALID_EXPERIENCE_LEVELS = frozenset({"intern", "junior", "mid", "senior", "lead"})

_EXPERIENCE_LEVEL_ALIASES = {
    "fresher": "intern",
    "thuc-tap": "intern",
    "internship": "intern",
    "entry": "junior",
    "entry-level": "junior",
    "junior-level": "junior",
    "mid-level": "mid",
    "middle": "mid",
    "staff": "senior",
    "senior-level": "senior",
    "principal": "lead",
    "lead-level": "lead",
}


def normalize_experience_level(value: str) -> str:
    """Chuẩn hóa cấp độ kinh nghiệm: Intern, Junior, Senior, ..."""
    key = value.strip().lower().replace(" ", "-").replace("_", "-")
    key = _EXPERIENCE_LEVEL_ALIASES.get(key, key)
    if key not in VALID_EXPERIENCE_LEVELS:
        raise ValueError(
            f"experienceLevel không hợp lệ: '{value}'. "
            f"Cho phép: {', '.join(sorted(VALID_EXPERIENCE_LEVELS))}"
        )
    return key


def normalize_question_type(value: str) -> str:
    """Chuẩn hóa alias question type: Technical, system_design, Problem-Solving, ..."""
    key = value.strip().lower().replace(" ", "-").replace("_", "-")
    if key not in VALID_QUESTION_TYPES:
        raise ValueError(
            f"questionTypes không hợp lệ: '{value}'. "
            f"Cho phép: {', '.join(sorted(VALID_QUESTION_TYPES))}"
        )
    return key


def normalize_question_types_list(value: object) -> object:
    if not isinstance(value, list):
        return value
    return [normalize_question_type(str(item)) for item in value]


class IngestRequest(BaseModel):
    document_id: str = Field(..., alias="documentId")
    blob_read_url: str = Field(..., alias="blobReadUrl")
    scope: ScopeType
    owner_id: str | None = Field(default=None, alias="ownerId")
    file_name: str = Field(..., alias="fileName")
    source_title: str | None = Field(default=None, alias="sourceTitle")
    source_url: str | None = Field(default=None, alias="sourceUrl")
    section: str | None = None
    year: int | None = None

    model_config = ConfigDict(populate_by_name=True)


class IngestResponse(BaseModel):
    document_id: str = Field(..., alias="documentId")
    status: Literal["COMPLETED", "FAILED"]
    chunk_count: int | None = Field(default=None, alias="chunkCount")
    error_message: str | None = Field(default=None, alias="errorMessage")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class JobGenerationBaseRequest(BaseModel):
    """Shared fields cho generate-questions và generate-plan."""

    owner_id: str = Field(..., alias="ownerId")
    job_description: str = Field(..., alias="jobDescription")
    number_of_questions: int = Field(..., alias="numberOfQuestions", ge=1, le=50)
    difficulty: DifficultyType = "medium"
    question_types: list[QuestionType] = Field(
        default_factory=lambda: ["technical", "behavioral"],
        alias="questionTypes",
    )
    skills: list[str] = Field(default_factory=list)
    hr_note: str | None = Field(default=None, alias="hrNote", max_length=2000)
    language: str | None = Field(
        default=None,
        alias="language",
        description="Vietnamese | English — ngôn ngữ free-text đầu ra",
    )
    # SCRUM-388: KnowledgeDocumentIds Studio Selected → filter HR chunks
    document_ids: list[str] = Field(
        default_factory=list,
        alias="documentIds",
        description="Optional HR knowledge document IDs to restrict retrieval",
    )

    model_config = ConfigDict(populate_by_name=True)

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, value: object) -> str:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("question_types", mode="before")
    @classmethod
    def normalize_question_types(cls, value: object) -> object:
        return normalize_question_types_list(value)


class GenerateQuestionsRequest(JobGenerationBaseRequest):
    pass


class GeneratePlanRequest(JobGenerationBaseRequest):
    pass


class GeneratePlanAsyncRequest(GeneratePlanRequest):
    job_id: str = Field(..., alias="jobId")

    model_config = ConfigDict(populate_by_name=True)


class GenerateQuestionsAsyncRequest(GenerateQuestionsRequest):
    job_id: str = Field(..., alias="jobId")

    model_config = ConfigDict(populate_by_name=True)


class QuestionCitationItem(BaseModel):
    knowledge_base: str = Field(..., alias="knowledgeBase")
    source_file: str = Field(..., alias="sourceFile")
    chunk_index: int = Field(..., alias="chunkIndex")
    excerpt: str = ""

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


PlanCitationItem = QuestionCitationItem


class GeneratedQuestionItem(BaseModel):
    question: str
    question_type: str = Field(..., alias="questionType")
    difficulty: str
    rationale: str = ""
    sample_answer: str = Field(default="", alias="sampleAnswer")
    citations: list[QuestionCitationItem] = Field(default_factory=list)
    order: int | None = None
    skill: str | None = None
    focus_area: str | None = Field(default=None, alias="focusArea")
    evaluation_criteria: list[str] = Field(
        default_factory=list, alias="evaluationCriteria"
    )
    code_template_type: str | None = Field(default=None, alias="codeTemplateType")
    code_snippet: str | None = Field(default=None, alias="codeSnippet")
    # Gợi ý text cho HR nên tìm/đính kèm hình nào — không phải AI gen ảnh
    image_hint: str | None = Field(default=None, alias="imageHint")
    # SCRUM-400: Candidate UI — Text (textarea) | Code (ô nhập code)
    answer_method: Literal["Text", "Code"] | None = Field(
        default=None, alias="answerMethod"
    )

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class GenerateQuestionsResponse(BaseModel):
    success: bool
    questions: list[GeneratedQuestionItem] = Field(default_factory=list)
    raw_answer: str | None = Field(default=None, alias="rawAnswer")
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class QuestionTypeDistributionItem(BaseModel):
    type: str
    count: int
    reason: str = ""

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class DifficultyDistributionItem(BaseModel):
    difficulty: str
    count: int

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class SkillCoverageItem(BaseModel):
    skill: str
    question_count: int = Field(..., alias="questionCount")
    focus_areas: list[str] = Field(default_factory=list, alias="focusAreas")
    # SCRUM-369: tên file RAG (source_file) gắn với focus — Studio UI hiển thị
    source_files: list[str] = Field(default_factory=list, alias="sourceFiles")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class RecommendedQuestionOutlineItem(BaseModel):
    order: int
    type: str
    difficulty: str
    skill: str = ""
    focus_area: str = Field(default="", alias="focusArea")
    goal: str = ""

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class QuestionGenerationPlan(BaseModel):
    role_title: str = Field(..., alias="roleTitle")
    summary: str = ""
    difficulty: str = "medium"
    level: str | None = Field(default=None, alias="level")
    experience_level: ExperienceLevelType = Field(..., alias="experienceLevel")
    total_questions: int = Field(..., alias="totalQuestions")
    skills: list[str] = Field(default_factory=list)
    question_type_distribution: list[QuestionTypeDistributionItem] = Field(
        default_factory=list, alias="questionTypeDistribution"
    )
    difficulty_distribution: list[DifficultyDistributionItem] = Field(
        default_factory=list, alias="difficultyDistribution"
    )
    coverage: list[SkillCoverageItem] = Field(default_factory=list)
    recommended_question_outline: list[RecommendedQuestionOutlineItem] = Field(
        default_factory=list, alias="recommendedQuestionOutline"
    )
    notes: str = ""
    citations: list[PlanCitationItem] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    @field_validator("experience_level", mode="before")
    @classmethod
    def normalize_experience_level_field(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            raise ValueError("experienceLevel là bắt buộc")
        return normalize_experience_level(str(value))


class GeneratePlanResponse(BaseModel):
    success: bool
    plan: QuestionGenerationPlan | None = None
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None
    detail: str | None = None
    stage: str | None = None
    exception_type: str | None = Field(default=None, alias="exceptionType")
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class GenerateQuestionsFromPlanRequest(BaseModel):
    owner_id: str = Field(..., alias="ownerId")
    job_description: str = Field(..., alias="jobDescription")
    approved_plan: QuestionGenerationPlan = Field(..., alias="approvedPlan")
    hr_note: str | None = Field(default=None, alias="hrNote", max_length=2000)
    language: str | None = Field(
        default=None,
        alias="language",
        description="Vietnamese | English — ngôn ngữ câu hỏi sinh ra",
    )

    model_config = ConfigDict(populate_by_name=True)


class CandidateGeneratePlanRequest(GeneratePlanRequest):
    """Luồng Candidate: plan luyện tập, không phải plan HR duyệt."""

    audience: str = Field(default="jd_practice", alias="audience")
    cv_context: str | None = Field(default=None, alias="cvContext")
    candidate_note: str | None = Field(default=None, alias="candidateNote", max_length=2000)


class CandidateGenerateQuestionsFromPlanRequest(GenerateQuestionsFromPlanRequest):
    audience: str = Field(default="jd_practice", alias="audience")
    cv_context: str | None = Field(default=None, alias="cvContext")
    candidate_note: str | None = Field(default=None, alias="candidateNote", max_length=2000)


class GenerateQuestionsFromPlanAsyncRequest(GenerateQuestionsFromPlanRequest):
    job_id: str = Field(..., alias="jobId")

    model_config = ConfigDict(populate_by_name=True)


class GenerateQuestionsFromPlanResponse(BaseModel):
    success: bool
    questions: list[GeneratedQuestionItem] = Field(default_factory=list)
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class AsyncAcceptedResponse(BaseModel):
    accepted: bool = True
    job_id: str | None = Field(default=None, alias="jobId")
    document_id: str | None = Field(default=None, alias="documentId")
    phase: Literal["PLAN", "QUESTIONS", "INGEST"]

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class DeleteDocumentResponse(BaseModel):
    success: bool = True
    document_id: str = Field(..., alias="documentId")
    deleted_count: int = Field(..., alias="deletedCount")
    message: str = ""

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["up", "down"]
    config: Literal["valid", "invalid"]
    # SCRUM-378 — thông tin runtime đang active (không trả API key)
    chat_model: str | None = Field(default=None, alias="chatModel")
    ollama_base_url: str | None = Field(default=None, alias="ollamaBaseUrl")
    temperature: float | None = None
    top_k_system: int | None = Field(default=None, alias="topKSystem")
    top_k_hr: int | None = Field(default=None, alias="topKHr")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class RagModelItem(BaseModel):
    name: str
    size: int | None = None
    digest: str | None = None
    is_cloud: bool = Field(False, alias="isCloud")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class RagModelsListResponse(BaseModel):
    models: list[RagModelItem] = Field(default_factory=list)
    error_message: str | None = Field(default=None, alias="errorMessage")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class ReloadConfigResponse(BaseModel):
    success: bool = True
    applied: bool = False
    connection_changed: bool = Field(False, alias="connectionChanged")
    chat_model: str | None = Field(default=None, alias="chatModel")
    ollama_base_url: str | None = Field(default=None, alias="ollamaBaseUrl")
    temperature: float | None = None
    top_k_system: int | None = Field(default=None, alias="topKSystem")
    top_k_hr: int | None = Field(default=None, alias="topKHr")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class ValidateJdRequest(BaseModel):
    job_description: str = Field(..., alias="jobDescription")
    file_name: str | None = Field(default=None, alias="fileName")

    model_config = ConfigDict(populate_by_name=True)


class ValidateJdResponse(BaseModel):
    success: bool
    job_description: str | None = Field(default=None, alias="jobDescription")
    file_name: str | None = Field(default=None, alias="fileName")
    warnings: list[str] = Field(default_factory=list)
    stats: dict | None = None
    error: str | None = None
    detail: str | None = None
    stage: str | None = None
    exception_type: str | None = Field(default=None, alias="exceptionType")
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class ParseJdResponse(BaseModel):
    success: bool
    job_description: str | None = Field(default=None, alias="jobDescription")
    file_name: str | None = Field(default=None, alias="fileName")
    warnings: list[str] = Field(default_factory=list)
    stats: dict | None = None
    error: str | None = None
    detail: str | None = None
    stage: str | None = None
    exception_type: str | None = Field(default=None, alias="exceptionType")
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class ParseCvResponse(BaseModel):
    """Kết quả parse CV (SCRUM-300) — khớp ParseCvResult phía Backend .NET."""

    success: bool
    skills: list[str] = Field(default_factory=list)
    summary: str | None = None
    file_name: str | None = Field(default=None, alias="fileName")
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    detail: str | None = None
    stage: str | None = None
    exception_type: str | None = Field(default=None, alias="exceptionType")
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class QuestionAssistChatMessage(BaseModel):
    role: str
    content: str


class QuestionAssistContext(BaseModel):
    job_description: str = Field(..., alias="jobDescription")
    hr_note: str | None = Field(default=None, alias="hrNote")
    plan: dict | list | None = None
    difficulty: str = "medium"
    skills: list[str] = Field(default_factory=list)
    current_question: dict = Field(..., alias="currentQuestion")
    is_manually_edited: bool = Field(default=False, alias="isManuallyEdited")

    model_config = ConfigDict(populate_by_name=True)


class QuestionAssistRequest(BaseModel):
    owner_id: str = Field(..., alias="ownerId")
    hr_message: str = Field(..., alias="hrMessage")
    context: QuestionAssistContext
    chat_history: list[QuestionAssistChatMessage] = Field(default_factory=list, alias="chatHistory")

    model_config = ConfigDict(populate_by_name=True)


class QuestionAssistSuggestion(BaseModel):
    question: str | None = None
    rationale: str | None = None
    sample_answer: str | None = Field(default=None, alias="sampleAnswer")
    difficulty: str | None = None
    question_type: str | None = Field(default=None, alias="questionType")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class QuestionAssistResponse(BaseModel):
    success: bool = True
    reply: str = ""
    suggestion: QuestionAssistSuggestion | None = None
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None
    detail: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


# ── Evaluate Answer (SCRUM-281) — chấm câu trả lời Candidate ─────────────────


class EvaluateAnswerRequest(BaseModel):
    """Input BE gửi khi Candidate submit answer trong practice session."""

    question: str
    evaluation_criteria: list[str] = Field(default_factory=list, alias="evaluationCriteria")
    candidate_answer: str = Field(..., alias="candidateAnswer")
    sample_answer: str | None = Field(default=None, alias="sampleAnswer")
    jd_context: str | None = Field(default=None, alias="jdContext")
    skill: str | None = None
    question_type: str | None = Field(default=None, alias="questionType")

    model_config = ConfigDict(populate_by_name=True)


class EvaluateAnswerResponse(BaseModel):
    """Kết quả chấm điểm — không bao giờ trả sampleAnswer ra ngoài."""

    success: bool = True
    score: float | None = None
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    suggestion: str | None = None
    dimension_scores: dict[str, float] | None = Field(default=None, alias="dimensionScores")
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None
    detail: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


# ── Practice Session Insight (SCRUM-305) — nhận xét tổng quan + skills ───────


class QuestionInsightSummary(BaseModel):
    """Tóm tắt 1 câu đã chấm — BE gửi sang RAG để sinh insight phiên."""

    question_type: str | None = Field(default=None, alias="questionType")
    skill: str | None = None
    score: float | None = None
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    dimension_scores: dict[str, float] | None = Field(default=None, alias="dimensionScores")

    model_config = ConfigDict(populate_by_name=True)


class PracticeSessionInsightRequest(BaseModel):
    """Input BE khi complete phiên practice — sinh AI Insight song ngữ."""

    overall_score: float | None = Field(default=None, alias="overallScore")
    total_questions: int = Field(..., alias="totalQuestions")
    answered_count: int = Field(..., alias="answeredCount")
    set_title: str | None = Field(default=None, alias="setTitle")
    set_skills: list[str] = Field(default_factory=list, alias="setSkills")
    question_summaries: list[QuestionInsightSummary] = Field(
        default_factory=list, alias="questionSummaries"
    )

    model_config = ConfigDict(populate_by_name=True)


class PracticeSessionInsightResponse(BaseModel):
    """Nhận xét tổng quan + kỹ năng cần cải thiện (Vi/En)."""

    success: bool = True
    insight_vi: str | None = Field(default=None, alias="insightVi")
    insight_en: str | None = Field(default=None, alias="insightEn")
    skills_to_improve_vi: list[str] = Field(default_factory=list, alias="skillsToImproveVi")
    skills_to_improve_en: list[str] = Field(default_factory=list, alias="skillsToImproveEn")
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None
    detail: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


# ── JD Fit Review — qualitative only (no numeric scores) ─────────────────────

JD_FIT_VERDICTS = frozenset({"unfit", "fair", "good", "excellent"})
JD_FIT_FLAGS = frozenset({"onJd", "weak", "offJd", "duplicate"})
JD_FIT_ACTION_TYPES = frozenset({"add", "rewrite", "remove"})


class EvaluateQuestionSetItem(BaseModel):
    question_id: str | None = Field(default=None, alias="questionId")
    order: int | None = None
    question: str
    question_type: str | None = Field(default=None, alias="questionType")
    difficulty: str | None = None
    skill: str | None = None
    focus_area: str | None = Field(default=None, alias="focusArea")
    rationale: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class EvaluateQuestionSetRequest(BaseModel):
    owner_id: str | None = Field(default=None, alias="ownerId")
    job_description: str = Field(..., alias="jobDescription")
    hr_note: str | None = Field(default=None, alias="hrNote")
    set_title: str | None = Field(default=None, alias="setTitle")
    plan: dict | list | None = None
    questions: list[EvaluateQuestionSetItem]

    model_config = ConfigDict(populate_by_name=True)


class JdFitSourceItem(BaseModel):
    chunk_index: int = Field(..., alias="chunkIndex")
    excerpt: str = ""

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class JdFitQuestionFlagItem(BaseModel):
    question_id: str | None = Field(default=None, alias="questionId")
    order: int | None = None
    flag: str
    note_vi: str | None = Field(default=None, alias="noteVi")
    note_en: str | None = Field(default=None, alias="noteEn")
    sources: list[JdFitSourceItem] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class JdFitSuggestedActionItem(BaseModel):
    type: str
    question_id: str | None = Field(default=None, alias="questionId")
    reason_vi: str | None = Field(default=None, alias="reasonVi")
    reason_en: str | None = Field(default=None, alias="reasonEn")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class EvaluateQuestionSetResponse(BaseModel):
    success: bool = True
    verdict: str | None = None
    summary_vi: str | None = Field(default=None, alias="summaryVi")
    summary_en: str | None = Field(default=None, alias="summaryEn")
    question_flags: list[JdFitQuestionFlagItem] = Field(default_factory=list, alias="questionFlags")
    missing_topics: list[str] = Field(default_factory=list, alias="missingTopics")
    suggested_actions: list[JdFitSuggestedActionItem] = Field(
        default_factory=list, alias="suggestedActions"
    )
    jd_sources: list[JdFitSourceItem] = Field(default_factory=list, alias="jdSources")
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None
    detail: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)
