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

    model_config = ConfigDict(populate_by_name=True)


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
