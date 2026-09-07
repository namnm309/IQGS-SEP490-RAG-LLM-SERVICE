"""Pydantic schemas cho internal RAG API."""
from __future__ import annotations

from typing import Any, Literal

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


class QuestionDistributionItem(BaseModel):
    category: str
    percentage: int
    question_count: int = Field(..., alias="questionCount")

    model_config = ConfigDict(populate_by_name=True)


class FocusAreaItem(BaseModel):
    name: str
    weight: float
    order_index: int = Field(..., alias="orderIndex")
    description: str | None = None
    source_reason: str | None = Field(default=None, alias="sourceReason")

    model_config = ConfigDict(populate_by_name=True)


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
    # SCRUM-417: HR đã confirm cấp độ — plan.experience_level phải khớp (optional).
    experience_level: ExperienceLevelType | None = Field(
        default=None,
        alias="experienceLevel",
        description="intern|junior|mid|senior|lead — ưu tiên hơn suy luận từ JD khi có",
    )
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
    question_distribution: list[QuestionDistributionItem] | None = Field(
        default=None,
        alias="questionDistribution",
    )
    focus_areas: list[FocusAreaItem] | None = Field(default=None, alias="focusAreas")
    question_styles: list[str] | None = Field(default=None, alias="questionStyles")
    coding_task_types: list[str] | None = Field(default=None, alias="codingTaskTypes")

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

    @field_validator("experience_level", mode="before")
    @classmethod
    def normalize_optional_experience_level(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not str(value).strip()):
            return None
        return normalize_experience_level(str(value))


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


PlanOriginType = Literal["HR", "SYSTEM", "LLM"]


class QuestionCitationItem(BaseModel):
    knowledge_base: str = Field(..., alias="knowledgeBase")
    source_file: str = Field(..., alias="sourceFile")
    chunk_index: int = Field(..., alias="chunkIndex")
    excerpt: str = ""
    # SCRUM-421: waterfall provenance trên từng citation
    origin: PlanOriginType | None = None
    used_for: list[str] = Field(default_factory=list, alias="usedFor")
    reason: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


PlanCitationItem = QuestionCitationItem


class RubricCriterionItem(BaseModel):
    """SCRUM-418: Một tiêu chí chấm có trọng số và mốc."""

    id: str = ""
    label: str = ""
    weight: int = 0
    anchors: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


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
    evaluation_criteria: list[RubricCriterionItem | str] = Field(
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

    # SCRUM-421: provenance tóm tắt + cảnh báo thiếu Admin KB
    source_provenance: ProvenanceBlock | None = Field(
        default=None, alias="sourceProvenance"
    )
    missing_admin_warning: bool | None = Field(
        default=None, alias="missingAdminWarning"
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
    # SCRUM-420: provenance waterfall HR → SYSTEM → LLM
    provenance: "ProvenanceBlock | None" = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class ProvenanceItem(BaseModel):
    origin: PlanOriginType
    source_file: str | None = Field(default=None, alias="sourceFile")
    chunk_index: int | None = Field(default=None, alias="chunkIndex")
    excerpt: str | None = None
    used_for: list[str] = Field(default_factory=list, alias="usedFor")
    reason: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class ProvenanceBlock(BaseModel):
    primary_origin: PlanOriginType = Field(..., alias="primaryOrigin")
    items: list[ProvenanceItem] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class PlanPatch(BaseModel):
    """Delta refine — field null/omit = giữ baseline."""

    replace_coverage: list[SkillCoverageItem] | None = Field(
        default=None, alias="replaceCoverage"
    )
    replace_skills: list[str] | None = Field(default=None, alias="replaceSkills")
    replace_outline: list["RecommendedQuestionOutlineItem"] | None = Field(
        default=None, alias="replaceOutline"
    )
    replace_question_type_distribution: list[QuestionTypeDistributionItem] | None = (
        Field(default=None, alias="replaceQuestionTypeDistribution")
    )
    replace_difficulty_distribution: list[DifficultyDistributionItem] | None = Field(
        default=None, alias="replaceDifficultyDistribution"
    )
    update_summary: str | None = Field(default=None, alias="updateSummary")
    replace_citations: list[PlanCitationItem] | None = Field(
        default=None, alias="replaceCitations"
    )
    instruction_applied: str | None = Field(default=None, alias="instructionApplied")

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)

    def has_changes(self) -> bool:
        return any(
            [
                self.replace_coverage,
                self.replace_skills,
                self.replace_outline,
                self.replace_question_type_distribution,
                self.replace_difficulty_distribution,
                self.update_summary,
                self.replace_citations,
                self.instruction_applied,
            ]
        )


class RefinePlanRequest(JobGenerationBaseRequest):
    baseline_plan: dict[str, Any] = Field(..., alias="baselinePlan")

    model_config = ConfigDict(populate_by_name=True)


class RefinePlanResponse(BaseModel):
    success: bool
    patch: PlanPatch | None = None
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None
    detail: str | None = None
    stage: str | None = None
    exception_type: str | None = Field(default=None, alias="exceptionType")
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class RecommendedQuestionOutlineItem(BaseModel):
    order: int
    type: str
    difficulty: str
    skill: str = ""
    focus_area: str = Field(default="", alias="focusArea")
    goal: str = ""
    # Text = lý thuyết | Code = coding — HR chỉnh trên live preview
    answer_method: str | None = Field(default=None, alias="answerMethod")
    # SCRUM-426: nguồn đã khóa (JD why-asked + Admin technical-body) — Gen copy
    citations: list[PlanCitationItem] = Field(default_factory=list)

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


class BindOutlineSourcesRequest(BaseModel):
    """SCRUM-426: khóa/rebind citations trên outline (Apply Live Preview)."""

    owner_id: str = Field(..., alias="ownerId")
    job_description: str = Field(..., alias="jobDescription")
    outline: list[RecommendedQuestionOutlineItem] = Field(default_factory=list)
    document_ids: list[str] | None = Field(default=None, alias="documentIds")
    force_rebind: bool = Field(default=False, alias="forceRebind")

    model_config = ConfigDict(populate_by_name=True)


class BindOutlineSourcesResponse(BaseModel):
    success: bool
    outline: list[RecommendedQuestionOutlineItem] = Field(default_factory=list)
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class RetrieveRequest(BaseModel):
    """SCRUM-443/444: retrieve mỏng cho suggestions + preview."""

    owner_id: str = Field(..., alias="ownerId")
    job_description: str = Field(..., alias="jobDescription")
    document_ids: list[str] = Field(default_factory=list, alias="documentIds")
    query_extra: str | None = Field(default=None, alias="queryExtra")
    top_k_system: int | None = Field(default=None, alias="topKSystem")
    top_k_hr: int | None = Field(default=None, alias="topKHr")

    model_config = ConfigDict(populate_by_name=True)


class RetrievedChunkDto(BaseModel):
    document_id: str = Field(..., alias="documentId")
    chunk_index: int = Field(..., alias="chunkIndex")
    content: str
    scope: str
    score: float
    file_name: str | None = Field(default=None, alias="fileName")
    section: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class RetrieveResponse(BaseModel):
    success: bool
    system_chunks: list[RetrievedChunkDto] = Field(default_factory=list, alias="systemChunks")
    hr_chunks: list[RetrievedChunkDto] = Field(default_factory=list, alias="hrChunks")
    processing_time_ms: float | None = Field(default=None, alias="processingTimeMs")
    error: str | None = None

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class GenerateQuestionsFromPlanRequest(BaseModel):
    owner_id: str = Field(..., alias="ownerId")
    job_description: str = Field(..., alias="jobDescription")
    approved_plan: QuestionGenerationPlan = Field(..., alias="approvedPlan")
    # SCRUM-429: AVOID_QUESTIONS có thể dài — tăng trần hrNote
    hr_note: str | None = Field(default=None, alias="hrNote", max_length=8000)
    language: str | None = Field(
        default=None,
        alias="language",
        description="Vietnamese | English — ngôn ngữ câu hỏi sinh ra",
    )
    # SCRUM-388/421: filter HR chunks Selected (mirror plan generation)
    document_ids: list[str] = Field(
        default_factory=list,
        alias="documentIds",
        description="Optional HR knowledge document IDs to restrict retrieval",
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


class AnalyzeJdRequest(BaseModel):
    """SCRUM-416: phân tích metadata JD bằng LLM — không bịa giá trị mặc định."""

    job_description: str = Field(..., alias="jobDescription")
    file_name: str | None = Field(default=None, alias="fileName")

    model_config = ConfigDict(populate_by_name=True)


class AnalyzeJdResponse(BaseModel):
    success: bool
    position: str | None = None
    job_title: str | None = Field(default=None, alias="jobTitle")
    detected_role: str | None = Field(default=None, alias="detectedRole")
    detected_seniority: str | None = Field(default=None, alias="detectedSeniority")
    experience_level: str | None = Field(default=None, alias="experienceLevel")
    detected_language: str | None = Field(default=None, alias="detectedLanguage")
    skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    summary: str | None = None
    # SCRUM-432: classify trước lưu — chỉ pass khi job_description + isItRole
    document_type: str | None = Field(default=None, alias="documentType")
    is_it_role: bool | None = Field(default=None, alias="isItRole")
    reject_reason: str | None = Field(default=None, alias="rejectReason")
    error: str | None = None
    detail: str | None = None
    stage: str | None = None
    exception_type: str | None = Field(default=None, alias="exceptionType")
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, ser_json_by_alias=True)


class JobProfileInput(BaseModel):
    job_title: str | None = Field(default=None, alias="jobTitle")
    experience_level: str | None = Field(default=None, alias="experienceLevel")
    detected_role: str | None = Field(default=None, alias="detectedRole")
    detected_language: str | None = Field(default=None, alias="detectedLanguage")
    skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    summary: str | None = None

    model_config = ConfigDict(populate_by_name=True)


class RecommendInterviewConfigurationRequest(BaseModel):
    owner_id: str = Field(..., alias="ownerId")
    job_description: str = Field(..., alias="jobDescription")
    job_profile: JobProfileInput = Field(..., alias="jobProfile")
    document_ids: list[str] = Field(default_factory=list, alias="documentIds")
    number_of_questions: int | None = Field(default=None, alias="numberOfQuestions", ge=1, le=50)

    model_config = ConfigDict(populate_by_name=True)


class RecommendInterviewConfigurationResponse(BaseModel):
    success: bool
    recommended_configuration: dict | None = Field(
        default=None, alias="recommendedConfiguration"
    )
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
