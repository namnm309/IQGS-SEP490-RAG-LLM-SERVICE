from pathlib import Path


def infer_doc_type(file_path: str, knowledge_base: str) -> str:
    name = Path(file_path).stem.lower()
    parent = Path(file_path).parent.name.lower()

    rules = [
        (("jd_", "job_description", "job-desc"), "job_description"),
        (("policy", "policies"), "policy"),
        (("form_", "template", "checklist"), "form_template"),
        (("rubric", "swe", "interview_rubric"), "interview_rubric"),
        (("question_bank", "questions"), "question_bank"),
        (("role_context", "team"), "role_context"),
    ]
    haystack = f"{parent}/{name}"
    for tokens, doc_type in rules:
        if any(t in haystack for t in tokens):
            return doc_type

    if knowledge_base == "system":
        if Path(file_path).suffix.lower() == ".pdf" or "design" in name or "book" in name:
            return "tech_reference"
        return "general"
    return "general"
