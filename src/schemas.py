from typing import Literal
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents a single parsed section/chunk of a contract with traceability metadata."""
    chunk_id: str = Field(..., description="Unique identifier for the chunk (e.g., 'page_4_chunk_1')")
    text: str = Field(..., description="Clean Markdown text of the contract section")
    page_number: int = Field(..., description="1-indexed page number where this text appears")
    section_title: str = Field(default="General", description="Nearest Markdown heading above this chunk")


class ComplianceRule(BaseModel):
    """Represents a single company policy rule that a vendor contract must satisfy."""
    rule_id: str = Field(..., description="Short unique code, e.g., 'FIN-01'")
    category: str = Field(..., description="Domain category, e.g., 'Payment Terms', 'Liability', 'Termination'")
    rule_name: str = Field(..., description="Human-readable title of the rule")
    requirement: str = Field(..., description="Exact condition required for the contract to PASS")
    search_keywords: list[str] = Field(
        default_factory=list,
        description="Specific legal terms to boost BM25 keyword retrieval (e.g., ['Net 30', 'invoice'])"
    )


class ClauseEvaluation(BaseModel):
    """Strict structured output returned by the LLM for a single compliance rule."""
    status: Literal["PASS", "FAIL", "MISSING"] = Field(
        ...,
        description="PASS if contract satisfies the rule; FAIL if it violates the rule; MISSING if the topic is absent."
    )
    verbatim_quote: str = Field(
        ...,
        description="Exact word-for-word quote from the contract supporting the status, or 'N/A' if MISSING."
    )
    page_number: int | None = Field(
        default=None,
        description="Exact page number where the verbatim_quote appears, or None if MISSING."
    )
    section_title: str = Field(
        default="N/A",
        description="Section heading where the quote was found, or 'N/A' if MISSING."
    )
    risk_reasoning: str = Field(
        ...,
        description="Concise 1-2 sentence explanation of why this clause passes, fails, or creates risk."
    )


class AuditReportItem(BaseModel):
    """Combines the input rule with the LLM's evaluation for final display/export."""
    rule: ComplianceRule
    evaluation: ClauseEvaluation

class ContractAuditReport(BaseModel):
    """Aggregated compliance audit report across all rules for a single contract."""
    contract_filename: str
    total_rules: int
    passed_count: int
    failed_count: int
    missing_count: int
    evaluations: list[ClauseEvaluation]