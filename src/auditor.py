import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.retriever import HybridContractRetriever
from src.schemas import ClauseEvaluation, ComplianceRule, ContractAuditReport, DocumentChunk

load_dotenv()

# Fast, high-reasoning Gemini model supporting native Pydantic structured outputs
GEMINI_MODEL = "gemini-3.5-flash"

SYSTEM_INSTRUCTION = """You are a senior corporate contracts attorney performing a strict compliance audit.
You will be given ONE Compliance Rule and the top retrieved excerpts from a vendor contract.

Your rules:
1. Evaluate the contract excerpts ONLY against the provided Compliance Rule. Do not assume unstated terms.
2. Assign status:
   - "PASS": The contract explicitly satisfies the rule's requirement.
   - "FAIL": The contract contains a clause that contradicts or falls short of the rule's requirement.
   - "MISSING": None of the provided excerpts address this rule's topic.
3. Evidence & Citations:
   - For "PASS" or "FAIL", copy an exact, word-for-word `verbatim_quote` from the provided excerpts and record its exact `cited_page` and `cited_section`.
   - For "MISSING", set `verbatim_quote`, `cited_page`, and `cited_section` to null.
4. Remediation:
   - For "FAIL" or "MISSING", provide a concrete, ready-to-insert `suggested_redline` clause that brings the contract into compliance.
   - For "PASS", set `suggested_redline` to null.
"""


def _format_chunks_for_prompt(chunks: list[DocumentChunk]) -> str:
    """Formats retrieved DocumentChunks with clear page and section headers for LLM grounding."""
    if not chunks:
        return "NO RELEVANT CLAUSES FOUND IN CONTRACT."

    formatted_blocks = []
    for c in chunks:
        header = f"--- EXCERPT [Page {c.page_number} | Section: {c.section_title} | ID: {c.chunk_id}] ---"
        formatted_blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(formatted_blocks)


class ContractAuditor:
    """
    Orchestrates rule-by-rule retrieval and Gemini structured compliance evaluation.
    """

    def __init__(self, model_name: str = GEMINI_MODEL):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY is missing from environment variables.")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def evaluate_rule(
        self,
        rule: ComplianceRule,
        retrieved_chunks: list[DocumentChunk],
    ) -> ClauseEvaluation:
        """Evaluates a single ComplianceRule against its retrieved contract chunks."""
        context_text = _format_chunks_for_prompt(retrieved_chunks)

        user_prompt = f"""### COMPLIANCE RULE TO AUDIT
- Rule ID: {rule.rule_id}
- Rule Name: {rule.rule_name}
- Category: {rule.category}
- Severity if Violated: {rule.severity}
- Mandatory Requirement: {rule.requirement}

### RETRIEVED CONTRACT EXCERPTS
{context_text}
"""

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=ClauseEvaluation,
            ),
        )

        # The google-genai SDK automatically parses the validated JSON into response.parsed
        if response.parsed is not None:
            return response.parsed

        return ClauseEvaluation.model_validate_json(response.text)

    def audit_contract(
        self,
        contract_filename: str,
        rules: list[ComplianceRule],
        retriever: HybridContractRetriever,
        top_n_chunks: int = 4,
    ) -> ContractAuditReport:
        """Runs the full compliance audit across all rules and compiles the final report."""
        evaluations: list[ClauseEvaluation] = []

        for rule in rules:
            chunks = retriever.retrieve_for_rule(rule, top_n=top_n_chunks)
            evaluation = self.evaluate_rule(rule, chunks)
            evaluations.append(evaluation)

        passed = sum(1 for e in evaluations if e.status == "PASS")
        failed = sum(1 for e in evaluations if e.status == "FAIL")
        missing = sum(1 for e in evaluations if e.status == "MISSING")

        return ContractAuditReport(
            contract_filename=contract_filename,
            total_rules=len(rules),
            passed_count=passed,
            failed_count=failed,
            missing_count=missing,
            evaluations=evaluations,
        )