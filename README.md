# ClauseGuard — Contract Risk & Playbook Auditor

An AI-assisted contract compliance auditing system that parses incoming agreements, retrieves relevant clauses using hybrid vector + keyword search, and evaluates every rule against a company playbook with exact page citations and structured pass/fail verdicts.

## Architecture & Workflow

1. **Document Parsing & Chunking:**
   - Uses LlamaParse to extract clean layout-aware Markdown from uploaded vendor contracts while preserving page numbers for verifiable citations.
2. **Hybrid Retrieval (`HybridContractRetriever`):**
   - Combines ChromaDB dense vector search, BM25 keyword matching, and cross-encoder reranking so both exact legal terms and semantic obligations are retrieved accurately.
3. **Structured Playbook Auditing (`ContractAuditor`):**
   - Evaluates each compliance rule via Gemini 3.5 Flash with strict Pydantic schemas (`ComplianceRule`, `ClauseEvaluation`, `ContractAuditReport`) to output `PASS`, `FAIL`, or `MISSING` verdicts alongside verbatim quotes and page numbers.
4. **Dynamic Playbook Compiler:**
   - Converts plain-English policy notes into structured, machine-checkable audit rules ready for immediate execution.
5. **FastAPI REST Endpoints (`api.py`):**
   - `POST /api/generate-playbook` — Compiles unstructured policy notes into typed `ComplianceRule` JSON.
   - `POST /api/audit` — Audits an uploaded contract PDF against a selected or custom playbook.

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload