import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from src.auditor import ContractAuditor
from src.ingestion import ingest_contract
from src.retriever import HybridContractRetriever
from src.schemas import ComplianceRule

load_dotenv()

st.set_page_config(
    page_title="Automated Contract Compliance Auditor",
    page_icon="⚖️",
    layout="wide",
)

DEFAULT_RULES_PATH = Path("data/sample/sample_rules.json")


def load_default_rules() -> list[dict]:
    if DEFAULT_RULES_PATH.exists():
        with open(DEFAULT_RULES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# --- SIDEBAR: API Status & Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    llama_key = os.getenv("LLAMA_CLOUD_API_KEY") or st.text_input(
        "LlamaCloud API Key", type="password"
    )
    gemini_key = os.getenv("GEMINI_API_KEY") or st.text_input(
        "Gemini API Key", type="password"
    )

    if llama_key:
        os.environ["LLAMA_CLOUD_API_KEY"] = llama_key
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    st.divider()
    top_k_chunks = st.slider(
        "Retrieved Chunks per Rule (Top-K)",
        min_value=2,
        max_value=8,
        value=4,
        help="Number of reranked contract clauses passed to Gemini for each rule.",
    )
    st.caption(
        "Pipeline: LlamaParse (Agentic) → ChromaDB + BM25 (RRF) → "
        "MS-MARCO Cross-Encoder → Gemini Structured Output"
    )

# --- HEADER ---
st.title("⚖️ Automated Contract Compliance Auditor")
st.markdown(
    "Upload a vendor contract PDF and audit it against your corporate compliance playbook "
    "with **page-level citations** and **hybrid RAG verification**."
)

# --- MAIN WORKSPACE: 2-Column Input Setup ---
col_upload, col_rules = st.columns([1, 1.4], gap="large")

with col_upload:
    st.subheader("1. Upload Contract PDF")
    uploaded_pdf = st.file_uploader(
        "Select a contract (.pdf)",
        type=["pdf"],
        help="Parsed with layout-aware OCR to preserve tables and section headers.",
    )

with col_rules:
    st.subheader("2. Compliance Playbook Rules")
    if "rules_df" not in st.session_state:
        raw_rules = load_default_rules()
        for r in raw_rules:
            r["search_keywords"] = ", ".join(r.get("search_keywords", []))
        st.session_state["rules_df"] = pd.DataFrame(raw_rules)

    edited_df = st.data_editor(
        st.session_state["rules_df"],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "rule_id": st.column_config.TextColumn("ID", width="small"),
            "rule_name": st.column_config.TextColumn("Rule Name", width="medium"),
            "category": st.column_config.TextColumn("Category", width="small"),
            "requirement": st.column_config.TextColumn("Mandatory Requirement", width="large"),
            "search_keywords": st.column_config.TextColumn("BM25 Keywords (comma-sep)", width="medium"),
        },
    )

# --- RUN AUDIT BUTTON ---
run_clicked = st.button("🚀 Run Compliance Audit", type="primary", use_container_width=True)

if run_clicked:
    if not uploaded_pdf:
        st.error("Please upload a contract PDF before running the audit.")
    elif not os.getenv("LLAMA_CLOUD_API_KEY") or not os.getenv("GEMINI_API_KEY"):
        st.error("Please provide both LLAMA_CLOUD_API_KEY and GEMINI_API_KEY in the sidebar or .env file.")
    else:
        # Parse edited playbook table back into validated ComplianceRule objects
        active_rules: list[ComplianceRule] = []
        for _, row in edited_df.dropna(subset=["rule_id", "requirement"]).iterrows():
            kw_raw = str(row.get("search_keywords", ""))
            keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
            active_rules.append(
                ComplianceRule(
                    rule_id=str(row["rule_id"]).strip(),
                    rule_name=str(row.get("rule_name", "")).strip(),
                    category=str(row.get("category", "General")).strip(),
                    requirement=str(row["requirement"]).strip(),
                    search_keywords=keywords,
                )
            )

        if not active_rules:
            st.error("Add at least one valid compliance rule to run the audit.")
        else:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_pdf.getvalue())
                    tmp_path = tmp_file.name

                with st.status("Running End-to-End Compliance Pipeline...", expanded=True) as status:
                    st.write("📄 **Stage 1/3:** Parsing PDF layout & indexing into ChromaDB...")
                    chunks = ingest_contract(tmp_path)
                    st.write(f"✅ Indexed **{len(chunks)}** semantic contract sections.")

                    st.write("🔍 **Stage 2/3:** Initializing Hybrid BM25 + Vector Retriever & Cross-Encoder...")
                    retriever = HybridContractRetriever(chunks)

                    st.write(f"🧠 **Stage 3/3:** Auditing **{len(active_rules)}** rules with Gemini...")
                    auditor = ContractAuditor()
                    report = auditor.audit_contract(
                        contract_filename=uploaded_pdf.name,
                        rules=active_rules,
                        retriever=retriever,
                        top_n_chunks=top_k_chunks,
                    )
                    status.update(label="Audit Complete!", state="complete", expanded=False)

                st.session_state["audit_report"] = report
                st.session_state["audited_rules"] = active_rules

            except Exception as exc:
                st.error(f"Audit failed: {exc}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)

# --- RESULTS DASHBOARD ---
if "audit_report" in st.session_state:
    report = st.session_state["audit_report"]
    audited_rules = st.session_state["audited_rules"]

    st.divider()
    st.subheader(f"📊 Compliance Report: `{report.contract_filename}`")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Rules Audited", report.total_rules)
    m2.metric("✅ Passed", report.passed_count)
    m3.metric("❌ Failed", report.failed_count)
    m4.metric("⚠️ Missing", report.missing_count)

    # Exportable JSON report download
    export_payload = {
        "contract_filename": report.contract_filename,
        "summary": {
            "total_rules": report.total_rules,
            "passed": report.passed_count,
            "failed": report.failed_count,
            "missing": report.missing_count,
        },
        "results": [
            {
                "rule_id": rule.rule_id,
                "rule_name": rule.rule_name,
                "category": rule.category,
                **ev.model_dump(),
            }
            for rule, ev in zip(audited_rules, report.evaluations)
        ],
    }
    st.download_button(
        label="⬇️ Download Audit Report (JSON)",
        data=json.dumps(export_payload, indent=2),
        file_name=f"{Path(report.contract_filename).stem}_audit_report.json",
        mime="application/json",
    )

    st.markdown("### Clause-by-Clause Findings")
    badge_map = {"PASS": "✅ PASS", "FAIL": "❌ FAIL", "MISSING": "⚠️ MISSING"}

    for rule, ev in zip(audited_rules, report.evaluations):
        badge = badge_map.get(ev.status, ev.status)
        location = (
            f"Page {ev.page_number} • {ev.section_title}"
            if ev.page_number
            else "Not found in contract"
        )
        expanded_by_default = ev.status != "PASS"

        with st.expander(
            f"{badge} | **{rule.rule_id}: {rule.rule_name}** ({location})",
            expanded=expanded_by_default,
        ):
            st.markdown(f"**Mandatory Requirement:** {rule.requirement}")
            st.markdown(f"**Attorney Risk Assessment:** {ev.risk_reasoning}")
            if ev.verbatim_quote:
                st.info(f"**Verbatim Contract Evidence (Page {ev.page_number}):**\n\n> {ev.verbatim_quote}")