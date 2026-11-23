import re
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.ingestion import CHROMA_PATH, get_embedding_function
from src.schemas import ComplianceRule, DocumentChunk

# Cross-Encoder model trained on MS MARCO passage ranking for high-precision scoring
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _tokenize(text: str) -> list[str]:
    """Converts text into lowercase alphanumeric word tokens for BM25."""
    return re.findall(r"\b\w+\b", text.lower())


class HybridContractRetriever:
    """
    3-Stage Retrieval Funnel:
      1. Dense Vector Search (ChromaDB) + Sparse Keyword Search (BM25)
      2. Reciprocal Rank Fusion (RRF)
      3. Cross-Encoder Reranking
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        collection_name: str = "active_contract",
        chroma_path: str = CHROMA_PATH,
    ):
        if not chunks:
            raise ValueError("Cannot initialize retriever with an empty chunk list.")

        self.chunks = chunks
        # O(1) lookup table mapping chunk_id -> DocumentChunk object
        self.chunk_map: dict[str, DocumentChunk] = {c.chunk_id: c for c in chunks}

        # Stage 1A Setup: Connect to the existing ChromaDB vector index from Step 2
        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_collection(
            name=collection_name,
            embedding_function=get_embedding_function(),
        )

        # Stage 1B Setup: Build in-memory BM25 inverted index over tokenized chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Stage 3 Setup: Load Cross-Encoder weights once so all rules reuse them
        self.reranker = CrossEncoder(RERANKER_MODEL_NAME)

    def retrieve_for_rule(
        self,
        rule: ComplianceRule,
        candidate_k: int = 10,
        top_n: int = 4,
    ) -> list[DocumentChunk]:
        """Executes the 3-stage retrieval funnel for a single ComplianceRule."""
        # Tailored queries: natural prose for dense/cross-encoder, keyword-boosted for BM25
        semantic_query = f"{rule.category} - {rule.rule_name}: {rule.requirement}"
        keyword_query = f"{rule.rule_name} {' '.join(rule.search_keywords)} {rule.requirement}"

        # --- STAGE 1A: Dense Semantic Search (ChromaDB) ---
        n_results = min(candidate_k, len(self.chunks))
        chroma_res = self.collection.query(
            query_texts=[semantic_query],
            n_results=n_results,
        )
        vector_ids: list[str] = chroma_res["ids"][0] if chroma_res["ids"] else []

        # --- STAGE 1B: Sparse Keyword Search (BM25) ---
        bm25_scores = self.bm25.get_scores(_tokenize(keyword_query))
        ranked_bm25_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:n_results]
        # Filter out score == 0.0 so zero-match chunks never get fake rank boosts
        bm25_ids = [
            self.chunks[i].chunk_id
            for i in ranked_bm25_indices
            if bm25_scores[i] > 0
        ]

        # --- STAGE 2: Reciprocal Rank Fusion (RRF) ---
        rrf_scores: dict[str, float] = {}
        k_constant = 60

        for rank, cid in enumerate(vector_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank)

        for rank, cid in enumerate(bm25_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank)

        fused_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        candidates = [self.chunk_map[cid] for cid in fused_ids[: candidate_k * 2]]

        if not candidates:
            return []

        # --- STAGE 3: Cross-Encoder Reranking ---
        pairs = [(semantic_query, c.text) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        scored_candidates = sorted(
            zip(candidates, rerank_scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        return [chunk for chunk, _score in scored_candidates[:top_n]]