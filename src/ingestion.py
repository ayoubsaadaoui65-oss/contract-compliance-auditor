import re
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from llama_cloud import LlamaCloud
from src.schemas import DocumentChunk

load_dotenv()

# Central embedding config so ingestion and retrieval always use the exact same model
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_PATH = "./chroma_db"


def get_embedding_function():
    """Returns the local SentenceTransformer embedding function used across ingestion and retrieval."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME,
        normalize_embeddings=True,
    )


def parse_pdf_to_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """
    Uploads a PDF to LlamaCloud and parses it into clean Markdown page by page.
    Returns a list of (1-indexed page_number, markdown_text) tuples.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {path}")

    # 1. Initialize LlamaCloud client (automatically reads LLAMA_CLOUD_API_KEY from .env)
    client = LlamaCloud()

    # 2. Upload the PDF, then parse it using the 'agentic' tier with Markdown tables enabled
    uploaded_file = client.files.create(file=path, purpose="parse")
    result = client.parsing.parse(
        file_id=uploaded_file.id,
        tier="agentic",
        version="latest",
        expand=["markdown"],
        output_options={
            "markdown": {
                "tables": {"output_tables_as_markdown": True}
            }
        },
    )

    if not result.markdown or not result.markdown.pages:
        raise RuntimeError("LlamaParse returned no markdown pages.")

    pages: list[tuple[int, str]] = []
    for idx, page in enumerate(result.markdown.pages, start=1):
        md_text = (page.markdown or "").strip()
        if md_text:
            pages.append((idx, md_text))

    return pages


def chunk_markdown_pages(
    pages: list[tuple[int, str]],
    max_chunk_chars: int = 1200,
    overlap_chars: int = 150,
) -> list[DocumentChunk]:
    """
    Splits each page's Markdown along headings (#, ##, ###) first so legal clauses stay whole,
    and subdivides overly long sections with overlap while preserving exact page_number and section_title.
    """
    chunks: list[DocumentChunk] = []
    current_section = "General"

    for page_num, page_md in pages:
        # Split page into blocks whenever a Markdown heading (#, ##, ###) starts a line
        blocks = re.split(r"(?m)^(?=#{1,3}\s+)", page_md)
        page_chunk_idx = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            # Update current_section if this block begins with a Markdown heading
            first_line = block.splitlines()[0].strip()
            if first_line.startswith("#"):
                current_section = first_line.lstrip("#").strip()

            # Subdivide long blocks safely on paragraph boundaries with overlap
            sub_texts = _split_long_block(block, max_chunk_chars, overlap_chars)
            for text in sub_texts:
                page_chunk_idx += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"page_{page_num}_chunk_{page_chunk_idx}",
                        text=text,
                        page_number=page_num,
                        section_title=current_section,
                    )
                )

    return chunks


def _split_long_block(text: str, max_chars: int, overlap: int) -> list[str]:
    """Helper that splits an oversized section into smaller pieces with character overlap."""
    if len(text) <= max_chars:
        return [text]

    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end < len(text):
            break_pos = text.rfind("\n\n", start, end)
            if break_pos == -1:
                break_pos = text.rfind("\n", start, end)
            if break_pos > start + (max_chars // 2):
                end = break_pos

        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)

        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return pieces


def index_chunks_to_chroma(
    chunks: list[DocumentChunk],
    collection_name: str = "active_contract",
    chroma_path: str = CHROMA_PATH,
) -> chromadb.Collection:
    """
    Replaces any previous collection of the same name and indexes all DocumentChunks
    into a persistent local ChromaDB collection using cosine similarity.
    """
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        metadatas=[
            {"page_number": c.page_number, "section_title": c.section_title}
            for c in chunks
        ],
    )

    return collection


def ingest_contract(pdf_path: str | Path, collection_name: str = "active_contract") -> list[DocumentChunk]:
    """End-to-end ingestion entrypoint: PDF -> LlamaParse -> Chunks -> ChromaDB."""
    pages = parse_pdf_to_pages(pdf_path)
    chunks = chunk_markdown_pages(pages)
    index_chunks_to_chroma(chunks, collection_name=collection_name)
    return chunks