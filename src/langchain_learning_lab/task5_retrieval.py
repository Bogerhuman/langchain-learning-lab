"""Task 5: build the retrieval half of a minimal two-step RAG pipeline."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_learning_lab.credentials import ensure_project_credential


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "task5_handbook.md"
EMBEDDING_MODEL = "text-embedding-v4"


def load_knowledge(path: Path = KNOWLEDGE_PATH) -> list[Document]:
    """Load one local Markdown file into LangChain's Document abstraction."""
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        )
    ]


def split_documents(documents: list[Document]) -> list[Document]:
    """Split source documents into overlapping chunks while retaining metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=260,
        chunk_overlap=40,
        separators=["\n## ", "\n\n", "。", "；", "，", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def create_embeddings() -> Embeddings:
    """Create the Tongyi embedding client using only the environment variable."""
    ensure_project_credential("DASHSCOPE_API_KEY")
    # The provider integration is imported only for real calls, so offline tests
    # do not require credentials or initialize a network client.
    from langchain_community.embeddings import DashScopeEmbeddings

    return DashScopeEmbeddings(model=EMBEDDING_MODEL)


def build_vector_store(
    chunks: list[Document], embeddings: Embeddings
) -> InMemoryVectorStore:
    """Embed chunks and keep their vectors and source text in process memory."""
    return InMemoryVectorStore.from_documents(chunks, embedding=embeddings)


def retrieve(
    question: str,
    vector_store: InMemoryVectorStore,
    *,
    k: int = 3,
) -> list[tuple[Document, float]]:
    """Embed a query and return the k chunks with the highest cosine similarity."""
    return vector_store.similarity_search_with_score(question, k=k)


def main() -> None:
    source_documents = load_knowledge()
    chunks = split_documents(source_documents)
    embeddings = create_embeddings()
    vector_store = build_vector_store(chunks, embeddings)

    question = "普通生产发布安排在什么时间？发布前有哪些要求？"
    results = retrieve(question, vector_store)

    print(f"Source documents: {len(source_documents)}")
    print(f"Chunks after splitting: {len(chunks)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"\nQuestion: {question}")
    print("\nRetrieved chunks (this step has not called DeepSeek):")

    for rank, (document, score) in enumerate(results, start=1):
        print(
            f"\n[{rank}] score={score:.4f} "
            f"source={document.metadata['source']} "
            f"chunk_id={document.metadata['chunk_id']}"
        )
        print(document.page_content)


if __name__ == "__main__":
    main()
