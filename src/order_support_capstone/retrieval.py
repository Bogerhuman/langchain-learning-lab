"""RAG construction for the fictional order-policy knowledge base."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool, tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

POLICY_PATH = Path(__file__).parent / "data" / "order_policy.md"


def load_policy_document(path: Path = POLICY_PATH) -> list[Document]:
    """Load policy text and attach source metadata."""
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.name},
        )
    ]

def split_policy_documents(documents: list[Document]) -> list[Document]:
    """Split policy documents into retrievable semantic chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=25,
        separators=["\n## ", "\n\n", "。", "；", "，", ""],
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index
    return chunks


def build_policy_retriever(
    chunks: list[Document], embeddings: Embeddings
) -> Runnable[str, list[Document]]:
    """Embed chunks and return a retriever with a bounded Top K."""
    vector_stores = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)

    retriever = vector_stores.as_retriever(search_type="similarity",search_kwargs={"k": 3})
    return retriever


def build_policy_search_tool(
    retriever: Runnable[str, list[Document]],
) -> BaseTool:
    """Expose policy retrieval to the Agent as a read-only Tool."""
    @tool('search_order_policy')
    def search_order_policy(query:str) -> list[Document]:
        """Search internal policy documents and return relevant excerpts with sources.

        Args:
            query: The search query describing what policy information is needed.

        Returns:
           document list containing policy content and source metadata.
        """
        docs = retriever.invoke(query)

        results = []
        if not docs:
            return results

        for doc in docs:
            results.append(
                {
                    "content": doc.page_content.strip(),
                    "source": doc.metadata.get("source", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                }
            )

        return results

    return search_order_policy
