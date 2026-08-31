import math
from collections.abc import Sequence

from langchain_core.embeddings import Embeddings

from langchain_learning_lab.task5_retrieval import (
    build_vector_store,
    load_knowledge,
    retrieve,
    split_documents,
)


class KeywordEmbeddings(Embeddings):
    """Small deterministic embedding used only by offline tests."""

    vocabulary: Sequence[str] = (
        "生产",
        "变更",
        "发布",
        "测试",
        "评审",
        "报销",
        "住宿",
        "远程",
        "学习",
    )

    def _embed(self, text: str) -> list[float]:
        vector = [float(text.count(word)) for word in self.vocabulary]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_split_documents_keeps_source_and_adds_chunk_id() -> None:
    chunks = split_documents(load_knowledge())

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all(chunk.metadata["source"] == "task5_handbook.md" for chunk in chunks)


def test_retrieve_returns_relevant_release_chunk_without_network() -> None:
    chunks = split_documents(load_knowledge())
    vector_store = build_vector_store(chunks, KeywordEmbeddings())

    results = retrieve("生产发布前要测试和评审吗？", vector_store, k=1)

    assert len(results) == 1
    document, score = results[0]
    assert "普通生产变更窗口" in document.page_content
    assert "自动化测试" in document.page_content
    assert score > 0
