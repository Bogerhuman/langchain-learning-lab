import math

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_learning_lab.task5_retriever_strategies import create_retrievers


class SmallEmbeddings(Embeddings):
    vocabulary = ("发布", "测试", "报销", "食堂", "天气")

    def _embed(self, text: str) -> list[float]:
        vector = [float(text.count(word)) for word in self.vocabulary]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_test_store() -> InMemoryVectorStore:
    documents = [
        Document(page_content="发布前必须完成测试", metadata={"chunk_id": 0}),
        Document(page_content="报销需要发票", metadata={"chunk_id": 1}),
        Document(page_content="食堂供应早餐", metadata={"chunk_id": 2}),
    ]
    return InMemoryVectorStore.from_documents(documents, SmallEmbeddings())


def test_retrievers_share_invoke_interface_but_keep_different_policies() -> None:
    retrievers = create_retrievers(build_test_store())

    assert retrievers["similarity"].search_type == "similarity"
    assert retrievers["similarity"].search_kwargs == {"k": 2}
    assert retrievers["score_threshold"].invoke("发布测试")[0].page_content == (
        "发布前必须完成测试"
    )
    assert retrievers["mmr"].search_type == "mmr"
    assert retrievers["mmr"].search_kwargs["fetch_k"] == 3


def test_similarity_retriever_returns_documents_without_scores() -> None:
    retriever = create_retrievers(build_test_store())["similarity"]

    documents = retriever.invoke("发布测试")

    assert len(documents) == 2
    assert all(isinstance(document, Document) for document in documents)
    assert documents[0].page_content == "发布前必须完成测试"


def test_threshold_retriever_can_return_no_documents() -> None:
    retriever = create_retrievers(build_test_store())["score_threshold"]

    documents = retriever.invoke("天气如何")

    assert documents == []
