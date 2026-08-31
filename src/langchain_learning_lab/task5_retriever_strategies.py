"""Task 5: compare three VectorStoreRetriever search strategies."""

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, chain
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_learning_lab.task5_retrieval import (
    build_vector_store,
    create_embeddings,
    load_knowledge,
    split_documents,
)
from langchain_learning_lab.task5_two_step_rag import source_label


# This threshold is calibrated only for this teaching dataset and embedding model.
# It is not a universal definition of relevance.
TEACHING_SCORE_THRESHOLD = 0.50


def create_retrievers(
    vector_store: InMemoryVectorStore,
) -> dict[str, Runnable[str, list[Document]]]:
    """Expose different vector search policies through one Runnable interface."""
    # InMemoryVectorStore exposes raw cosine scores but does not currently define
    # the generic relevance-score conversion required by
    # search_type="similarity_score_threshold". Wrap its scored search instead.
    @chain
    def score_threshold_retriever(question: str) -> list[Document]:
        scored_results = vector_store.similarity_search_with_score(question, k=3)
        return [
            document
            for document, score in scored_results
            if score >= TEACHING_SCORE_THRESHOLD
        ]

    return {
        "similarity": vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 2},
        ),
        "score_threshold": score_threshold_retriever,
        "mmr": vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 2,
                "fetch_k": 3,
                "lambda_mult": 0.5,
            },
        ),
    }


def first_line(document: Document) -> str:
    """Return a short preview without changing the retrieved Document."""
    return next(
        (line.strip() for line in document.page_content.splitlines() if line.strip()),
        "(empty document)",
    )


def print_strategy_results(
    question: str,
    retrievers: dict[str, Runnable[str, list[Document]]],
) -> None:
    print(f"\nQuestion: {question}")
    for strategy_name, retriever in retrievers.items():
        documents = retriever.invoke(question)
        print(f"\n{strategy_name}: {len(documents)} document(s)")
        for rank, document in enumerate(documents, start=1):
            print(
                f"  [{rank}] {source_label(document)} | {first_line(document)}"
            )


def main() -> None:
    chunks = split_documents(load_knowledge())
    vector_store = build_vector_store(chunks, create_embeddings())
    retrievers = create_retrievers(vector_store)

    questions = [
        "普通生产发布安排在什么时间？",
        "公司食堂早餐几点开始供应？",
    ]
    for question in questions:
        print_strategy_results(question, retrievers)


if __name__ == "__main__":
    main()
