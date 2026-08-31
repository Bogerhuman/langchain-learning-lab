import importlib
import json

from langchain_core.embeddings import DeterministicFakeEmbedding

import order_support_capstone.retrieval as retrieval_module
from order_support_capstone.retrieval import (
    POLICY_PATH,
    build_policy_retriever,
    build_policy_search_tool,
    load_policy_document,
    split_policy_documents,
)


def test_policy_loader_keeps_content_and_source_metadata() -> None:
    documents = load_policy_document()

    assert len(documents) == 1
    assert "取消订单" in documents[0].page_content
    assert documents[0].metadata["source"] == POLICY_PATH.name


def test_splitter_keeps_source_and_assigns_stable_chunk_ids() -> None:
    chunks = split_policy_documents(load_policy_document())

    assert len(chunks) > 1
    assert [chunk.metadata["chunk_id"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all(chunk.metadata["source"] == POLICY_PATH.name for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_retriever_returns_at_most_three_documents() -> None:
    chunks = split_policy_documents(load_policy_document())
    retriever = build_policy_retriever(
        chunks,
        DeterministicFakeEmbedding(size=64),
    )

    results = retriever.invoke("已发货订单能否取消")

    assert 0 < len(results) <= 3
    assert all("source" in result.metadata for result in results)
    assert all("chunk_id" in result.metadata for result in results)


def test_policy_tool_has_expected_contract_and_json_safe_output() -> None:
    chunks = split_policy_documents(load_policy_document())
    retriever = build_policy_retriever(
        chunks,
        DeterministicFakeEmbedding(size=64),
    )
    search_tool = build_policy_search_tool(retriever)

    assert search_tool.name == "search_order_policy"
    assert set(search_tool.args) == {"query"}

    results = search_tool.invoke({"query": "已发货订单能否取消"})
    assert isinstance(results, list)
    assert 0 < len(results) <= 3
    assert all(set(item) == {"content", "source", "chunk_id"} for item in results)
    json.dumps(results, ensure_ascii=False)


def test_importing_retrieval_module_has_no_demo_side_effects(capsys) -> None:
    importlib.reload(retrieval_module)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
