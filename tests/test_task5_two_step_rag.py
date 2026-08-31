from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from langchain_learning_lab.task5_two_step_rag import (
    answer_with_context,
    build_augmented_prompt,
    format_context,
)


RETRIEVED = [
    (
        Document(
            page_content="普通生产变更窗口为每周二和周四。",
            metadata={"source": "handbook.md", "chunk_id": 2},
        ),
        0.82,
    )
]


def test_format_context_preserves_text_source_and_score() -> None:
    context = format_context(RETRIEVED)

    assert "普通生产变更窗口" in context
    assert "handbook.md#chunk-2" in context
    assert "0.8200" in context


def test_augmented_prompt_contains_question_and_retrieved_evidence() -> None:
    prompt = build_augmented_prompt("什么时候可以发布？", RETRIEVED)
    messages = prompt.to_messages()

    assert len(messages) == 2
    assert "只能使用检索资料" in str(messages[0].content)
    assert "什么时候可以发布" in str(messages[1].content)
    assert "handbook.md#chunk-2" in str(messages[1].content)


def test_answer_keeps_evidence_next_to_model_output() -> None:
    model = FakeListChatModel(
        responses=["每周二和周四。[handbook.md#chunk-2]"]
    )

    result = answer_with_context("什么时候可以发布？", RETRIEVED, model)

    assert result.answer == "每周二和周四。[handbook.md#chunk-2]"
    assert result.retrieved == RETRIEVED


def test_empty_retrieval_is_explicit_in_prompt() -> None:
    assert format_context([]) == "（没有检索到任何资料）"
