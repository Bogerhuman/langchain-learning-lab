from langchain_core.messages import HumanMessage, SystemMessage

from langchain_learning_lab.task1_messages import build_messages


def test_build_messages_keeps_roles_and_topic() -> None:
    messages = build_messages("LangGraph")

    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "LangGraph" in str(messages[1].content)
