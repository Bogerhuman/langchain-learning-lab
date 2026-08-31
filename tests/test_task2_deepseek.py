from langchain_core.messages import HumanMessage, SystemMessage

from langchain_learning_lab.task2_deepseek import build_messages


def test_build_messages_renders_roles_and_variables() -> None:
    messages = build_messages(topic="prompt templates", level="intermediate")

    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "intermediate" in str(messages[0].content)
    assert "prompt templates" in str(messages[1].content)
