"""Task 2: invoke DeepSeek with LangChain messages and a prompt template."""

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from langchain_learning_lab.credentials import ensure_project_credential

MODEL_NAME = "deepseek-v4-flash"

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a patient Java teacher. Explain concepts accurately "
            "for a {level} learner, using no more than three sentences."
            "start a new line when explain a different concept",

        ),
        ("human", "Please explain: {topic}"),
    ]
)


def build_messages(topic: str, level: str = "beginner") -> list[BaseMessage]:
    """Render reusable template variables into provider-independent messages."""
    return PROMPT.invoke({"topic": topic, "level": level}).to_messages()


def create_model() -> ChatDeepSeek:
    """Create the model after ensuring its project-scoped credential exists."""
    ensure_project_credential("DEEPSEEK_API_KEY")

    return ChatDeepSeek(
        model=MODEL_NAME,
        temperature=0,
        max_tokens=1000,
        reasoning_effort="low",
        timeout=30,
        max_retries=2,
        extra_body={
            "thinking": {
                "type": "enabled",
            }
        },
    )


# def main() -> None:
#     # messages = build_messages(
#     #     topic="the difference between Sychronized and ReentranctLock",
#     #     level="java beginner",
#     # )
#     #
#     # print("Messages sent to the model:")
#     # for message in messages:
#     #     print(f"- {message.type}: {message.content}")
#
#     # response = create_model().invoke(messages)
#     # 1 model.invoke("Explain JVM memory areas")
#     # response = create_model().invoke("Explain JVM memory areas")
#
#     # 2 model.invoke([
#     #     SystemMessage(content="You are a Java teacher."),
#     #     HumanMessage(content="Explain JVM memory areas."),
#     # ])
#
#     # 3 model.invoke(
#     #     PROMPT.invoke({
#     #         "level": "beginner",
#     #         "topic": "JVM memory areas",
#     #     })
#     # )
#     response = create_model().invoke(
#         PROMPT.invoke({
#             "level": "beginner",
#             "topic": "JVM memory areas",
#         })
#     )
#
#     print(f"\nResponse type: {response.type}")
#     print(f"Response content:\n{response.content}")
#     print(f"\nToken usage: {response.usage_metadata}")

def main() -> None:
    model = create_model()
    chain = PROMPT | model

    # response = chain.invoke({
    #     "level": "senior Java developer",
    #     "topic": "the JVM happens-before relationship",
    # })
    #
    # print(f"Response type: {response.type}")
    # print(f"Response content:\n{response.content}")
    # print(f"Token usage: {response.usage_metadata}")
    # print(f"Response metadata: {response.response_metadata}")

    chunks = chain.stream({
        "level": "senior Java developer",
        "topic": "the JVM happens-before relationship",
    })

    for chunk in chunks:
        if chunk.content:
            print(chunk.content, end="", flush=True)

    print()


if __name__ == "__main__":
    main()
