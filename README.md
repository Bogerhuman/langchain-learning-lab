# LangChain Learning Lab

这是 LangChain 一日学习计划的 Task 1 项目。目标是先理解框架中的核心边界，而不是立即绑定某一家模型服务。

## Task 1 学习目标

- LLM：生成与推理能力的来源。
- LangChain：组织模型、消息、工具和 Agent 等 AI 应用组件。
- LangGraph：为有状态、可循环的 Agent 工作流提供底层运行时。
- LangSmith：用于追踪、调试和评估；本 Task 不要求安装或配置。

当前最小示例只使用 LangChain 的 `Message` 抽象，因此不联网、不调用模型，也不需要 API Key。等你明确选择模型提供商后，再增加对应的集成包和环境变量。

## 环境说明

- 检测到系统 Python：3.14.7
- 检测到包管理器：uv 0.12.3
- 项目要求：Python 3.11 至 3.14
- 虚拟环境：项目内 `.venv`，不会修改全局 Python 或复用上级目录的 `venv`

## 初始化

进入项目目录：

```bash
cd /Users/xuchengbo/Documents/Code/PythonProject/langchain-learning-lab
```

创建隔离环境并安装依赖：

```bash
uv sync
```

`uv sync` 会依据 `.python-version` 使用 Python 3.14，并在当前项目创建 `.venv`。它不会把包安装到全局环境。

## 运行示例

```bash
uv run python -m langchain_learning_lab.task1_messages
```

预期会打印一条 system message 和一条 human message。这展示了应用如何先构造与模型提供商无关的标准消息；真正调用模型是后续单独增加的一层。

运行测试：

```bash
uv run pytest
```

## 目录结构

```text
langchain-learning-lab/
├── .env.example
├── .gitignore
├── .python-version
├── pyproject.toml
├── README.md
├── src/
│   └── langchain_learning_lab/
│       ├── __init__.py
│       └── task1_messages.py
└── tests/
    └── test_task1_messages.py
```

## 自检问题

1. LangChain 与 LLM 各自负责什么？
2. 为什么这个示例没有 API Key 仍然能够运行？
3. `SystemMessage` 和 `HumanMessage` 为什么不直接写成普通字符串？
4. LangChain Agent 与 LangGraph 的关系是什么？
5. 如果完全不用 LangChain，应用需要自己管理哪些部分？

## 下一步（等待你决定）

选择模型提供商后，可增加一个真实模型调用示例。届时只安装对应的集成包，并通过本地 `.env` 提供 Key；`.env` 已被 Git 忽略。当前项目没有替你选择提供商，也没有配置任何凭据。

## Task 2：DeepSeek、Message 与 Prompt Template

Task 2 使用官方 `langchain-deepseek` 集成和 `deepseek-v4-flash` 模型。密钥只从
`DEEPSEEK_API_KEY` 环境变量读取，不会保存在项目中。

在本项目目录运行一次真实调用：

```bash
uv run python -m langchain_learning_lab.task2_deepseek
```

观察输出中的三层对象：

1. `ChatPromptTemplate` 保存可复用的提示词结构和变量占位符。
2. 模板填充变量后产生 `SystemMessage` 与 `HumanMessage`。
3. Chat Model 接收消息列表并返回 `AIMessage`，其中还可能包含 token 用量等元数据。

运行不访问网络的单元测试：

```bash
uv run pytest
```

## Task 3：手动 Tool Calling 循环

运行：

```bash
uv run python -m langchain_learning_lab.task3_manual_tool_loop
```

该示例刻意不使用 Agent，完整展示：模型生成 `tool_calls`、Python 根据工具名执行函数、
工具结果以 `ToolMessage` 回传，以及模型结合结果生成最终回答。

## Task 4：create_agent 自动循环

运行：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task4_agent_loop
```

该示例使用和 Task 3 相同的工具，但不再手写工具分发循环。`create_agent()` 底层生成
LangGraph 图，由 `model` 与 `tools` 节点自动产生完整消息轨迹。

短期记忆与线程隔离：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task4_memory
```

该示例使用一个 `InMemorySaver` 保存 Agent State。相同 `thread_id` 可以读取前一轮消息，
不同 `thread_id` 使用相互隔离的消息历史。

查看 Agent 每个节点完成后的状态更新：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task4_streaming
```

该示例使用 `stream_mode="updates"` 展示 `model -> tools -> model`，并通过
`recursion_limit` 限制单次 Agent 执行的最大图步骤数。

## Task 5：Two-step RAG 的检索阶段

本阶段先不调用 DeepSeek，只观察 RAG 的前半段：读取本地文档、切分 Chunk、使用通义
`text-embedding-v4` 生成向量，再从内存向量库召回与问题最相近的内容。

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task5_retrieval
```

示例知识库位于 `data/task5_handbook.md`，内容全部为虚构资料。运行输出会包含数据源数量、
Chunk 数量、相似度分数和被召回的原文；看到这些结果并不代表模型已经生成答案。

离线测试不会访问通义服务：

```bash
PYTHONPATH=src uv run pytest
```

### PyCharm 中的凭据加载

终端会通过 Zsh 目录钩子加载凭据，但 PyCharm 的运行按钮不会执行 `~/.zshrc`。项目中的
`credentials.py` 因此提供了安全回退：仅当项目实际位于 `PythonProject` 目录下时，才从
macOS Keychain 读取对应凭据，并且只写入当前 Python 进程的环境变量。Key 不会写入源码、
`.env` 或 PyCharm 配置文件，子进程结束后该环境变量也随之消失。

### Two-step RAG：检索后生成

完成检索阶段后，可以把 Top K 原文、来源标签和用户问题一起交给 DeepSeek：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task5_two_step_rag
```

程序会连续询问一个知识库内问题和一个知识库外问题，用于观察“有依据回答”和“资料不足”
两种行为。该流程固定为先检索、后生成，DeepSeek 不会决定是否检索。

### Retriever 与召回策略

对比普通相似度、最低分数阈值和 MMR：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task5_retriever_strategies
```

`VectorStore` 保存向量并实现搜索算法；`VectorStoreRetriever` 把具体搜索策略封装为统一的
Runnable 接口。示例中的 `0.50` 阈值只针对当前教学数据校准，不能直接复制到其他模型或
知识库。当前 `InMemoryVectorStore` 没有实现通用 relevance-score 转换，因此阈值策略使用
`@chain` 把已有的带分数搜索包装成同样可 `invoke()` 的 Runnable Retriever。

## Task 6：Structured Output

对比普通自然语言响应、原始结构化响应和经过 Pydantic 校验的业务对象：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_structured_output
```

示例使用 `with_structured_output(SupportTicket, include_raw=True)`，因此可以同时观察
`raw`、`parsed` 和 `parsing_error`。Pydantic Schema 还会拒绝非法枚举值、范围外数值和
未声明字段。
