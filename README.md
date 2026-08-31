# LangChain Learning Lab

一个从 LangChain 基础抽象逐步推进到可测试 Agent 应用的学习项目。

项目覆盖 Message、Prompt、Model、LCEL、Tool Calling、LangGraph Agent、Memory、RAG、Structured Output、Middleware、Retry、Fallback、调用上限和 Human-in-the-loop，并以“智能订单客服 Agent”作为综合结项实战。

## 项目状态

```text
Task 1–6：已完成
综合实战：已完成
离线测试：69 passed
Python：3.11–3.14
包管理器：uv
```

完整测试均使用 Fake Model 或 Fake Embeddings，不访问外部服务；真实示例会调用 DeepSeek 或通义，需要本机凭据与网络。

## 学习路线

| 阶段 | 主题 | 主要成果 |
|---|---|---|
| Task 1 | LangChain 整体架构 | 理解 LLM、LangChain、LangGraph、LangSmith 的职责边界 |
| Task 2 | Model、Message、Prompt、LCEL | 统一模型输入、Prompt Template、Runnable Chain、流式输出 |
| Task 3 | Tool Calling | Tool Schema、手动工具循环、多工具请求与 `tool_call_id` |
| Task 4 | Agent 与 State | `create_agent`、LangGraph Runtime、Checkpointer、Streaming |
| Task 5 | RAG | 文档加载、切片、Embedding、VectorStore、Retriever、Two-step RAG |
| Task 6 | 工程化 Agent | Structured Output、Middleware、Retry、Fallback、Limits、HITL |
| Capstone | 智能订单客服 | RAG、Tools、Memory、审批、结构化交付与离线测试的完整组合 |

## 综合实战

`order_support_capstone` 实现了一个教学用订单客服 Agent：

```text
用户问题
  ↓
Order Support Agent
  ├─ query_order：查询订单状态
  ├─ search_order_policy：检索订单政策
  └─ cancel_order：取消订单
                         ↓
                    Human-in-the-loop
  ↓
ServiceResolution
```

关键边界：

- 模型只能提出操作，业务工具由应用服务执行；
- Repository 强制执行订单规则，不能依赖 Prompt 保证业务安全；
- 查询与政策检索是只读工具，可以针对瞬时错误重试；
- 取消订单是写操作，执行前必须人工审批；
- Agent 最终通过 Pydantic `ServiceResolution` 交付结构化结果；
- Checkpointer 使用 `thread_id` 保存对话和审批暂停现场；
- 调用上限防止模型或工具循环失控。

综合实战详细说明位于 [src/order_support_capstone/README.md](src/order_support_capstone/README.md)。

## 技术栈

- Python 3.11–3.14
- uv
- LangChain 1.x
- LangGraph Checkpoint
- `langchain-deepseek`
- DashScope `text-embedding-v4`
- Pydantic
- pytest

真实调用使用：

```text
Chat Model：DeepSeek
Embedding：通义 DashScope
```

离线测试使用：

```text
FakeMessagesListChatModel
DeterministicFakeEmbedding
InMemorySaver
```

## 环境初始化

进入项目：

```bash
cd /Users/xuchengbo/Documents/Code/PythonProject/langchain-learning-lab
```

安装依赖并创建项目虚拟环境：

```bash
uv sync
```

`uv` 会使用 `.python-version` 中指定的 Python，并在项目内创建 `.venv`，不会修改全局 Python 环境。

## 凭据管理

真实调用需要：

```text
DEEPSEEK_API_KEY
DASHSCOPE_API_KEY
```

本项目不把 Key 写入源码、README、`.env` 或 IDE 配置。

当前 macOS 环境使用 Keychain 保存凭据。终端可以提前提供环境变量；PyCharm 直接运行时，`credentials.py` 会执行安全回退：

1. 优先使用当前进程已有的环境变量；
2. 验证项目实际位于 `PythonProject` 目录下；
3. 从 macOS Keychain 读取对应凭据；
4. 只写入当前 Python 进程；
5. Python 进程结束后环境变量随之消失。

凭据加载实现位于 [credentials.py](src/langchain_learning_lab/credentials.py)。

## 快速验证

运行全部离线测试：

```bash
uv run pytest
```

当前预期：

```text
69 passed
```

测试不会调用 DeepSeek 或通义服务，也不需要 API Key。

## 示例运行

以下命令均在项目根目录执行。

### Task 1：Message

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task1_messages
```

构造与模型提供商无关的 `SystemMessage`、`HumanMessage` 和 `AIMessage`。

### Task 2：Prompt、LCEL 与流式输出

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task2_deepseek
```

使用 `ChatPromptTemplate | ChatDeepSeek` 构成 Runnable Chain，并观察流式 Chunk。该命令会访问 DeepSeek。

### Task 3：手动 Tool Calling Loop

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task3_manual_tool_loop
```

展示模型生成 `tool_calls`、Python 执行工具、`ToolMessage` 回传和模型最终总结。该命令会访问 DeepSeek。

### Task 4：Agent、Memory 与 Streaming

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task4_agent_loop
PYTHONPATH=src uv run python -m langchain_learning_lab.task4_memory
PYTHONPATH=src uv run python -m langchain_learning_lab.task4_streaming
```

- `task4_agent_loop`：`create_agent` 自动执行 `model → tools → model`；
- `task4_memory`：Checkpointer 与 `thread_id` 的短期记忆和线程隔离；
- `task4_streaming`：节点增量、Messages 和 `recursion_limit`。

### Task 5：RAG

检索阶段：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task5_retrieval
```

Two-step RAG：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task5_two_step_rag
```

Retriever 策略对比：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task5_retriever_strategies
```

示例覆盖 Similarity、Score Threshold 和 MMR。真实 Embedding 调用会访问通义；Two-step RAG 还会调用 DeepSeek。

### Task 6：Structured Output 与 Middleware

Model 级 Structured Output：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_structured_output
```

Agent 级 Structured Output：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_agent_structured_output
```

Middleware 生命周期：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_middleware_lifecycle
```

Model Retry 与 Tool Retry：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_retry_layers
```

Fallback 与调用上限：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_fallback_and_limits
```

Human-in-the-loop：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_human_in_the_loop
```

`task6_structured_output`、`task6_agent_structured_output` 和
`task6_middleware_lifecycle` 默认调用 DeepSeek；`task6_retry_layers`、
`task6_fallback_and_limits` 和 `task6_human_in_the_loop` 使用确定性 Fake Model，
可以完全离线运行。

### 综合实战：智能订单客服

```bash
PYTHONPATH=src uv run python -m order_support_capstone.main
```

入口会：

1. 创建 DeepSeek Chat Model 和通义 Embeddings；
2. 查询 `ORDER-1001` 状态与取消政策；
3. 输出第一次 `ServiceResolution`；
4. 请求取消订单并触发 `__interrupt__`；
5. 展示待审批 Action；
6. 使用固定 `approve` 恢复同一线程；
7. 输出最终结构化结果和 Repository 状态。

该命令会访问 DeepSeek 和通义服务，并使用本机凭据。

## 项目结构

```text
langchain-learning-lab/
├── data/
│   └── task5_handbook.md
├── src/
│   ├── langchain_learning_lab/
│   │   ├── credentials.py
│   │   ├── task1_messages.py
│   │   ├── task2_deepseek.py
│   │   ├── task3_manual_tool_loop.py
│   │   ├── task4_agent_loop.py
│   │   ├── task4_memory.py
│   │   ├── task4_streaming.py
│   │   ├── task5_retrieval.py
│   │   ├── task5_two_step_rag.py
│   │   ├── task5_retriever_strategies.py
│   │   ├── task6_structured_output.py
│   │   ├── task6_agent_structured_output.py
│   │   ├── task6_middleware_lifecycle.py
│   │   ├── task6_retry_layers.py
│   │   ├── task6_fallback_and_limits.py
│   │   └── task6_human_in_the_loop.py
│   └── order_support_capstone/
│       ├── domain.py
│       ├── repository.py
│       ├── order_tools.py
│       ├── retrieval.py
│       ├── agent.py
│       ├── main.py
│       ├── README.md
│       └── data/order_policy.md
├── study-notes/
│   ├── 01_LANGCHAIN_STUDY_NOTES_TASK1-5.md
│   └── 02_LANGCHAIN_STUDY_NOTES_TASK6_CAPSTONE.md
├── tests/
├── pyproject.toml
├── uv.lock
└── README.md
```

## 测试策略

测试重点是业务行为与 Agent 状态，不依赖模型随机发挥：

- Message、Prompt 与 Tool Schema；
- 手动 Tool Loop 和多工具请求；
- Agent 图节点、Streaming 与线程隔离；
- 文档切片、VectorStore、Retriever 和 RAG 输出；
- Pydantic Schema 与 Structured Output 重试；
- Middleware 生命周期；
- Model/Tool Retry；
- Model Fallback 与调用上限；
- HITL 的 `approve`、`edit`、`reject`；
- 订单 Repository 的业务规则与幂等性；
- 综合 Agent 的 Tool 调用和 `ServiceResolution`。

Fake Model 负责生成确定性的 Tool Call，Fake Embeddings 负责验证离线检索数据流。它们可以证明流程和约束正确，但不能代替真实模型质量评估、RAG 召回评估、网络故障演练和生产负载测试。

## 学习笔记

- [Task 1–5：基础、Tool、Agent 与 RAG](study-notes/01_LANGCHAIN_STUDY_NOTES_TASK1-5.md)
- [Task 6 与综合实战：Structured Output、Middleware、HITL](study-notes/02_LANGCHAIN_STUDY_NOTES_TASK6_CAPSTONE.md)

两份笔记记录了关键概念、代码流程、常见误区、测试结论和后续学习建议。

## 核心结论

```text
LLM
= 提供语言理解、生成和决策能力

LangChain
= 组织 Model、Message、Prompt、Tool、Retriever 和 Agent

LangGraph
= 执行有状态、可循环、可暂停恢复的图工作流

Tool
= 连接模型与业务能力，但由应用服务真正执行

RAG
= 为模型提供外部证据，不保证召回结果一定相关

Schema、Repository、Middleware、Checkpointer 和人工审批
= 构成 Agent 真正可交付的工程边界
```

## 后续方向

项目结项后，推荐继续学习：

1. LangSmith Trace 与 Evaluation；
2. 数据库型持久化 Checkpointer；
3. RAG 召回率、准确率和答案忠实度评估；
4. 写操作幂等键、补偿和不确定结果处理；
5. 身份、权限、审批审计和多实例部署；
6. 异步 Agent、并发 Tool Calling、超时与熔断。
