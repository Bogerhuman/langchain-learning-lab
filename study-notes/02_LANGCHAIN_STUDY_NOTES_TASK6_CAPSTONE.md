# LangChain 当日学习总结（二）

日期：2026-08-31  
项目：`langchain-learning-lab`  
前置笔记：[Task 1–5 学习总结](./01_LANGCHAIN_STUDY_NOTES_TASK1-5.md)

## 1. 今日完成情况

| 阶段 | 内容 | 状态 |
|---|---|---|
| Task 6.1 | Model 级 Structured Output | 已完成 |
| Task 6.2 | Agent 级 Structured Output | 已完成 |
| Task 6.3 | Middleware 生命周期 | 已完成 |
| Task 6.4 | Model Retry 与 Tool Retry | 已完成 |
| Task 6.5 | Model Fallback 与调用上限 | 已完成 |
| Task 6.6 | Human-in-the-loop | 已完成 |
| 综合实战 1 | 订单领域、Repository 与 Tools | 已完成 |
| 综合实战 2 | 订单政策 RAG 与检索 Tool | 已完成 |
| 综合实战 3 | Agent、Memory 与 Structured Output | 已完成 |
| 综合实战 4 | Middleware 与人工审批 | 已完成 |
| 综合实战 5 | 真实模型入口 | 已完成代码装配 |

今天完成了从“能够运行 Agent”到“构建可校验、可恢复、可审批、有限额且可测试的 Agent 应用”的进阶过程。

离线完整验收结果：

```text
69 passed
```

---

## 2. Structured Output：从自然语言变成业务对象

### 2.1 为什么需要结构化输出

普通模型返回的是自然语言：

```text
这个问题属于数据库故障，优先级很高，需要人工介入。
```

业务接口更希望获得稳定对象：

```json
{
  "category": "database",
  "priority": "critical",
  "requires_human": true
}
```

Structured Output 的目的不是让回答看起来像 JSON，而是让输出经过 Schema 校验后成为业务系统可以直接消费的对象。

### 2.2 Pydantic Schema 的职责

Schema 可以限制：

- 字段名称与类型；
- 枚举范围；
- 数字最小值与最大值；
- 字符串长度；
- 必填字段；
- 是否允许额外字段。

示例：

```python
class SupportTicket(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["database", "network", "application", "other"]
    priority: Literal["low", "medium", "high", "critical"]
    confidence: float = Field(ge=0, le=1)
```

`Literal` 表示只能从给定值中选择；`Field(ge=0, le=1)` 表示数值必须位于闭区间 `[0, 1]`；`extra="forbid"` 拒绝业务接口未声明的额外字段。

Schema 只负责定义和校验对象，不代表模型已经执行了某个业务操作。

### 2.3 Model 级 with_structured_output

```python
structured_model = model.with_structured_output(
    SupportTicket,
    include_raw=True,
)
```

`include_raw=True` 时通常可以观察：

```text
raw
= 模型原始 AIMessage，包括元数据和 Token 用量

parsed
= 校验成功后的 Pydantic 对象

parsing_error
= 解析或 Schema 校验失败信息
```

如果不需要模型原始响应，可以不保留 `raw`；如果需要调试、Token 统计或审计，保留原始响应更有价值。

### 2.4 Agent 级 ToolStrategy

```python
agent = create_agent(
    model=model,
    tools=tools,
    response_format=ToolStrategy(
        SupportTicket,
        handle_errors=True,
    ),
)
```

Agent 最终把已校验对象放入：

```python
result["structured_response"]
```

完整状态中仍保留 `messages`，因此诊断轨迹与业务交付可以分开：

```text
messages
= 模型、工具和结构化提交的完整过程

structured_response
= 业务系统最终消费的对象
```

当模型生成的结构不合法时，`ToolStrategy(handle_errors=True)` 可以把校验错误作为反馈交给模型，让模型修正后重新提交。

---

## 3. Middleware：Agent 执行过程中的统一切面

### 3.1 Middleware 的定位

Middleware 类似 Java/Spring 中的 AOP：业务代码不需要在每个模型或工具调用周围重复编写日志、重试、限流和审批逻辑。

常见 Hook：

```text
before_agent
after_agent
before_model
after_model
wrap_model_call
wrap_tool_call
```

### 3.2 Hook 次数不是固定的

一次典型 Agent Loop：

```text
before_agent
  ↓
before_model
wrap_model_call
after_model
  ↓
tool
  ↓
before_model
wrap_model_call
after_model
  ↓
after_agent
```

如果第一次模型调用产生 Tool Request，工具执行完成后还需要第二次模型调用来生成最终回答。因此：

- `before_agent` / `after_agent` 在一次正常完整 Agent 运行中通常执行一次；
- Model Hook 会按真实模型调用次数执行；
- Tool Hook 会按工具执行次数执行；
- Agent Loop 中模型和工具次数可能动态变化。

### 3.3 为什么计时更适合 wrap_model_call

```python
def wrap_model_call(self, request, handler):
    started = time.perf_counter()
    try:
        return handler(request)
    finally:
        elapsed = time.perf_counter() - started
```

`wrap_model_call` 精确包裹一次模型调用，适合耗时、异常和重试观测。生产环境中 Middleware 实例可能被并发请求共享，不能把非线程安全的可变统计直接保存在普通实例字段中。

---

## 4. Retry：失败发生在哪一层，就在哪一层重试

### 4.1 Model Retry

```python
ModelRetryMiddleware(
    max_retries=2,
    retry_on=(TimeoutError,),
    on_failure="error",
)
```

Model Retry 处理模型 API 的临时失败，例如超时或连接错误。失败尝试发生在模型调用包装层中，在最终成功前不会分别变成 Agent Message。

### 4.2 Tool Retry

```python
ToolRetryMiddleware(
    max_retries=2,
    tools=["query_inventory"],
    retry_on=(TimeoutError,),
    on_failure="error",
)
```

Tool Retry 处理工具服务的临时失败。成功前的内部失败不会各自产生一条 `ToolMessage`；最终模型通常只看到一次成功结果，或者在重试耗尽后看到最终失败。

### 4.3 只读与写操作的重试边界

```text
查询库存、查询订单、检索知识库
= 通常是只读操作，可以针对瞬时错误自动重试

取消订单、创建记录、转账、发送消息
= 会修改外部状态，不应在没有幂等机制时自动重试
```

即使客户端收到超时，也不能证明服务端没有成功。写操作重试需要业务幂等键、状态查询或补偿机制。

### 4.4 retry_on 的作用

只重试明确认定为临时故障的异常：

```python
retry_on=(TimeoutError, ConnectionError)
```

参数错误、权限错误和业务拒绝不应该盲目重试。重试主要解决系统稳定性，不应把每次基础设施异常都交给模型理解；这些错误还应进入日志、指标和告警系统。

---

## 5. Model Fallback：主模型失败后切换备用模型

### 5.1 Fallback 与 Retry 的区别

```text
Retry
= 使用同一个模型重新尝试

Fallback
= 主模型最终失败后，切换另一个模型或提供商
```

```python
ModelFallbackMiddleware(fallback_model)
```

Fallback 通常由模型调用异常触发。主模型正常返回一个质量很差的答案，并不会自动触发 Fallback。

如果业务希望“低质量结果也降级”，必须增加结果评估，并明确把不合格结果路由到备用模型；普通的事后业务异常不会天然被 Fallback 捕获。

### 5.2 备用模型的兼容要求

备用模型需要兼容当前 Agent 所依赖的能力：

- Tool Calling；
- Structured Output；
- 当前上下文长度；
- 消息格式；
- 必要的模型参数和安全策略。

否则虽然模型成功切换，Agent 仍可能无法继续当前流程。

### 5.3 重试与降级的调用放大

```text
模型客户端内部重试
× ModelRetryMiddleware
× 多个 Fallback 模型
```

多层机制叠加后，最坏请求次数和费用可能快速增加，因此还必须配置更高层的调用上限。

---

## 6. 调用上限与 recursion_limit

### 6.1 ModelCallLimit

```python
ModelCallLimitMiddleware(
    run_limit=6,
    thread_limit=20,
    exit_behavior="end",
)
```

- `run_limit`：单次 `agent.invoke()` 中允许的模型调用次数；
- `thread_limit`：同一 `thread_id` 跨多次运行累计允许的模型调用次数。

`thread_limit` 必须配合 Checkpointer 和稳定的 `thread_id`，否则无法跨运行保存累计计数。

### 6.2 ToolCallLimit

```python
ToolCallLimitMiddleware(
    tool_name="search_order_policy",
    run_limit=3,
    exit_behavior="continue",
)
```

Tool Limit 可以限制全部工具，也可以只限制一个具体工具。`continue` 会把超限请求变成失败的 `ToolMessage`，让模型根据已有信息继续规划。

### 6.3 三种边界的区别

| 机制 | 统计对象 | 主要目的 |
|---|---|---|
| `recursion_limit` | LangGraph 图执行步骤 | 防止整张图失控 |
| `ModelCallLimitMiddleware` | 模型调用次数 | 控制模型成本与循环 |
| `ToolCallLimitMiddleware` | 工具调用次数 | 控制工具配额与外部资源 |

工具节点执行不消耗 Model Call 次数；模型生成工具请求、工具执行、模型总结是不同阶段。

---

## 7. Human-in-the-loop：模型申请，人工授权

### 7.1 核心原则

```text
模型负责提出 Action
Middleware 负责暂停
Checkpointer 负责保存现场
人工负责授权
业务工具负责最终执行
```

模型产生 `cancel_order` Tool Call 不代表订单已经取消。在人工批准并恢复图执行之前，工具还没有运行。

### 7.2 配置审批策略

```python
HumanInTheLoopMiddleware(
    interrupt_on={
        "query_order": False,
        "cancel_order": InterruptOnConfig(
            allowed_decisions=["approve", "edit", "reject"]
        ),
    },
    description_prefix="订单取消操作等待人工审批",
)
```

未配置或配置为 `False` 的安全工具可以自动执行；高风险工具在模型响应后、工具执行前触发 Interrupt。

### 7.3 四种人工决定

| 决定 | 是否执行真实工具 | 行为 |
|---|---:|---|
| `approve` | 是 | 原工具、原参数执行 |
| `edit` | 是 | 执行人工编辑后的 Action |
| `reject` | 否 | 生成拒绝反馈，交给模型 |
| `respond` | 否 | 人工回答成为成功 ToolMessage |

`edit` 仍然处理原始 Tool Request，因此生成的 `ToolMessage` 保留原 `tool_call_id`。生产系统通常应限制只修改允许的参数，避免在审批过程中任意替换操作类型。

### 7.4 暂停与恢复

```python
config = {"configurable": {"thread_id": "approval-001"}}

paused = agent.invoke(inputs, config=config)

resumed = agent.invoke(
    Command(
        resume={"decisions": [{"type": "approve"}]}
    ),
    config=config,
)
```

恢复时必须使用相同 `thread_id`。`__interrupt__` 是给业务界面展示的审批单；Checkpoint 才是 LangGraph 恢复执行所需的完整运行现场。

多个待审批 Action 需要提交同样数量的决定，并按照 `action_requests` 的顺序一一对应，不是依赖业务代码自行按 ID 重新排序。

### 7.5 HITL 不是完整权限系统

业务系统仍需负责：

- 审批人身份认证；
- 审批权限判断；
- 参数二次校验；
- 审批超时；
- 操作幂等性；
- 决策与执行审计；
- 多实例与持久化恢复。

`InMemorySaver` 适合教学和测试，进程重启后数据会丢失；持续数小时或数天的审批需要数据库型 Checkpointer。

---

## 8. 综合实战：智能订单客服 Agent

### 8.1 总体架构

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

项目目录：

```text
src/order_support_capstone/
├── domain.py
├── repository.py
├── order_tools.py
├── retrieval.py
├── agent.py
├── main.py
├── README.md
└── data/
    └── order_policy.md
```

### 8.2 Milestone 1：业务边界与 Tools

模拟订单：

```text
ORDER-1001 = pending
ORDER-1002 = shipped
ORDER-1003 = cancelled
```

Repository 负责真正业务规则：

- 只有 `pending` 可以取消；
- `shipped` 不能直接取消；
- `cancelled` 重复取消不会产生第二次操作；
- 原因不能为空；
- 不存在的订单返回稳定错误码；
- 成功取消后状态变为 `cancelled`；
- `cancel_count` 只在真实状态变更时增加。

```text
Prompt 中的规则
= 帮助模型选择正确操作

Repository 中的规则
= 即使模型犯错也能保证业务安全
```

Tool Adapter 只负责：

```text
接收入参
→ 调用 Repository
→ Pydantic Result.model_dump(mode="json")
→ 返回 JSON-safe dict
```

### 8.3 Milestone 2：把 RAG 变成 Tool

```text
order_policy.md
→ Document
→ RecursiveCharacterTextSplitter
→ Chunks + source + chunk_id
→ InMemoryVectorStore
→ Retriever
→ search_order_policy Tool
```

Retriever 已经是 Runnable，因此调用方式为：

```python
documents = retriever.invoke(query)
```

不是：

```python
retriever.search(query)
```

Tool 返回结构化的 JSON-safe 数据：

```json
{
  "content": "状态为 shipped 的订单不能直接取消……",
  "source": "order_policy.md",
  "chunk_id": 1
}
```

测试用 `DeterministicFakeEmbedding` 只能验证数据流、Top K 和输出结构，不能证明语义排序质量；真实语义效果仍需使用实际 Embedding 和评估集。

模块导入不应自动生成向量或打印演示结果。演示代码必须放在 `main()` 或 `if __name__ == "__main__"` 中，避免 import 副作用和意外费用。

### 8.4 Milestone 3：Agent 装配与依赖注入

`build_order_support_agent()` 接收外部依赖：

```python
model
repository
embeddings
checkpointer
```

函数内部只负责组装，不创建 DeepSeek、通义客户端或 API Key，也不直接执行 `invoke()`。

```python
return create_agent(
    model=model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    response_format=ToolStrategy(ServiceResolution),
    checkpointer=checkpointer,
)
```

构建函数必须返回 `create_agent()` 产生的 Compiled StateGraph，不能创建后丢弃。

### 8.5 Milestone 4：稳定性和审批

最终 Agent 配置：

```text
ModelRetryMiddleware
ToolRetryMiddleware（只读工具）
ModelCallLimitMiddleware
ToolCallLimitMiddleware（政策检索）
HumanInTheLoopMiddleware（取消订单）
```

离线测试验证了三条路径：

```text
approve
→ 原始 ORDER-1001 被取消

edit
→ 只执行修改后的订单号，原请求订单不变

reject
→ Repository 不变，模型收到失败 ToolMessage
```

### 8.6 Milestone 5：真实入口

`main.py` 使用：

```text
DeepSeek Chat Model
通义 text-embedding-v4
InMemorySaver
稳定 thread_id
```

流程：

```text
查询订单状态与取消政策
→ 输出第一次 ServiceResolution
→ 请求取消订单
→ 展示 __interrupt__ 审批单
→ 固定 approve
→ 输出最终 ServiceResolution 和 Repository 状态
```

源码不保存 API Key，仍通过项目原有的 macOS Keychain 机制注入当前 Python 进程。

---

## 9. 测试策略

### 9.1 为什么大量使用 Fake Model

确定性 Fake Model 可以预先给出 Tool Call：

```text
第 1 次模型响应：query_order
第 2 次模型响应：ServiceResolution
```

这样可以验证 Agent Runtime、Messages、Tool 执行、Structured Output 和 Middleware，而不会受到模型随机性、网络和费用影响。

### 9.2 离线测试覆盖

```text
Repository 业务规则
Tool 名称、说明和参数 Schema
RAG 文档加载与切片
Retriever Top K
JSON-safe 检索结果
Agent model/tools 节点
ToolStrategy structured_response
Model/Tool Retry
Fallback 与调用上限
HITL approve/edit/reject
线程恢复与业务状态变化
```

最终完整测试：

```text
69 passed
```

离线测试证明代码路径和业务约束正确，但不能替代真实模型质量评估、网络故障演练和生产负载测试。

---

## 10. 今日纠正的重要误区

### 10.1 thread_limit 不是 invoke 次数

`thread_limit` 统计同一线程累计的模型或工具调用次数，具体取决于对应 Middleware；不是简单统计调用了多少次 `agent.invoke()`。

### 10.2 edit 不等于随意换工具

`edit` 表示编辑待审批 Action，通常只应修改原工具允许编辑的参数。它仍对应同一个 Tool Request。

### 10.3 幂等不等于只读

```text
只读
= 不修改业务状态

幂等
= 执行一次和多次的最终业务状态相同
```

“把状态设置为 cancelled”可以设计成幂等写操作，但它仍然修改数据，仍可能需要审批。

### 10.4 Top K 返回不等于有答案

普通相似度检索即使面对无关问题也会返回相对最接近的 Chunk，因此：

```text
检索成功
≠ 结果相关
≠ 知识库足以回答
≠ 最终回答正确
```

### 10.5 类型警告不一定是运行错误

PyCharm 对嵌套字典可能推断为 `dict[str, list[str]]`，而 HITL 要求 `InterruptOnConfig`。显式构造 TypedDict 可以让静态检查理解 Literal 限制：

```python
InterruptOnConfig(
    allowed_decisions=["approve", "edit", "reject"]
)
```

---

## 11. 项目代码索引

```text
src/langchain_learning_lab/
├── task6_structured_output.py          # Model 级结构化输出
├── task6_agent_structured_output.py    # Agent 级 ToolStrategy
├── task6_middleware_lifecycle.py       # Hook 生命周期与模型计时
├── task6_retry_layers.py               # Model Retry / Tool Retry
├── task6_fallback_and_limits.py        # Fallback 与调用上限
└── task6_human_in_the_loop.py          # approve / edit / reject

src/order_support_capstone/
├── domain.py                           # 领域对象与 ServiceResolution
├── repository.py                       # 订单业务规则与幂等性
├── order_tools.py                      # Repository 的 Tool Adapter
├── retrieval.py                        # 订单政策 Retriever 与 Tool
├── agent.py                            # Agent 和 Middleware 装配
├── main.py                             # 真实模型端到端入口
└── data/order_policy.md                # 教学用订单政策

tests/
├── test_order_support_capstone_milestone1.py
├── test_order_support_capstone_milestone2.py
├── test_order_support_capstone_milestone3.py
└── test_order_support_capstone_milestone4.py
```

---

## 12. 常用运行命令

进入项目：

```bash
cd /Users/xuchengbo/Documents/Code/PythonProject/langchain-learning-lab
```

运行全部离线测试：

```bash
PYTHONPATH=src uv run pytest
```

运行 Task 6 示例：

```bash
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_structured_output
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_agent_structured_output
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_middleware_lifecycle
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_retry_layers
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_fallback_and_limits
PYTHONPATH=src uv run python -m langchain_learning_lab.task6_human_in_the_loop
```

运行综合实战：

```bash
PYTHONPATH=src uv run python -m order_support_capstone.main
```

---

## 13. 当前掌握情况

### 已经掌握

- 用 Pydantic 把模型输出转换为业务契约；
- 区分原始 AIMessage、解析结果与 Agent State；
- 理解 Middleware 的 Agent、Model 和 Tool 生命周期；
- 区分 Model Retry、Tool Retry、Fallback 和调用上限；
- 理解内部失败尝试与 Agent Message 的边界；
- 使用 Checkpointer 和 `thread_id` 暂停、恢复 Agent；
- 将高风险工具置于 Human-in-the-loop 审批后；
- 把 Repository 作为模型之外的业务安全边界；
- 把 Retriever 包装为 Agent 可选择的 Tool；
- 使用依赖注入和 Fake Model 构建离线确定性测试；
- 将 RAG、Memory、Tools、Structured Output 和 Middleware 组合成完整应用。

### 继续强化

- Python 类型标注、导入顺序和统一格式；
- 异步 Agent 与并发工具执行；
- 数据库型持久化 Checkpointer；
- 写操作的业务幂等键与不确定结果处理；
- RAG 召回率、准确率和答案忠实度评估；
- LangSmith Trace、数据集和自动化 Evaluation；
- 真实用户身份、权限与审批审计；
- 多实例部署、超时、熔断和监控告警。

---

## 14. 下一阶段建议

推荐按以下顺序继续：

1. LangSmith Trace：观察 Agent、Model、Tool 和 Retriever 的完整调用链。
2. Evaluation：为订单客服准备固定问题集和预期行为。
3. 持久化 Checkpointer：将 `InMemorySaver` 替换为数据库存储。
4. RAG 评估：验证召回 Chunk 是否正确，而不只检查是否返回 Top K。
5. 生产化：身份、权限、幂等、审计、告警和成本控制。

---

## 15. 一句话总览

```text
模型负责理解和提出 Action，LangGraph 负责执行有状态流程，Middleware 负责横切控制；
Repository、Schema、Checkpointer 与人工审批共同构成真正可交付的业务边界。
```
