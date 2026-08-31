# Order Support Capstone

这是 LangChain 学习的最终综合项目。框架已经提供类型、模块边界和业务验收条件，核心实现以 `TODO(milestone-N)` 标记。

## 模块关系

```text
main.py
  └─ agent.py
      ├─ order_tools.py ── repository.py ── domain.py
      └─ retrieval.py ── data/order_policy.md
```

## Milestone 1：领域与 Tools

只完成 `repository.py` 和 `order_tools.py` 中的 TODO，不创建 Agent。

验收条件：

1. 查询三条样例订单以及不存在的订单。
2. 只有 `pending` 可以真正变更为 `cancelled`。
3. 空原因、已发货、已取消和不存在分别返回稳定错误码。
4. 重复取消不会再次增加 `cancel_count`。
5. Tool 返回 JSON-safe dict，并保留名称、说明和参数 Schema。

## Milestone 2：政策 RAG

完成 `retrieval.py`，验证库内问题能召回政策原文及来源，库外问题不能假装有可靠依据。

## Milestone 3：Agent、Memory 与 Structured Output

完成 `agent.py` 的基本装配，让模型自行选择查询订单或检索政策，并以 `ServiceResolution` 交付结果。

## Milestone 4：稳定性与 HITL

添加 Retry、Call Limit 和 Human-in-the-loop。`query_order` 与政策检索可以自动执行；`cancel_order` 必须暂停审批。

## Milestone 5：端到端验收

完成 `main.py` 和离线测试，覆盖查询、政策问答、批准取消、编辑参数、拒绝取消、线程隔离和调用上限。

## 查找待办项

在项目根目录运行：

```bash
rg "TODO\(milestone-" src/order_support_capstone
```

当前阶段从 Milestone 1 开始。实现后运行现有回归测试，确保旧示例仍然通过：

```bash
uv run pytest
```
