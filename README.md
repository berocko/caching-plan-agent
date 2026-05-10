# APC Keyword Cache Subsystem

用三层缓存漏斗替代单层 keyword 匹配的 agent plan 缓存系统。核心创新是让 LLM 从"凭空生成 keyword"变为"在向量召回的已有 keyword 候选中做复用/新建决策"。

## 架构

```
请求进入 POST /agent/run
  │
  ├─ L1 精确匹配 (< 1ms, 0 LLM)        → 命中直接返回模板
  ├─ kw_cache 关键词缓存 (< 1ms, 0 LLM) → 命中跳过 embedding+LLM
  ├─ 向量归一化 (~5ms, 可选 LLM)        → KNN 召回候选 → 短路决策
  ├─ L2 关键词索引 (O(1))               → keyword → {template_id}
  ├─ L3 上下文指纹筛选                    → ctx_fp 兼容性校验
  └─ MISS → Large LM 生成 → template_gen 回写所有层
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Redis (需要 Redis 7+)
redis-server --appendonly yes

# 启动服务
uvicorn apc_cache.main:app --host 0.0.0.0 --port 8000

# 发送请求
curl -X POST http://localhost:8000/agent/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Calculate the working capital ratio for Q3",
    "agent_id": "finance_agent",
    "tools": [{"name": "calculator"}, {"name": "data_fetcher"}],
    "tools_hash": "abc123def456"
  }'
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `APC_REDIS_URL` | `redis://localhost:6379/0` | Redis 连接 |
| `APC_EMBED_MODEL_NAME` | `all-MiniLM-L6-v2` | embedding 模型 |
| `APC_EMBED_MODEL_VER` | `v1` | 模型版本标签 |
| `APC_CACHE_MAX_SIZE` | `100` | 最大模板数 |
| `APC_KW_CACHE_TTL_MIN` | `300` | kw_cache 最小 TTL(s) |
| `APC_KW_CACHE_TTL_MAX` | `3600` | kw_cache 最大 TTL(s) |
| `APC_DRIFT_SAMPLE_RATE` | `0.05` | 漂移采样率 |
| `APC_CANDIDATE_DIST_HIGH` | `0.85` | 强复用阈值 |
| `APC_CANDIDATE_DIST_LOW` | `0.50` | 强新建阈值 |
| `APC_DECISION_LOG_ENABLED` | `True` | 是否写决策日志 |
| `APC_SNAPSHOT_ENABLED` | `False` | 是否启用 S3 快照 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

完整配置见 `apc_cache/config.py`。

## 项目结构

```
apc_cache/
├── main.py                # FastAPI + lifespan
├── graph.py               # LangGraph 图编排 (5 节点)
├── config.py              # 配置 (env → dataclass)
├── normalize.py           # L1 输入标准化
├── fingerprint.py         # 上下文结构化指纹
├── metrics.py             # Prometheus 指标
├── alert_rules.yml        # 告警规则
├── grafana_dashboard.json # 监控看板
├── cache/                 # 缓存层
│   ├── lookup.py          # 五条出口路径
│   ├── template_gen.py    # 模板回写
│   └── eviction.py        # 级联驱逐
├── keyword/               # 关键词子系统
│   ├── embedding.py       # 本地 embedding
│   ├── keyword_index.py   # 向量索引 + 同步
│   ├── candidates.py      # 候选短路
│   ├── kw_cache.py        # 自适应 TTL
│   ├── llm_normalize.py   # LLM 归一化 + 回退链
│   ├── sanitize.py        # 关键词清洗
│   └── types.py           # 核心类型
├── ops/                   # 运维工具
│   ├── snapshot.py        # S3 快照
│   ├── reindex.py         # 重索引脚本
│   └── alias_cli.py       # 别名管理
└── decision_log/          # 决策日志
    ├── models.py          # ORM
    └── writer.py          # 异步写入
```

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/agent/run` | POST | 主入口：执行 agent 并返回缓存命中结果 |
| `/health` | GET | 健康检查 |
| `/metrics/info` | GET | 子系统状态信息 |

### POST /agent/run

```json
// Request
{
  "query": "Calculate working capital ratio",
  "context": {"balance_sheet": {...}, "income_statement": {...}},
  "agent_id": "finance_agent",
  "tools": [{"name": "calculator"}, {"name": "data_fetcher"}],
  "tools_hash": "abc123"
}

// Response
{
  "cache_hit": true,
  "cache_hit_layer": "L2_L3",
  "keyword": "working_capital_ratio",
  "final_output": "# Plan (adapted from template)...",
  "iteration_count": 0
}
```

## 设计文档

- `plan/blueprint.md` — 实现蓝图（权威参考）
- `plan/overview.md` — 架构概述
- `plan/final.md` — 最终架构设计
- `report/architecture.md` — 系统架构说明
- `report/overview.md` — 项目概览
- `report/optimization.md` — 优化建议（未实施）

## 依赖

- Python 3.12+
- Redis 7+
- FastAPI + Uvicorn
- LangGraph
- sentence-transformers
- numpy
- prometheus-client
- PostgreSQL + pgvector（仅决策日志，可选）
