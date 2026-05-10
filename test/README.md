# APC Keyword Cache — 测试套件

## 目录结构

```
test/
├── conftest.py                        # 共享 fixtures（fakeredis、mock LLM、预置数据）
├── unit/                              # 单元测试 — 纯函数，无外部依赖
│   ├── test_normalize.py              # L1 输入标准化
│   ├── test_fingerprint.py            # 上下文指纹 + 兼容性判定
│   ├── test_types.py                  # CtxFingerprint / PlanTemplate / CandidateResult
│   ├── test_sanitize.py               # 关键词清洗 + 实体注入检测
│   ├── test_candidates.py             # 候选短路逻辑
│   ├── test_kw_cache.py               # 漂移追踪 + 自适应 TTL
│   ├── test_llm_normalize.py          # LLM 归一化 + 四级回退链
│   ├── test_multilang.py              # 多语言模型注册 + 语言检测
│   ├── test_config.py                 # 配置管理
│   ├── test_template_gen_helpers.py   # rule_filter + classify_ttl
│   └── test_metrics.py                # Prometheus 指标结构验证
├── integration/                       # 集成测试 — 使用 fakeredis
│   ├── test_keyword_index.py          # KeywordIndexManager 全生命周期
│   ├── test_cache_lookup.py           # L1 / kw_cache / L2-L3 路径
│   ├── test_template_gen.py           # 模板回写所有缓存层
│   └── test_cache_eviction.py         # LRU 驱逐 + 级联清理
└── e2e/                               # 端到端测试 — FastAPI + LangGraph
    ├── test_api.py                    # HTTP 请求完整生命周期
    └── test_graph_workflow.py         # 图编排 6 条路由边全覆盖
```

## 运行

```bash
# 安装测试依赖
pip install pytest pytest-asyncio httpx fakeredis[lua]

# 全部
pytest test/ -v

# 分层运行
pytest test/unit/ -v         # 仅单元测试（无外部依赖）
pytest test/integration/ -v  # 集成测试（fakeredis）
pytest test/e2e/ -v          # 端到端测试（FastAPI + LangGraph）

# 覆盖率
pytest test/ --cov=apc_cache --cov-report=html
```

## 覆盖场景速查

| 层 | 关键测试 |
|----|---------|
| **normalize** | NFKC 归一化、空白合并、大小写、确定性、幂等 |
| **fingerprint** | 6 种 ctx 类型分类、schema 提取（max_depth=2）、兼容性 5 重短路与 |
| **candidates** | 空索引/全低于阈值→shortcut_new、强候选→reuse、领先不足→ask_llm |
| **llm_normalize** | LLM 复用/新建、超时回退 top-1、幻觉检测、实体注入重试 |
| **kw_cache** | 漂移窗口 (FIFO capped 100)、自适应 TTL 三档 (3600/1200/300) |
| **cache_lookup** | L1 命中、kw_cache 命中→L2/L3、ctx 不兼容 miss、L1 提升写入 |
| **template_gen** | 写入 tpl/tpl_idx/L1/kw_meta、别名解析、去重、ctx_fp 持久化 |
| **eviction** | 超限驱逐最旧、级联删 L1 refs→tpl_idx→空 keyword 清理 |
| **graph** | cache_lookup→small/large、planner→END/actor、actor→small/large |
| **api** | /health、/metrics/info、cache miss→L1 hit 回环、kw_cache→L2/L3 命中 |

## Fixtures 说明

| fixture | 作用域 | 说明 |
|---------|--------|------|
| `default_cfg` | function | 全默认 APCConfig |
| `small_cache_cfg` | function | cache_max_size=3（驱逐测试用） |
| `drift_sensitive_cfg` | function | 收紧的漂移阈值 |
| `fake_redis` | function | fakeredis 实例（每次测试全新） |
| `seeded_redis` | function | 预置 3 模板 + 2 keyword 的 fakeredis |
| `mock_llm` | function | MagicMock 模拟 LangChain LLM |
| `mock_kw_index` | function | MagicMock 模拟 KeywordIndexManager |
| `make_state` | function | AgentState dict 工厂函数 |
| `ctx_fp_finance` | function | 预构建的 CtxFingerprint (financial_report) |
| `api_client` | function | httpx AsyncClient (FastAPI + fakeredis + 注入图) |
