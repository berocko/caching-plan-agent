# APC Keyword 子系统 — 生产实现方案 v1

> 版本：pv1
> 基于：plan/final.md 最终架构设计
> 范围：keyword 缓存子系统的工程落地，含模块拆分、接口契约、配置、测试策略
> 前置依赖：v2 中已修复的 LangGraph 图拓扑（template_gen 路由 bug 已修）

---

## 1. 实现范围与边界

### 1.1 本次实现

| 模块 | 来源 | 优先级 |
|------|------|--------|
| L1 精确匹配层（含 ctx_fp） | final §2 | P0 |
| ctx_fingerprint 结构化计算 | final §6 | P0 |
| kw_cache + 自适应 TTL | final §3 | P1 |
| 向量归一化层（keyword_index + 候选短路） | final §4 | P1 |
| LLM 归一化 + 失败回退链 | final §4.4-4.5 | P1 |
| L2 关键词索引层 | final §5 | P0 |
| L3 ctx 筛选层 | final §6 | P0 |
| Embedding 模型生命周期 | final §8 | P1 |
| 多副本 ZSET 增量同步 | final §7 | P2 |
| 模板级联驱逐 | final §9.3 | P2 |
| 并发 keyword 写入保护 | final §9.4 | P2 |
| 冷启动快照恢复 | final §10 | P2 |
| 决策日志收集 | final §11.3 | P2 |
| Prometheus 指标 | final §11.1 | P1 |
| Keyword 别名管理 CLI | final §9.5 | P3 |

### 1.2 明确不实现

- Redis Stack VECTOR / pgvector 在线查询
- LLM 驱动的自动 keyword 合并
- 微批处理 encode（EmbedBatcher）
- Pub/sub 多副本同步
- 推测性预生成

---

## 2. 项目结构

```
src/apc/
├── __init__.py
├── config.py                  # 所有配置项集中管理
├── graph.py                   # LangGraph 图组装（修改：keyword_node 合并进 cache_lookup）
├── state.py                   # AgentState 定义（新增字段）
│
├── keyword/                   # ★ 本次新增核心模块
│   ├── __init__.py
│   ├── normalize.py           # normalize() — 无损表面标准化
│   ├── fingerprint.py         # compute_fingerprint() / ctx_compatible()
│   ├── embedding.py           # Embedding 模型加载 / encode / 版本管理
│   ├── kw_cache.py            # kw_cache 读写 + 漂移采样 + 自适应 TTL
│   ├── candidates.py          # build_candidates() + 候选短路逻辑
│   ├── llm_normalize.py       # LLM 归一化决策 + 失败回退链
│   ├── sanitize.py            # sanitize_keyword() + 反实体注入检测
│   ├── keyword_index.py       # keyword_index 内存管理 + KNN + ZSET 同步
│   └── types.py               # CandidateResult / KeywordMeta 等 dataclass
│
├── cache/                     # 缓存查找与写入（修改）
│   ├── __init__.py
│   ├── lookup.py              # cache_lookup 节点（合并 L1+L2+L3）
│   ├── template_gen.py        # template_gen 节点（写入 L1/L2/kw_index）
│   └── eviction.py            # 级联驱逐逻辑
│
├── decision_log/              # 决策日志（P2）
│   ├── __init__.py
│   ├── models.py              # SQLAlchemy model
│   └── writer.py              # 异步批量写入
│
├── ops/                       # 运维工具（P3）
│   ├── reindex.py             # reindex_keywords 脚本
│   ├── alias_cli.py           # keyword 别名管理 CLI
│   └── snapshot.py            # keyword_index S3 快照导出/恢复
│
└── metrics.py                 # Prometheus 指标定义（修改：新增指标）
```

### 2.1 与现有代码的关系

v1 中的 `keyword_node` + `cache_lookup_node` 是两个独立节点。本方案将它们**合并为单个 `cache_lookup` 节点**，内部执行完整的三层漏斗。LangGraph 图拓扑变化：

```
v1:  keyword_node → cache_lookup → route → small/large_planner → actor ⇄ planner → template_gen → END

pv1: cache_lookup（内部含 L1→kw_cache→向量归一化→L2→L3）→ route → small/large_planner → actor ⇄ planner → template_gen → END
```

合并理由：L1、kw_cache、向量归一化、L2、L3 五步之间有严格的顺序依赖和 early-return 语义，拆成多个 LangGraph 节点反而增加不必要的 checkpoint 开销和状态序列化成本。

---

## 3. AgentState 变更

```python
class AgentState(TypedDict):
    # ── 输入（不变）────────────────────────────
    query: str
    context: Any
    agent_id: str                          # 新增
    tools: list[ToolDef]                   # 新增（替代隐式获取）
    tools_hash: str                        # 新增

    # ── 关键词 & 缓存（变更）─────────────────
    keyword: Optional[str]                 # 保留
    cache_hit: bool                        # 保留
    cache_hit_layer: str                   # 新增："L1" | "L2_L3"
    plan_template: Optional[dict]          # 保留

    # ── 执行状态（不变）──────────────────────
    current_plan: Optional[str]
    actor_responses: list[str]
    execution_log: list[dict]
    iteration_count: int

    # ── 输出（不变）──────────────────────────
    final_output: Optional[str]
    is_complete: bool
```

变更要点：
- `agent_id` 和 `tools` 从隐式获取提升为显式 state 字段，保证三层 key 计算的可重复性
- `tools_hash` 在请求入口处预计算（`sha256(sorted_tool_names)`）
- `cache_hit_layer` 用于可观测性（区分 L1 命中还是 L2/L3 命中）

---

## 4. 模块接口契约

### 4.1 normalize.py

```python
def normalize(query: str) -> str:
    """仅做无损表面标准化：NFKC → 小写 → 合并空白符。
    不做语义归一化（停用词、同义词、数字占位符）。"""
    ...

def query_hash(query: str) -> str:
    """sha256(normalize(query))[:16]"""
    ...
```

### 4.2 fingerprint.py

```python
@dataclass
class CtxFingerprint:
    context_type: str          # "financial_report" | "tabular_data" | ...
    length_bucket: str         # "short" | "medium" | "long"
    tools: frozenset[str]
    agent_role: str
    context_schema: frozenset[str]


def compute_fingerprint(
    context: Any,
    tools: list[ToolDef],
    agent_role: str,
) -> CtxFingerprint:
    """计算结构化上下文指纹。对 context 做类型推断和 schema 提取，
    不包含具体值/实体。"""
    ...


def ctx_compatible(
    tpl_ctx: CtxFingerprint,
    query_ctx: CtxFingerprint,
) -> bool:
    """判断模板的上下文指纹是否兼容当前请求。
    context_type/tools/agent_role 精确匹配；
    length_bucket 允许相邻区间；
    context_schema 用包含关系（模板 ⊆ 请求）。"""
    ...


def fingerprint_hash(fp: CtxFingerprint) -> str:
    """sha256(json.dumps(fp, sort_keys=True, default=str))[:12]"""
    ...


# 内部辅助
def _classify_context_type(context: Any) -> str: ...
def _bucket_length(context: Any) -> str: ...
def _extract_schema(context: Any, max_depth: int = 2) -> frozenset[str]: ...
```

**实现注意事项**：
- `_extract_schema` 对 dict 提取前 2 层 key 名，对表格数据提取列名，对字符串返回空集
- `_classify_context_type` 用结构特征（key 名模式、类型标记）而非内容做推断
- `_bucket_length` 阈值：short < 1K tokens, medium 1K-10K, long > 10K

### 4.3 embedding.py

```python
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_MODEL_VER  = "v1"            # 模型/权重变更时手动 bump
EMBED_DIM        = 384

# 全局单例
embed_model: SentenceTransformer | None = None
embed_executor: ThreadPoolExecutor | None = None   # max_workers=2
embed_semaphore: asyncio.Semaphore | None = None    # value=4


async def init_embedding():
    """在 lifespan 中调用。加载模型 + 热身推理。"""

async def embed(text: str) -> np.ndarray:
    """异步安全的 embedding 调用。信号量限流 + 专用线程池。"""

def get_model_fingerprint() -> dict:
    """返回 {"name": ..., "ver": ..., "dim": ...}，供写入 kw_meta 和启动校验。"""
```

### 4.4 kw_cache.py

```python
DRIFT_SAMPLE_RATE = 0.05           # 5% 采样


async def get_cached_keyword(query: str) -> str | None:
    """查 kw_cache。命中时以 DRIFT_SAMPLE_RATE 概率触发旁路漂移检测。"""

async def set_cached_keyword(query: str, keyword: str):
    """写入 kw_cache。TTL 由 current_kw_cache_ttl() 决定。"""

def current_kw_cache_ttl() -> int:
    """基于漂移率动态返回 TTL（300 ~ 3600s）。由漂移检测回调更新。"""

def update_drift_rate(stable: int, changed: int):
    """漂移检测回调。更新滑动窗口内的漂移率，影响 current_kw_cache_ttl()。"""

# 内部
async def _check_drift(query: str, cached_kw: str):
    """异步旁路：重跑完整归一化，比对缓存值和最新值。更新漂移率。"""
```

### 4.5 candidates.py

```python
from .types import CandidateResult

DIST_HIGH = 0.85
DIST_LOW  = 0.50


async def build_candidates(
    query_vec: np.ndarray,
    kw_index: "KeywordIndexManager",
    top_k: int = 5,
) -> CandidateResult:
    """
    在 keyword_index 中 KNN 搜索 top_k 候选，根据距离分布做自适应短路决策。

    返回 CandidateResult:
      - action="shortcut_reuse": items 含最佳候选，直接复用，不调 LLM
      - action="shortcut_new":   items 为空，直接走 fallback_extract_keyword
      - action="ask_llm":        items 含 top-3 候选，喂给 LLM 决策
    """
```

### 4.6 types.py

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class CandidateResult:
    items: list[tuple[str, float]]           # [(keyword, cosine_distance), ...]
    action: Literal["shortcut_reuse", "shortcut_new", "ask_llm"]

    @property
    def top_keyword(self) -> str | None: ...

    @property
    def top_score(self) -> float | None: ...


@dataclass
class KeywordMeta:
    keyword: str
    embedding: np.ndarray
    model_ver: str
    dim: int
    created_at: float
```

### 4.7 llm_normalize.py

```python
NORMALIZE_TIMEOUT = 2.0              # LLM 调用超时（秒）


async def normalize_keyword(
    query: str,
    candidates: list[tuple[str, float]],
) -> str:
    """
    LLM 归一化决策主入口。

    内部流程：
      1. 调用 lightweight LLM（含超时控制）
      2. 成功 → 反幻觉校验（声称复用但不在候选 → 降级）
      3. 成功 → 反实体注入检测（含 query 实体 → retry）
      4. 任何失败 → 走回退链
    """

async def fallback_extract_keyword(query: str) -> str:
    """回退：纯 LLM 生成 keyword（v2 行为），不做向量候选。"""

async def retry_with_stricter_prompt(
    query: str, candidates: list[tuple[str, float]]
) -> str:
    """重试：更严格的 prompt，强调去实体化。"""


# 内部常量
KEYWORD_NORMALIZATION_PROMPT: str   # 归一化 prompt 模板
KEYWORD_FALLBACK_PROMPT: str        # 回退 prompt 模板
KEYWORD_STRICT_PROMPT: str          # 严格 prompt 模板
```

**LLM 调用方责任**：
- 调用 `lightweight_llm.ainvoke()` 时必须传 `timeout=NORMALIZE_TIMEOUT`
- 调用方需捕获 `TimeoutError` 和 `RateLimitError` 并走回退
- LLM 返回后统一走 `_parse_llm_keyword()` 解析（strip + 提取第一行 + 去引号）

### 4.8 sanitize.py

```python
MAX_KEYWORD_LEN = 64
MIN_KEYWORD_LEN = 3
KEYWORD_PATTERN = re.compile(r'[^a-z0-9\s\-_]')


def sanitize_keyword(raw: str) -> str:
    """
    截断到 MAX_KEYWORD_LEN → 小写 → 正则白名单过滤 → 合并空白符。
    结果长度 < MIN_KEYWORD_LEN 时返回空字符串。
    """

def contains_query_entities(keyword: str, query: str) -> bool:
    """
    检查 keyword 是否包含 query 中的专有名词/数字/年份。
    用简单的 token overlap 检测（不引入 NER 依赖）。
    """
```

### 4.9 keyword_index.py

```python
KW_INDEX_SYNC_INTERVAL = 5          # 秒


class KeywordIndexManager:
    """
    各 API 副本本地维护的 keyword embedding 缓存。

    生命周期：
      start()  → 全量加载 + 启动定时同步
      search() → 本地 KNN（numpy 暴力计算余弦相似度）
      on_keyword_written() → 写入副本立即更新本地
      stop()   → 取消定时任务
    """

    def __init__(self, redis: Redis): ...

    async def start(self): ...
    async def stop(self): ...

    async def search(
        self, query_vec: np.ndarray, top_k: int = 5
    ) -> list[tuple[str, float]]:
        """本地 KNN。只搜索 model_ver 匹配的 keyword。"""

    async def on_keyword_written(self, keyword: str):
        """template_gen 写入 keyword 后立即更新本地缓存。"""

    # 内部
    async def _full_reload(self): ...
    async def _periodic_sync(self): ...
```

**Redis 侧配合**（template_gen 调用方负责）：
- `HSET apc:kw_meta:<keyword> embedding <bytes> model_ver <str> dim <int> created_at <float>`
- `ZADD apc:kw_timeline <timestamp> <keyword>`

### 4.10 cache/lookup.py

```python
async def cache_lookup(state: AgentState) -> AgentState:
    """
    合并后的缓存查找节点（原 keyword_node + cache_lookup_node）。

    流程：
      1. L1  精确匹配
      2. kw_cache 查关键词缓存
      3. 向量归一化（embed → 候选短路 → LLM 归一化）
      4. L2  关键词索引查找
      5. L3  ctx_fingerprint 筛选

    返回 state，设置 cache_hit / cache_hit_layer / keyword / plan_template。
    """

# 内部辅助
def _build_l1_key(state: AgentState) -> str: ...
def _build_task_sig(state: AgentState) -> str: ...
async def _load_and_validate_template(tpl_id: str, tools_hash: str) -> dict | None: ...
def _hit(state: AgentState, tpl: dict, layer: str) -> AgentState: ...
```

### 4.11 cache/template_gen.py

```python
async def template_gen(state: AgentState) -> AgentState:
    """
    模板生成 + 写入所有索引层。

    流程：
      1. rule_filter + llm_filter 生成泛化模板
      2. 写入模板本体（apc:tpl:<tpl_id>）
      3. 写入 L2 索引（apc:tpl_idx:<agent_id>:<keyword>）
      4. 维护 keyword_index（向量索引 + timeline）
      5. 写入 L1 键（提升当前请求的路径到精确匹配）
      6. 维护反向索引（apc:tpl_refs:<tpl_id>）
      7. 并发写入保护（分布式锁 + 近重复检测）
      8. 异步持久化决策日志
    """

# 内部辅助
def _generate_tpl_id() -> str: ...
async def _maintain_keyword_index(keyword: str): ...
async def _upsert_keyword_with_lock(keyword: str, embedding: np.ndarray): ...
async def _write_l1_promotion(state: AgentState, tpl_id: str): ...
```

### 4.12 cache/eviction.py

```python
async def evict_template(tpl_id: str):
    """
    级联驱逐模板。

    1. 通过 apc:tpl_refs:<tpl_id> 查找所有 L1 键 → 批量 DELETE
    2. 遍历所有 apc:tpl_idx:* 移除该 tpl_id
    3. 清理空的 keyword 索引项（tpl_idx + kw_meta + kw_timeline + kw_tombstone）
    4. 删除模板本体和反向索引
    """

async def enforce_lru_eviction(max_size: int = 100):
    """
    Lua 脚本实现的原子 LRU 驱逐（基于 apc:kw_timeline）。
    当模板总数超过 max_size 时触发。
    """
```

---

## 5. 配置清单

```python
# config.py

@dataclass
class APCConfig:
    # ── 缓存容量 ────────────────────────────
    cache_max_size: int = 100               # 全局模板数上限
    max_tpl_per_keyword: int = 32           # 单 keyword 下模板数上限
    max_candidates_to_check: int = 10       # L3 遍历上限

    # ── kw_cache ────────────────────────────
    kw_cache_ttl_min: int = 300             # 最小 TTL（5min）
    kw_cache_ttl_max: int = 3600            # 最大 TTL（1h）
    drift_sample_rate: float = 0.05         # 漂移采样比例
    drift_high_threshold: float = 0.15      # 漂移率 > 此值 → TTL 降至 min
    drift_low_threshold: float = 0.05       # 漂移率 < 此值 → TTL 升至 max

    # ── 向量归一化 ──────────────────────────
    embed_model_name: str = "all-MiniLM-L6-v2"
    embed_model_ver: str = "v1"
    embed_dim: int = 384
    embed_max_workers: int = 2
    embed_max_concurrent: int = 4
    candidate_top_k: int = 5
    candidate_dist_high: float = 0.85       # 强复用阈值
    candidate_dist_low: float = 0.50        # 强新建阈值
    candidate_dist_fallback: float = 0.70   # LLM 失败时兜底复用阈值

    # ── LLM 归一化 ──────────────────────────
    normalize_llm_timeout: float = 2.0      # 超时（秒）
    max_retry_on_entity: int = 2            # 反实体注入最大重试次数
    max_keyword_len: int = 64
    min_keyword_len: int = 3

    # ── 多副本同步 ──────────────────────────
    kw_index_sync_interval: int = 5         # 增量同步间隔（秒）

    # ── 决策日志 ────────────────────────────
    decision_log_enabled: bool = True
    decision_log_batch_size: int = 50
    decision_log_flush_interval: float = 5.0  # 批量写入间隔（秒）

    # ── 冷启动 ──────────────────────────────
    snapshot_enabled: bool = False
    snapshot_interval: int = 3600           # 快照间隔（秒）
    snapshot_bucket: str = ""               # S3 bucket
    snapshot_prefix: str = "apc/kw_index/"


# 环境变量覆盖
def load_config() -> APCConfig:
    """从环境变量加载配置，覆盖默认值。"""
```

---

## 6. 关键实现决策

### 6.1 keyword 写入时 LLM 的调用归属

在 cache miss 路径中，large_planner 生成 plan 时一并产出 keyword 标注，不需要为 keyword 单独调用一次 LLM：

```
cache miss → large_planner 被调用（无论如何都要走）
  → large_planner 输出: { "keyword": "working_capital_ratio", "plan": ..., "answer": ... }
  → template_gen 拿到 keyword，决定复用还是新建
```

这消除了论文方案中 keyword extraction 占总成本 0.12% 的独立开销。对于命中路径中 kw_cache miss 调用的归一化 LLM（`normalize_keyword`），因为是 constrained 的决策任务而非 open-ended 生成，延迟通常比论文的 keyword extraction 更快（~100-200ms vs ~300-500ms，因为输出 token 更少）。

### 6.2 本地 KNN 的性能边界

keyword < 2000 时，384 维 float32 向量 × 2000 = ~3MB，numpy 暴力计算余弦相似度 < 1ms。当 keyword > 5000 时考虑切换 Redis Stack VECTOR。切换触发条件：`kw_index_size` 指标 > 5000 且 P99 搜索延迟 > 5ms。

### 6.3 L1 TTL 与模板 TTL 的关系

L1 键 TTL（24h）> 模板 TTL（根据 query 类型 1h-24h）。当模板因 TTL 过期被驱逐时，级联删除 L1 键。当 L1 键先于模板过期时（正常），下次同 query+ctx 请求走 L2/L3 命中后重新提升到 L1。

### 6.4 漂移采样的异步安全

`_check_drift()` 是 fire-and-forget 异步任务（`asyncio.create_task`），不阻塞主路径。采样率 5% 意味着每 20 次 kw_cache 命中触发 1 次旁路验证。验证失败只更新 Prometheus 计数器和内部滑动窗口，不修改缓存。

---

## 7. LangGraph 图组装（变更部分）

```python
def build_apc_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # 注册节点
    g.add_node("cache_lookup",  cache_lookup)          # ★ 合并后的节点
    g.add_node("small_planner", small_planner_node)     # 不变
    g.add_node("large_planner", large_planner_node)     # 不变
    g.add_node("actor",         actor_node)             # 不变
    g.add_node("template_gen",  template_gen)           # ★ 变更：新的写入逻辑

    g.set_entry_point("cache_lookup")                   # ★ 入口从 keyword_node 改为 cache_lookup

    # 路由
    g.add_conditional_edges("cache_lookup",
        lambda s: "small_planner" if s["cache_hit"] else "large_planner")

    g.add_conditional_edges("small_planner",
        lambda s: END if s["is_complete"] else "actor")

    g.add_conditional_edges("large_planner",
        lambda s: "template_gen" if s["is_complete"] else "actor")

    g.add_conditional_edges("actor",
        lambda s: "small_planner" if s["cache_hit"] else "large_planner")

    g.add_edge("template_gen", END)

    return g.compile(checkpointer=...)
```

State 变更：请求入口处需填充 `agent_id`、`tools`、`tools_hash` 三个新字段。FastAPI 层从请求中提取这些信息后传入 initial_state。

---

## 8. 测试策略

### 8.1 单元测试

| 模块 | 测试重点 | 关键 case |
|------|---------|----------|
| `normalize` | 幂等性、空白符处理、Unicode | 全角→半角、组合字符、空字符串 |
| `fingerprint` | ctx 分类准确度、schema 提取深度 | 嵌套 dict、空 context、表格/文本分类 |
| `candidates` | 短路逻辑正确性 | 空索引、单候选高置信、全低置信、边界 0.85/0.50 |
| `sanitize` | 边界 case | 超长、纯数字、纯标点、中英混合、SQL 注入字符 |
| `llm_normalize` | 回退链触发 | mock LLM 超时、mock LLM 返回非法格式、mock LLM 返回不在候选列表的字符串 |
| `eviction` | 级联清理完整性 | 有 L1 引用的模板、空 keyword 索引清理 |

### 8.2 集成测试

```python
@pytest.mark.integration
class TestCacheLookup:
    async def test_l1_exact_hit_second_request(self, redis, embed_model):
        """相同 query + ctx 发送两次，第二次 L1 命中。"""

    async def test_kw_cache_hit_skips_llm(self, redis, embed_model, mock_llm):
        """kw_cache 命中时不调用 LLM。"""

    async def test_shortcut_reuse_skips_llm(self, redis, embed_model, mock_llm):
        """候选 cosine > 0.85 且优势明显时直接复用。"""

    async def test_shortcut_new_skips_llm(self, redis, embed_model, mock_llm):
        """所有候选 < 0.50 时直接走生成回退。"""

    async def test_llm_normalize_with_candidates(self, redis, embed_model):
        """模糊区调用 LLM 做复用/新建决策。"""

    async def test_fallback_on_llm_timeout(self, redis, embed_model, mock_llm):
        """LLM 超时走回退链。"""

    async def test_ctx_mismatch_falls_through(self, redis, embed_model):
        """同 keyword 但不同 ctx 的请求不会错误命中。"""

    async def test_l1_promotion_and_cascade_eviction(self, redis, embed_model):
        """L2/L3 命中 → 提升到 L1 → 驱逐模板 → L1 键被级联清理。"""


@pytest.mark.integration
class TestKeywordIndex:
    async def test_incremental_sync_cross_replicas(self, redis):
        """两副本场景：副本 A 写入新 keyword，副本 B 在 sync_interval 内感知。"""

    async def test_concurrent_write_dedup(self, redis):
        """并发写入相同 keyword 不产生重复。"""


@pytest.mark.integration
class TestEndToEnd:
    async def test_hit_miss_consistency(self, redis, embed_model, graph):
        """同一个 query 的 miss 路径和后续 hit 路径产出结果语义一致。"""

    async def test_keyword_convergence(self, redis, embed_model):
        """语义相似的 query 逐步收敛到同一个 keyword。"""
```

### 8.3 性能基准测试

```python
@pytest.mark.benchmark
class TestBenchmarks:
    async def test_l1_latency_under_1ms(self, benchmark): ...

    async def test_knn_under_5ms_with_2000_keywords(self, benchmark): ...

    async def test_full_miss_path_under_baseline(self, benchmark): ...
```

---

## 9. 上线阶段

| 阶段 | 内容 | 灰度方式 | 验收标准 |
|------|------|---------|---------|
| **Phase 1** | P0 模块（L1 + L2 + L3 + ctx_fingerprint） | 10% 流量 | 无串数据、命中率 ≥ v1 基线 |
| **Phase 2** | P1 模块（向量归一化 + kw_cache + 回退链 + 指标） | 50% 流量 | 归一化回退率 < 5%、漂移率可见 |
| **Phase 3** | P2 模块（多副本同步 + 级联驱逐 + 冷启动 + 决策日志） | 全量 | 多副本一致性测试通过 |
| **Phase 4** | P3 模块（别名 CLI + reindex + 多语言） | 按需 | 运维工具可用 |

每个阶段独立部署和回滚，feature flag 控制是否启用向量归一化（关闭时退化为 v2 的纯 LLM 抽取 + L1/L2/L3 三层）。

---

## 10. 依赖清单

```toml
# pyproject.toml 新增依赖
[project]
dependencies = [
    # 现有（不变）
    "langgraph",
    "redis",
    "fastapi",
    "prometheus-client",

    # 新增
    "sentence-transformers>=3.0",    # embedding 模型
    "pgvector>=0.3",                 # 决策日志存储（P2，可选）
    "asyncpg",                       # PostgreSQL 异步驱动（P2，可选）
]
```

无新增基础设施依赖（不需要 Redis Stack、不需要独立向量数据库）。
