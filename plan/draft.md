##keyword设计的参考思路
#方案1
方案解释
这个设计本质上是在 APC 的单层关键词匹配之上，加了两个正交的确定性约束层，形成三层漏斗结构。
Layer 1（task_sig） 是一个纯哈希操作，不涉及任何语义理解。它把 normalize(query) + agent_id + tool_set_hash 做 SHA256，截取前 16 位。这一层解决的是一个论文方案完全没有处理的场景：完全相同的请求被反复发送。在生产环境中这种情况很常见，比如定时任务、用户刷新页面、同一个 pipeline 的批量调用。论文方案对这类请求仍然要走 keyword 抽取（一次 LLM 调用），而 task_sig 层让它 0 LLM 调用命中。
normalize(query) 的工作是处理空白符、大小写、标点等表面变化，使得"同一个问题的不同格式写法"映射到同一个签名。agent_id 和 tool_set_hash 的加入是一个关键设计决策：同一个 query，在不同 agent 角色或不同工具集环境下，不应该复用同一个计划模板。这是论文原始方案里完全缺失的约束。
Layer 2（keyword） 基本保留了论文的核心设计，但加了一个重要优化：keyword 抽取结果本身用 SETEX apc:kw_cache:<query_hash> 3600 缓存一小时。这解决了一个实际问题——同一个 query 在第二次到来时，论文方案仍然要调用 LLM 抽取 keyword（只是 Redis lookup 命中了，但 keyword 抽取这步没有被省掉）。加了这层缓存之后，第二次请求可以直接拿到上次的 keyword，完全不走 LLM。
这里的 apc:keyword_index 存的是 keyword → list of template_ids，而不是论文的 keyword → template（一对一）。这是一个结构性改进：允许同一个关键词对应多个模板，而不是只保留最新的那个。
Layer 3（ctx_fingerprint） 是这个方案里最有创意也最需要认真审视的部分。它计算 context + tools_available + agent_role 的指纹，在 Layer 2 找到候选 template_ids 之后，用这个指纹做二次筛选。背后的逻辑是：同一类任务（keyword 相同），但上下文结构不同，应该匹配不同的模板。比如"working capital ratio"这个 keyword，在有 10K token 财务文档的场景和只有表格数据的场景下，最优计划步骤可能不一样。
筛选方式是 Jaccard 相似度或精确比较，在候选模板集内部做，而不是在全量缓存里做——这是关键，保证了这步不会变成全局向量搜索。

Review
整体判断：方案方向正确，但有两个设计决策需要认真讨论。
真正的改进点
task_sig 层的引入是实质性进步，不是论文方案的修补，而是补充了一个完全不同的命中路径。论文关注的是"语义相似的不同 query"，而 task_sig 处理的是"字面相同或近似相同的 query"，这两个问题在生产环境里同等重要。
keyword 结果的二级缓存（apc:kw_cache）是一个工程上容易被忽略但实际收益明显的优化。论文方案每次都要调用 LLM 做 keyword 抽取，这个开销论文里算作 0.12% 的成本，但在高并发下这个数字会显著累积。
一对多的 keyword → [template_ids] 结构为 ctx_fingerprint 筛选创造了前提，这个结构比论文的一对一更合理。
需要讨论的问题
第一个问题：ctx_fingerprint 的计算方式决定了这层是否有效。
方案写的是 compute_fingerprint(context, tools_available, agent_role)，但"context"在 agent 场景里通常是一个动态文档（财务报告、网页内容、数据库查询结果）。如果直接对 context 内容做哈希，那么不同公司的财务报告会产生完全不同的 fingerprint，导致 ctx_fingerprint 层几乎永远无法匹配——因为计划模板的可复用性恰恰来自于"忽略具体数值和实体，保留操作结构"这个抽象，而 context 的原始内容恰恰包含了所有这些具体细节。
所以 compute_fingerprint 必须是结构化的，而不是内容哈希。比如：
pythondef compute_fingerprint(context, tools_available, agent_role):
    return {
        "context_type": classify_context_type(context),  
        # "financial_doc" / "tabular_data" / "web_page"
        "context_length_bucket": bucket(len(context)),   
        # "short" / "medium" / "long"
        "tools": frozenset(tools_available),
        "agent_role": agent_role,
    }
如果 compute_fingerprint 没有做这层结构化抽象，ctx_fingerprint 层的命中率会接近零，退化成只有 task_sig 和 keyword 在工作。
第二个问题：task_sig 里的 normalize(query) 边界在哪里。
normalize 处理空白和大小写是合理的，但"FY2019 working capital for Costco"和"FY2021 working capital for Amazon"经过 normalize 之后仍然是两个不同的签名，这是正确行为。但如果有人想让 normalize 更激进（比如把数字替换成占位符），那就和 keyword 层的工作重叠了，而且做不好的话会产生错误的 task_sig 命中。方案里应该明确 normalize 的边界：只做无损的表面标准化，不做语义归一化，后者是 keyword 层的职责。
和我方案的差异总结
维度你的方案我的方案处理"完全相同 query"task_sig 层，0 LLM 调用没有专门处理keyword 抽取成本二级缓存，重复请求不调 LLM每次都调用 lightweight LLM向量的角色完全没有向量向量辅助 keyword 归一化上下文约束ctx_fingerprint 显式过滤没有上下文层keyword 与模板的关系一对多一对一假阳性防护ctx_fingerprint 二次筛选依赖精确匹配的固有属性实现复杂度中高（三层+指纹设计）低（单层精确匹配）
你的方案相比我的方案，核心优势是在不引入向量搜索的前提下，通过纯确定性手段同时改善了命中率（task_sig 补充了一类命中场景）和精确性（ctx_fingerprint 过滤了错误复用）。我的方案引入向量是为了解决 keyword 漂移问题，但正如你之前指出的，向量用于命中决策本身就会带入过预测问题。你的方案绕开了这个问题，是在正确的思路上做的。
ctx_fingerprint 的设计细节（如何做结构化抽象而不是内容哈希）是这个方案最需要具体化的地方，这直接决定第三层是否真正有效。

#总结
把 Layer 2 的 keyword 抽取从直接调用 LLM 生成 keyword，改为先用向量找已有 keyword 候选，再让 LLM 决定复用还是新建。

逐层分析影响
Layer 1（task_sig）完全不受影响。
task_sig 是纯哈希，和 keyword 如何生成没有任何依赖关系。这层独立工作。
Layer 2 本身：改动是有效的，但引入了一个新的内部依赖。
改动后的 keyword_node 变成：
pythonasync def keyword_node(state):
    # 先查 kw_cache（这部分保留）
    cached = await redis.get(f"apc:kw_cache:{hash(query)}")
    if cached:
        return cached

    # 向量找候选（新增）
    query_vec = await embed(query)
    neighbors = await keyword_index.search(query_vec, top_k=3)

    # LLM 做归一化决策（改动后）
    keyword = await llm.invoke(f"""
已有关键词候选：{neighbors}
当前任务：{query}
判断是否复用某个候选，或生成新关键词。只返回关键词。
""")
    await redis.setex(f"apc:kw_cache:{hash(query)}", 3600, keyword)
    return keyword
这个改动对 Layer 2 自身的效果是：keyword 漂移问题被大幅缓解，"working capital ratio"和"working capital calculation"有更大概率被归一到同一个 keyword 上。这是实质性改善。
但是，这里有一个你需要注意的新依赖：keyword_index 是什么？它必须是一个存储所有已有 keyword 的向量索引，不是 template 的索引，不是 query 的索引，只是 keyword 字符串本身的索引。这个索引需要在每次 cache miss 生成新 keyword 后同步更新：
python# cache miss 写入时，同时维护 keyword_index
await redis.hset("apc:keyword_index", keyword, template_ids)
await keyword_vec_index.upsert(keyword, await embed(keyword))
# keyword 数量通常 < 1000，向量索引规模极小，搜索不会有延迟问题
keyword 的数量远小于 query 的数量（论文实验里 100 个 query 只产生约 46 个唯一 keyword），所以这个向量索引规模极小，不会出现之前讨论的大规模 ANN 延迟问题。这是和"对 query 做向量匹配"的本质区别。
Layer 3（ctx_fingerprint）完全不受影响。
ctx_fingerprint 的输入是 keyword（用来查候选 template_ids）和 context（用来做二次筛选）。keyword 如何生成不影响 ctx_fingerprint 的逻辑，只要 keyword 最终是一个字符串，Layer 3 就正常工作。实际上改动后 keyword 的稳定性提高了，Layer 3 拿到的候选集质量更好。
apc:kw_cache 的 TTL 需要重新考虑。
原方案 SETEX 3600（一小时）的假设是 keyword 抽取是确定性的——同一个 query 每次调用 LLM 结果大概率相同，所以缓存一小时是合理的。改动后 keyword 的生成依赖于当时 keyword_index 里有哪些候选，而 keyword_index 会随着新 template 的写入不断增长。这意味着一小时前缓存的 keyword，可能不是现在 keyword_index 状态下 LLM 会选择的结果。
这不是一个严重问题，但需要意识到：kw_cache 的 TTL 越长，keyword 归一化的一致性越差（新候选进来了但旧 query 还在用旧 keyword）。一个务实的处理是把 TTL 缩短（比如 10-15 分钟），或者在写入新 template 时主动 invalidate 相关 kw_cache 条目——但后者实现复杂，通常不值得。

改动后的完整流程
请求到来
    │
    ▼
[Layer 1] task_sig 精确匹配
    ├─ HIT  → 直接返回 template（0 LLM 调用）
    └─ MISS ↓
    
[kw_cache] 查关键词缓存
    ├─ HIT  → 拿到 keyword（0 LLM 调用）
    └─ MISS ↓
    
[向量归一化] embed(query) → keyword_index.search(top_k=3)
    └─ 候选 keywords 喂给 lightweight LLM
    └─ LLM 决定复用还是新建 → keyword
    └─ 写入 kw_cache
    
[Layer 2] keyword 精确匹配 keyword_index
    ├─ HIT  → 拿到 candidate template_ids ↓
    └─ MISS → Large LM，执行后生成 template，写入所有层
    
[Layer 3] ctx_fingerprint 筛选 candidate template_ids
    ├─ 匹配 → Small LM 适配模板，执行
    └─ 不匹配 → Large LM（视为 miss）