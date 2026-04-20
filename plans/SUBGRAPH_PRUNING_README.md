# GraphRAG 子图预裁剪优化实现说明

> **毕业设计项目**：GraphRAG Token消耗优化 - 方案二实现  
> **实现日期**：2026年4月  
> **版本**：v1.0

---

## 📋 目录

1. [项目概述](#项目概述)
2. [实现架构](#实现架构)
3. [核心模块说明](#核心模块说明)
4. [使用指南](#使用指南)
5. [配置说明](#配置说明)
6. [消融实验指南](#消融实验指南)
7. [性能评估](#性能评估)
8. [故障排查](#故障排查)

---

## 项目概述

### 优化目标

在GraphRAG的检索增强生成阶段，对召回的子图进行智能裁剪，通过以下策略减少Token消耗：

1. **关联性评分**：使用轻量级模型（Sentence Transformer）对实体和关系进行相关性打分
2. **多层级裁剪**：按相关性分层，优先保留高相关性元素
3. **图结构感知**：考虑图拓扑结构，保留关键路径和桥接节点

### 预期效果

| 指标 | 目标 |
|------|------|
| Token减少率 | 30-60% |
| 答案质量保持 | >95% |
| 延迟增加 | <10% |

---

## 实现架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户查询 (Query)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   LocalSearch                                │
│                 (查询处理入口)                                │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│           LocalSearchMixedContext                            │
│      (实体映射后统一应用子图裁剪并构建上下文)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              apply_subgraph_pruning()                        │
│              (裁剪实体候选集与候选关系)                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              _build_local_context()                          │
│         (使用裁剪后的实体/关系构建局部上下文)                  │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ Relevance   │ │ MultiLevel  │ │ GraphAware  │
│ Scorer      │ │ Pruner      │ │ Pruner      │
│ (评分器)     │ │ (多层级)     │ │ (图感知)     │
└─────────────┘ └─────────────┘ └─────────────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              裁剪后的实体和关系                               │
│              (Pruned Entities & Relationships)               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              构建最终上下文                                   │
│              (Build Final Context)                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM生成答案                                      │
│              (Generate Answer)                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心模块说明

### 1. RelevanceScorer (关联性评分器)

**文件位置**：[`graphrag/query/context_builder/relevance_scorer.py`](../graphrag/query/context_builder/relevance_scorer.py)

**功能**：
- 使用Sentence Transformer计算查询与实体/关系的语义相似度
- 支持多种评分方法：embedding、hybrid、simple
- 返回评分后的实体和关系列表

**核心算法**：
```python
# Embedding相似度
similarity = cosine_similarity(query_embedding, entity_embedding)

# 混合评分
score = 0.6 * embedding_sim + 0.2 * text_match + 0.2 * importance
```

**使用示例**：
```python
from graphrag.query.context_builder.relevance_scorer import RelevanceScorer

scorer = RelevanceScorer(
    scoring_method="embedding",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

scored_entities = scorer.score_entities(query, entities)
scored_relationships = scorer.score_relationships(query, relationships)
```

### 2. MultiLevelPruner (多层级裁剪器)

**文件位置**：[`graphrag/query/context_builder/subgraph_pruner.py`](../graphrag/query/context_builder/subgraph_pruner.py)

**功能**：
- 将实体/关系按相关性分为高/中/低三层
- 优先保留高相关性元素
- 在Token预算允许时添加中等相关性元素

**核心算法**：
```python
# 分层
high_entities = [e for e, s in scored if s >= 0.7]
medium_entities = [e for e, s in scored if 0.5 <= s < 0.7]

# 贪心选择
for entity in high_entities + medium_entities:
    if current_tokens + T(entity) <= max_tokens:
        selected.append(entity)
```

**使用示例**：
```python
from graphrag.query.context_builder.subgraph_pruner import MultiLevelPruner

pruner = MultiLevelPruner(
    high_relevance_threshold=0.7,
    medium_relevance_threshold=0.5
)

result = pruner.prune_subgraph(
    scored_entities,
    scored_relationships,
    max_context_tokens,
    tokenizer
)
```

### 3. GraphAwarePruner (图结构感知裁剪器)

**文件位置**：[`graphrag/query/context_builder/subgraph_pruner.py`](../graphrag/query/context_builder/subgraph_pruner.py)

**功能**：
- 计算节点的PageRank和Betweenness Centrality
- 保留与查询实体距离近的节点（k-hop邻居）
- 综合距离和重要性进行裁剪

**核心算法**：
```python
# 距离得分
distance_score = 1 / (1 + shortest_path_distance)

# 重要性得分
importance_score = 0.5 * PageRank + 0.5 * Betweenness

# 综合得分
final_score = 0.6 * distance_score + 0.4 * importance_score
```

**使用示例**：
```python
from graphrag.query/context_builder/subgraph_pruner import GraphAwarePruner

pruner = GraphAwarePruner(
    max_hops=2,
    importance_weight=0.5
)

result = pruner.prune_subgraph(
    entities,
    relationships,
    query_entities,
    max_context_tokens,
    tokenizer
)
```

### 4. 配置扩展

**文件位置**：[`graphrag/config/models/local_search_config.py`](../graphrag/config/models/local_search_config.py)

**新增配置项**：
```python
class LocalSearchConfig(BaseModel):
    # ... 原有配置 ...
    
    # 子图裁剪配置
    enable_subgraph_pruning: bool = False
    pruning_strategy: str = "multi_level"
    relevance_scoring_method: str = "embedding"
    embedding_model_for_scoring: str = "sentence-transformers/all-MiniLM-L6-v2"
    high_relevance_threshold: float = 0.7
    medium_relevance_threshold: float = 0.5
    max_hops_for_graph_pruning: int = 2
    graph_pruning_importance_weight: float = 0.5
    graph_pruning_distance_weight: float = 0.6
    hybrid_min_reduction_rate: float = 0.2
    pruning_token_buffer_ratio: float = 0.9
```

---

## 使用指南

### 快速开始

#### 1. 安装依赖

```bash
# 安装sentence-transformers（用于关联性评分）
pip install sentence-transformers

# 或者使用项目的requirements
pip install -r requirements.txt
```

#### 2. 配置启用

在`settings.yaml`中添加配置：

```yaml
local_search:
  enable_subgraph_pruning: true
  pruning_strategy: "multi_level"
  relevance_scoring_method: "embedding"
  high_relevance_threshold: 0.7
  medium_relevance_threshold: 0.5
```

#### 3. 运行查询

```python
from graphrag.query import LocalSearch

# 创建搜索引擎（通过query factory创建时会自动读取配置）
search_engine = LocalSearch(
    model=chat_model,
    context_builder=context_builder,
    tokenizer=tokenizer
)

# 执行查询
result = await search_engine.search(
    query="What are the main themes in the documents?"
)

print(f"Answer: {result.response}")
print(f"Context tokens: {result.context_tokens}")
```

### Python API使用

```python
from graphrag.query.context_builder.pruning_context_builder import apply_subgraph_pruning

# 手动调用裁剪
pruned_entities, pruned_relationships, pruning_result = apply_subgraph_pruning(
    query="your query here",
    selected_entities=entities,
    relationships=relationships,
    max_context_tokens=8000,
    tokenizer=tokenizer,
    enable_pruning=True,
    pruning_strategy="multi_level",
    relevance_scoring_method="embedding",
    high_relevance_threshold=0.7,
    medium_relevance_threshold=0.5
)

# 查看裁剪效果
if pruning_result:
    print(f"Token reduction: {pruning_result.token_reduction_rate:.1%}")
    print(f"Entity retention: {pruning_result.entity_retention_rate:.1%}")
    print(f"Tokens saved: {pruning_result.tokens_saved}")
```

---

## 配置说明

### 完整配置示例

参见：[`examples_notebooks/subgraph_pruning_config_example.yaml`](../examples_notebooks/subgraph_pruning_config_example.yaml)

说明：示例已经按当前 GraphRAG 配置体系使用 `models:`、
`local_search.chat_model_id` 和 `local_search.embedding_model_id` 组织，可直接作为消融实验配置模板继续扩展。

### 关键参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enable_subgraph_pruning` | bool | false | 是否启用子图裁剪 |
| `pruning_strategy` | str | "multi_level" | 裁剪策略：multi_level/graph_aware/hybrid |
| `relevance_scoring_method` | str | "embedding" | 评分方法：embedding/hybrid/simple |
| `embedding_model_for_scoring` | str | "all-MiniLM-L6-v2" | HuggingFace模型名称 |
| `high_relevance_threshold` | float | 0.7 | 高相关性阈值 (0-1) |
| `medium_relevance_threshold` | float | 0.5 | 中等相关性阈值 (0-1) |
| `max_hops_for_graph_pruning` | int | 2 | 图裁剪的最大跳数 |
| `graph_pruning_importance_weight` | float | 0.5 | PageRank与Betweenness的权重 |
| `graph_pruning_distance_weight` | float | 0.6 | 距离分数与重要性分数的权重 |
| `hybrid_min_reduction_rate` | float | 0.2 | Hybrid策略触发第二阶段的最小Token减少率 |
| `pruning_token_buffer_ratio` | float | 0.9 | Token预算使用比例 (0-1) |

### 推荐配置

#### 场景1：精确查询（已知实体）

```yaml
pruning_strategy: "graph_aware"
max_hops_for_graph_pruning: 1
high_relevance_threshold: 0.8
```

#### 场景2：探索性查询（模糊概念）

```yaml
pruning_strategy: "multi_level"
high_relevance_threshold: 0.6
medium_relevance_threshold: 0.4
```

#### 场景3：复杂多跳查询

```yaml
pruning_strategy: "hybrid"
max_hops_for_graph_pruning: 3
high_relevance_threshold: 0.7
```

---

## 消融实验指南

### 实验设计

为了评估各个组件的贡献，建议进行以下消融实验：

#### 实验1：Baseline（无裁剪）

```yaml
enable_subgraph_pruning: false
```

#### 实验2：仅多层级裁剪

```yaml
enable_subgraph_pruning: true
pruning_strategy: "multi_level"
relevance_scoring_method: "embedding"
```

#### 实验3：仅图结构裁剪

```yaml
enable_subgraph_pruning: true
pruning_strategy: "graph_aware"
max_hops_for_graph_pruning: 2
```

#### 实验4：混合策略

```yaml
enable_subgraph_pruning: true
pruning_strategy: "hybrid"
```

#### 实验5：不同评分方法

```yaml
# 5a: 纯Embedding
relevance_scoring_method: "embedding"

# 5b: 混合评分
relevance_scoring_method: "hybrid"

# 5c: 简单文本匹配
relevance_scoring_method: "simple"
```

#### 实验6：不同阈值

```yaml
# 6a: 宽松阈值
high_relevance_threshold: 0.6
medium_relevance_threshold: 0.4

# 6b: 默认阈值
high_relevance_threshold: 0.7
medium_relevance_threshold: 0.5

# 6c: 严格阈值
high_relevance_threshold: 0.8
medium_relevance_threshold: 0.6
```

### 评估指标

#### 1. Token消耗指标

```python
# Token减少率
token_reduction_rate = (baseline_tokens - optimized_tokens) / baseline_tokens

# 绝对Token节省
tokens_saved = baseline_tokens - optimized_tokens
```

#### 2. 信息保留指标

```python
# 实体保留率
entity_retention = len(pruned_entities) / len(original_entities)

# 关系保留率
relationship_retention = len(pruned_relationships) / len(original_relationships)
```

#### 3. 答案质量指标

```python
# ROUGE-L分数
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(optimized_answer, baseline_answer)

# 人工评分（1-5分）
quality_score = human_evaluation(optimized_answer)
```

#### 4. 性能指标

```python
# 查询延迟
latency = end_time - start_time

# 延迟增加率
latency_increase = (optimized_latency - baseline_latency) / baseline_latency
```

### 实验脚本示例

```python
import asyncio
import pandas as pd
from graphrag.query import LocalSearch

async def run_ablation_experiment(config_variants, queries, output_file):
    """运行消融实验"""
    results = []
    
    for config_name, config in config_variants.items():
        print(f"\n{'='*60}")
        print(f"Running: {config_name}")
        print(f"{'='*60}")
        
        # 创建搜索引擎
        search_engine = LocalSearch(
            model=chat_model,
            context_builder=context_builder,
            **config
        )
        
        for query in queries:
            result = await search_engine.search(query=query)
            
            results.append({
                "config": config_name,
                "query": query,
                "answer": result.response,
                "context_tokens": result.context_tokens,
                "generation_tokens": result.generation_tokens,
                "total_tokens": result.context_tokens + result.generation_tokens,
                "latency": result.latency
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

# 定义配置变体
config_variants = {
    "baseline": {"enable_subgraph_pruning": False},
    "multi_level": {
        "enable_subgraph_pruning": True,
        "pruning_strategy": "multi_level"
    },
    "graph_aware": {
        "enable_subgraph_pruning": True,
        "pruning_strategy": "graph_aware"
    },
    "hybrid": {
        "enable_subgraph_pruning": True,
        "pruning_strategy": "hybrid"
    }
}

# 定义测试查询
queries = [
    "What are the main organizations mentioned?",
    "Describe the relationships between key people.",
    "What events occurred and when?",
    "Summarize the main themes."
]

# 运行实验
asyncio.run(run_ablation_experiment(
    config_variants,
    queries,
    "ablation_results.csv"
))
```

### 运行时统计

启用子图裁剪后，`LocalSearch` 返回结果中的 `context_data` 会额外包含一张
`subgraph_pruning` 表，可用于记录：

- 原始/裁剪后实体数量
- 原始/裁剪后关系数量
- Token裁剪前后数量
- Token减少率
- 当前裁剪策略及其附加元数据

这张表适合直接接入消融实验脚本进行统计汇总。

---

## 性能评估

### 预期性能指标

| 指标 | Baseline | Multi-Level | Graph-Aware | Hybrid |
|------|----------|-------------|-------------|--------|
| Token减少率 | 0% | 35% | 40% | 45% |
| 实体保留率 | 100% | 45% | 50% | 48% |
| 答案质量 | 100% | 96% | 97% | 97% |
| 延迟增加 | 0ms | +50ms | +80ms | +100ms |

### 成本节省估算

假设：
- 每次查询平均Token：15,000
- GPT-4o-mini价格：$0.15/1M input tokens
- 每天查询量：1,000次

**月度成本对比**：

| 方案 | 每次Token | 月度Token | 月度成本 | 节省 |
|------|-----------|-----------|----------|------|
| Baseline | 15,000 | 450M | $67.50 | - |
| 优化后 (40%减少) | 9,000 | 270M | $40.50 | $27.00 (40%) |

---

## 故障排查

### 常见问题

#### 1. ImportError: No module named 'sentence_transformers'

**解决方案**：
```bash
pip install sentence-transformers
```

#### 2. 裁剪后答案质量下降明显

**可能原因**：
- 阈值设置过高，过滤了重要信息
- 评分模型不适合当前领域

**解决方案**：
```yaml
# 降低阈值
high_relevance_threshold: 0.6
medium_relevance_threshold: 0.4

# 或切换到混合评分
relevance_scoring_method: "hybrid"
```

#### 3. 延迟增加过多

**可能原因**：
- Embedding模型过大
- 图结构计算复杂度高

**解决方案**：
```yaml
# 使用更轻量的模型
embedding_model_for_scoring: "sentence-transformers/all-MiniLM-L6-v2"

# 或切换到多层级策略
pruning_strategy: "multi_level"
```

#### 4. Token减少不明显

**可能原因**：
- 原始候选集本身就很小
- 阈值设置过低

**解决方案**：
```yaml
# 提高阈值
high_relevance_threshold: 0.8
medium_relevance_threshold: 0.6

# 或使用更激进的策略
pruning_token_buffer_ratio: 0.8
```

### 调试技巧

#### 启用详细日志

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphrag.query.context_builder")
logger.setLevel(logging.DEBUG)
```

#### 查看裁剪统计

```python
# 在代码中添加
if pruning_result:
    print(f"Original: {pruning_result.original_entity_count} entities")
    print(f"Pruned: {len(pruning_result.pruned_entities)} entities")
    print(f"Retention: {pruning_result.entity_retention_rate:.1%}")
    print(f"Token saved: {pruning_result.tokens_saved}")
```

---

## 文档索引

- **技术方案**：[`plans/graphrag_token_optimization_plan.md`](graphrag_token_optimization_plan.md)
- **数学原理**：[`plans/subgraph_pruning_mathematical_principles.md`](subgraph_pruning_mathematical_principles.md)
- **配置示例**：[`examples_notebooks/subgraph_pruning_config_example.yaml`](../examples_notebooks/subgraph_pruning_config_example.yaml)
- **源代码**：
  - [`graphrag/query/context_builder/relevance_scorer.py`](../graphrag/query/context_builder/relevance_scorer.py)
  - [`graphrag/query/context_builder/subgraph_pruner.py`](../graphrag/query/context_builder/subgraph_pruner.py)
  - [`graphrag/query/context_builder/pruning_context_builder.py`](../graphrag/query/context_builder/pruning_context_builder.py)
  - [`graphrag/query/structured_search/local_search/mixed_context.py`](../graphrag/query/structured_search/local_search/mixed_context.py)
  - [`graphrag/query/factory.py`](../graphrag/query/factory.py)
  - [`graphrag/config/models/local_search_config.py`](../graphrag/config/models/local_search_config.py)

---

## 致谢

本实现基于Microsoft GraphRAG框架，感谢原作者团队的开源贡献。

---

**最后更新**：2026年4月3日
