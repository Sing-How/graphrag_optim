# 子图预裁剪优化方案 - 安装指南

## 快速安装

### 方法1：使用uv（推荐）

```bash
# 安装额外依赖
uv pip install -r requirements_subgraph_pruning.txt

# 或者只安装核心依赖
uv pip install sentence-transformers>=2.2.0
```

### 方法2：使用pip

```bash
# 安装额外依赖
pip install -r requirements_subgraph_pruning.txt

# 或者只安装核心依赖
pip install sentence-transformers>=2.2.0
```

## 依赖说明

### 核心依赖（必需）

| 包名 | 版本 | 大小 | 用途 |
|------|------|------|------|
| sentence-transformers | >=2.2.0 | ~10MB | 文本嵌入和相似度计算 |
| transformers | >=4.30.0 | ~500MB | Transformer模型支持 |
| torch | >=2.0.0 | ~800MB (CPU) | 深度学习框架 |

### 已有依赖（GraphRAG自带）

| 包名 | 版本 | 说明 |
|------|------|------|
| networkx | >=3.4.2 | 图算法库 |
| numpy | >=1.25.2 | 数值计算 |
| pandas | >=2.2.3 | 数据处理 |

### 可选依赖（推荐）

| 包名 | 版本 | 用途 |
|------|------|------|
| scipy | >=1.10.0 | 加速图算法 |
| scikit-learn | >=1.3.0 | 额外的相似度计算 |
| rouge-score | >=0.1.2 | 答案质量评估 |

## 安装验证

### 1. 验证sentence-transformers

```python
# 测试导入
from sentence_transformers import SentenceTransformer

# 加载模型（首次会自动下载）
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# 测试编码
embeddings = model.encode(["Hello world", "Test sentence"])
print(f"Embedding shape: {embeddings.shape}")
# 预期输出：Embedding shape: (2, 384)
```

### 2. 验证GraphRAG集成

```python
# 测试导入新模块
from graphrag.query.context_builder.relevance_scorer import RelevanceScorer
from graphrag.query.context_builder.subgraph_pruner import MultiLevelPruner, GraphAwarePruner
from graphrag.query.context_builder.pruning_context_builder import apply_subgraph_pruning

print("✅ All modules imported successfully!")
```

### 3. 完整功能测试

```python
from graphrag.query.context_builder.relevance_scorer import RelevanceScorer
from graphrag.data_model.entity import Entity

# 创建测试数据
entities = [
    Entity(id="1", title="Python", description="A programming language"),
    Entity(id="2", title="Java", description="Another programming language"),
    Entity(id="3", title="Apple", description="A fruit"),
]

# 创建评分器
scorer = RelevanceScorer(
    scoring_method="embedding",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# 执行评分
query = "What programming languages are mentioned?"
scored_entities = scorer.score_entities(query, entities)

# 打印结果
for entity, score in scored_entities:
    print(f"{entity.title}: {score:.3f}")

# 预期输出：
# Python: 0.7xx (高分)
# Java: 0.6xx (中等分)
# Apple: 0.2xx (低分)
```

## 常见问题

### Q1: torch安装失败或版本冲突

**解决方案**：

```bash
# 卸载现有torch
uv pip uninstall torch torchvision torchaudio

# 重新安装CPU版本
uv pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu

# 或GPU版本（CUDA 11.8）
uv pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

### Q2: 模型下载速度慢

**解决方案**：

```bash
# 方法1：使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法2：手动下载模型
# 访问：https://hf-mirror.com/sentence-transformers/all-MiniLM-L6-v2
# 下载后放到：~/.cache/huggingface/hub/
```

### Q3: ImportError: cannot import name 'SentenceTransformer'

**解决方案**：

```bash
# 确认安装
uv pip list | grep sentence-transformers

# 重新安装
uv pip install --force-reinstall sentence-transformers>=2.2.0
```

### Q4: CUDA out of memory（GPU版本）

**解决方案**：

```python
# 在代码中指定使用CPU
scorer = RelevanceScorer(
    scoring_method="embedding",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"  # 强制使用CPU
)
```

### Q5: 模型加载慢

**解决方案**：

```python
# 预加载模型（在应用启动时）
from sentence_transformers import SentenceTransformer

# 全局加载一次
_model_cache = {}

def get_model(model_name):
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]
```

## 系统要求

### 最低配置

- **CPU**: 2核心
- **内存**: 4GB RAM
- **磁盘**: 2GB 可用空间
- **Python**: 3.10-3.12

### 推荐配置

- **CPU**: 4核心+
- **内存**: 8GB+ RAM
- **磁盘**: 5GB+ 可用空间
- **GPU**: 可选，NVIDIA GPU with CUDA 11.8+

## 性能优化建议

### 1. 使用GPU加速（如果可用）

```python
scorer = RelevanceScorer(
    scoring_method="embedding",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    device="cuda"  # 使用GPU
)
```

### 2. 批量处理

```python
# 一次性评分多个实体，而不是逐个评分
scored_entities = scorer.score_entities(query, all_entities)
```

### 3. 模型缓存

```python
# 在应用启动时预加载模型
# 避免每次查询都重新加载
```

### 4. 使用更小的模型（如果精度要求不高）

```yaml
# 配置文件中
embedding_model_for_scoring: "sentence-transformers/all-MiniLM-L6-v2"  # 384维，快
# 或
embedding_model_for_scoring: "sentence-transformers/paraphrase-MiniLM-L3-v2"  # 更小更快
```

## 卸载

如果需要卸载新增的依赖：

```bash
# 卸载sentence-transformers及其依赖
uv pip uninstall sentence-transformers transformers torch torchvision

# 或保留torch（如果其他项目需要）
uv pip uninstall sentence-transformers transformers
```

## 下一步

安装完成后，请参考：
- [使用指南](plans/SUBGRAPH_PRUNING_README.md)
- [配置示例](examples_notebooks/subgraph_pruning_config_example.yaml)
- [数学原理](plans/subgraph_pruning_mathematical_principles.md)
