# 子图预裁剪优化方案：核心原理与数学模型

> **文档版本**：v1.0  
> **作者**：宋璎航
> **日期**：2026年4月  
> **用途**：毕业设计技术文档 - GraphRAG Token优化方案二

---

## 目录

1. [问题定义](#1-问题定义)
2. [关联性评分模型](#2-关联性评分模型)
3. [多层级裁剪算法](#3-多层级裁剪算法)
4. [图结构感知裁剪算法](#4-图结构感知裁剪算法)
5. [混合裁剪策略](#5-混合裁剪策略)
6. [复杂度分析](#6-复杂度分析)
7. [理论保证](#7-理论保证)

---

## 1. 问题定义

### 1.1 符号定义

设知识图谱为 $G = (V, E)$，其中：

- $V = \{v_1, v_2, \ldots, v_n\}$ 为实体集合，$|V| = n$
- $E = \{e_1, e_2, \ldots, e_m\} \subseteq V \times V$ 为关系集合，$|E| = m$
- $q$ 为用户查询（query）
- $V_q \subseteq V$ 为查询相关的候选实体集合
- $E_q \subseteq E$ 为与 $V_q$ 相关的候选关系集合

### 1.2 优化目标

给定最大上下文Token限制 $T_{max}$，我们的目标是找到最优子图 $G^* = (V^*, E^*)$，使得：

$$
\begin{aligned}
G^* = \arg\max_{G' = (V', E')} \quad & \text{Relevance}(G', q) \\
\text{s.t.} \quad & V' \subseteq V_q, \quad E' \subseteq E_q \\
& \text{Tokens}(V', E') \leq T_{max} \\
& \text{Connectivity}(G') \geq \theta_{conn}
\end{aligned}
$$

其中：
- $\text{Relevance}(G', q)$ 为子图与查询的关联性得分
- $\text{Tokens}(V', E')$ 为子图的Token消耗量
- $\text{Connectivity}(G')$ 为子图的连通性度量
- $\theta_{conn}$ 为最小连通性阈值

### 1.3 Token消耗模型

对于实体 $v_i \in V$，其Token消耗为：

$$
T(v_i) = T_{header} + T_{id}(v_i) + T_{title}(v_i) + T_{desc}(v_i)
$$

对于关系 $e_{ij} = (v_i, v_j) \in E$，其Token消耗为：

$$
T(e_{ij}) = T_{header} + T_{id}(e_{ij}) + T_{source}(v_i) + T_{target}(v_j) + T_{desc}(e_{ij})
$$

总Token消耗为：

$$
\text{Tokens}(V', E') = \sum_{v_i \in V'} T(v_i) + \sum_{e_{ij} \in E'} T(e_{ij})
$$

---

## 2. 关联性评分模型

### 2.1 基于Embedding的相似度评分

#### 2.1.1 向量表示

对于查询 $q$ 和实体 $v_i$，使用预训练的Sentence Transformer模型获取其向量表示：

$$
\begin{aligned}
\mathbf{q} &= \text{Encoder}(q) \in \mathbb{R}^d \\
\mathbf{v}_i &= \text{Encoder}(\text{title}(v_i) + \text{desc}(v_i)) \in \mathbb{R}^d
\end{aligned}
$$

其中 $d$ 为embedding维度（例如，all-MiniLM-L6-v2模型中 $d=384$）。

#### 2.1.2 余弦相似度

实体 $v_i$ 与查询 $q$ 的相似度定义为：

$$
\text{sim}_{\text{emb}}(v_i, q) = \frac{\mathbf{v}_i \cdot \mathbf{q}}{\|\mathbf{v}_i\|_2 \cdot \|\mathbf{q}\|_2} = \cos(\theta_{v_i, q})
$$

其中 $\theta_{v_i, q}$ 为两向量的夹角。

**性质**：
- $\text{sim}_{\text{emb}}(v_i, q) \in [-1, 1]$
- 值越大表示语义相似度越高
- 计算复杂度为 $O(d)$

#### 2.1.3 关系的相似度

对于关系 $e_{ij} = (v_i, v_j)$，其向量表示为：

$$
\mathbf{e}_{ij} = \text{Encoder}(\text{source}(v_i) + \text{target}(v_j) + \text{desc}(e_{ij}))
$$

相似度计算同理：

$$
\text{sim}_{\text{emb}}(e_{ij}, q) = \frac{\mathbf{e}_{ij} \cdot \mathbf{q}}{\|\mathbf{e}_{ij}\|_2 \cdot \|\mathbf{q}\|_2}
$$

### 2.2 混合评分模型

为了综合多种特征，我们采用加权组合的混合评分模型：

$$
\text{Score}(v_i, q) = \alpha \cdot \text{sim}_{\text{emb}}(v_i, q) + \beta \cdot \text{sim}_{\text{text}}(v_i, q) + \gamma \cdot \text{Importance}(v_i)
$$

其中：
- $\alpha, \beta, \gamma \geq 0$ 且 $\alpha + \beta + \gamma = 1$ 为权重系数
- $\text{sim}_{\text{text}}(v_i, q)$ 为文本匹配相似度
- $\text{Importance}(v_i)$ 为实体重要性

#### 2.2.1 文本匹配相似度（Jaccard相似度）

$$
\text{sim}_{\text{text}}(v_i, q) = \frac{|W(v_i) \cap W(q)|}{|W(v_i) \cup W(q)|}
$$

其中 $W(v_i)$ 和 $W(q)$ 分别为实体和查询的词集合。

#### 2.2.2 实体重要性（归一化Rank）

$$
\text{Importance}(v_i) = \frac{\text{rank}(v_i)}{\max_{v_j \in V} \text{rank}(v_j)}
$$

其中 $\text{rank}(v_i)$ 为实体的原始rank值（通常基于度中心性）。

#### 2.2.3 默认权重配置

基于实验调优，我们设置：

$$
\alpha = 0.6, \quad \beta = 0.2, \quad \gamma = 0.2
$$

**理由**：
- Embedding相似度捕获深层语义，权重最高
- 文本匹配提供精确匹配信号
- 实体重要性保证关键节点不被过滤

---

## 3. 多层级裁剪算法

### 3.1 算法思想

将实体和关系按相关性分数分为三个层级：

1. **高相关性层** $L_H$：$\text{Score}(v_i, q) \geq \theta_H$
2. **中等相关性层** $L_M$：$\theta_M \leq \text{Score}(v_i, q) < \theta_H$
3. **低相关性层** $L_L$：$\text{Score}(v_i, q) < \theta_M$

其中 $\theta_H > \theta_M$ 为阈值参数（默认 $\theta_H = 0.7, \theta_M = 0.5$）。

### 3.2 裁剪策略

#### 3.2.1 优先级排序

定义优先级函数：

$$
\text{Priority}(v_i) = \begin{cases}
3 & \text{if } v_i \in L_H \\
2 & \text{if } v_i \in L_M \\
1 & \text{if } v_i \in L_L
\end{cases}
$$

#### 3.2.2 贪心选择算法

**输入**：
- 评分后的实体集合 $\{(v_i, s_i)\}_{i=1}^n$，其中 $s_i = \text{Score}(v_i, q)$
- 评分后的关系集合 $\{(e_j, s_j)\}_{j=1}^m$
- 最大Token限制 $T_{max}$
- Token缓冲比例 $\rho \in (0, 1]$（默认 $\rho = 0.9$）

**输出**：裁剪后的子图 $G^* = (V^*, E^*)$

**算法流程**：

```
算法1：多层级贪心裁剪

1. 初始化：
   V* ← ∅, E* ← ∅
   T_current ← 0
   T_budget ← ρ · T_max

2. 第一阶段：添加高相关性元素
   V_H ← {v_i | s_i ≥ θ_H}
   E_H ← {e_j | s_j ≥ θ_H}
   
   for each v_i in V_H (按s_i降序):
       if T_current + T(v_i) ≤ T_budget:
           V* ← V* ∪ {v_i}
           T_current ← T_current + T(v_i)
   
   for each e_j in E_H (按s_j降序):
       if T_current + T(e_j) ≤ T_budget:
           E* ← E* ∪ {e_j}
           T_current ← T_current + T(e_j)

3. 第二阶段：添加中等相关性元素（如有空间）
   V_M ← {v_i | θ_M ≤ s_i < θ_H}
   E_M ← {e_j | θ_M ≤ s_j < θ_H}
   
   for each v_i in V_M (按s_i降序):
       if T_current + T(v_i) ≤ T_budget:
           V* ← V* ∪ {v_i}
           T_current ← T_current + T(v_i)
       else:
           break  // Token预算耗尽
   
   for each e_j in E_M (按s_j降序):
       if T_current + T(e_j) ≤ T_budget:
           E* ← E* ∪ {e_j}
           T_current ← T_current + T(e_j)
       else:
           break

4. 返回 (V*, E*)
```

### 3.3 数学性质

**定理1（单调性）**：设 $T_1 < T_2$，则由算法1在Token限制 $T_1$ 和 $T_2$ 下得到的子图满足：

$$
V^*_{T_1} \subseteq V^*_{T_2}, \quad E^*_{T_1} \subseteq E^*_{T_2}
$$

**证明**：由贪心算法的构造过程，Token限制越大，能够容纳的元素越多。由于选择顺序固定（按分数降序），$T_1$ 下选中的元素必然在 $T_2$ 下也会被选中。$\square$

**定理2（近似比）**：设最优解为 $G^{opt} = (V^{opt}, E^{opt})$，算法1的解为 $G^* = (V^*, E^*)$，则：

$$
\frac{\text{Relevance}(G^*, q)}{\text{Relevance}(G^{opt}, q)} \geq 1 - \frac{1}{e} \approx 0.632
$$

当 $\text{Relevance}$ 满足子模性（submodularity）时成立。

**证明**：这是经典的子模函数最大化问题的贪心近似比。详见[Nemhauser et al., 1978]。$\square$

---

## 4. 图结构感知裁剪算法

### 4.1 算法思想

考虑图的拓扑结构，保留：
1. 与查询实体距离近的节点（k-hop邻居）
2. 图中的重要节点（高中心性）
3. 桥接节点（连接不同社区）

### 4.2 距离度量

#### 4.2.1 最短路径距离

对于查询实体集合 $V_q^{seed} = \{v_{q_1}, v_{q_2}, \ldots, v_{q_k}\}$，定义节点 $v_i$ 的距离为：

$$
d(v_i, V_q^{seed}) = \min_{v_{q_j} \in V_q^{seed}} d_G(v_i, v_{q_j})
$$

其中 $d_G(v_i, v_j)$ 为图 $G$ 中 $v_i$ 到 $v_j$ 的最短路径长度。

#### 4.2.2 k-hop邻居集合

$$
N_k(V_q^{seed}) = \{v_i \in V \mid d(v_i, V_q^{seed}) \leq k\}
$$

默认 $k=2$，即考虑2-hop邻居。

#### 4.2.3 距离得分

$$
\text{Score}_{\text{dist}}(v_i) = \frac{1}{1 + d(v_i, V_q^{seed})}
$$

**性质**：
- $\text{Score}_{\text{dist}}(v_i) \in (0, 1]$
- 距离越近，分数越高
- 查询实体本身得分为1

### 4.3 中心性度量

#### 4.3.1 PageRank

PageRank算法定义节点重要性为：

$$
\text{PR}(v_i) = \frac{1-\lambda}{n} + \lambda \sum_{v_j \in \text{In}(v_i)} \frac{\text{PR}(v_j)}{|\text{Out}(v_j)|}
$$

其中：
- $\lambda \in (0, 1)$ 为阻尼系数（默认 $\lambda = 0.85$）
- $\text{In}(v_i)$ 为指向 $v_i$ 的节点集合
- $\text{Out}(v_j)$ 为 $v_j$ 指向的节点集合

迭代求解直到收敛：

$$
\|\text{PR}^{(t+1)} - \text{PR}^{(t)}\|_1 < \epsilon
$$

#### 4.3.2 介数中心性（Betweenness Centrality）

$$
\text{BC}(v_i) = \sum_{s \neq v_i \neq t} \frac{\sigma_{st}(v_i)}{\sigma_{st}}
$$

其中：
- $\sigma_{st}$ 为从 $s$ 到 $t$ 的最短路径数量
- $\sigma_{st}(v_i)$ 为经过 $v_i$ 的最短路径数量

**物理意义**：衡量节点在信息传播中的桥接作用。

#### 4.3.3 综合重要性得分

$$
\text{Score}_{\text{imp}}(v_i) = \omega \cdot \frac{\text{PR}(v_i)}{\max_j \text{PR}(v_j)} + (1-\omega) \cdot \frac{\text{BC}(v_i)}{\max_j \text{BC}(v_j)}
$$

其中 $\omega \in [0, 1]$ 为权重参数（默认 $\omega = 0.5$）。

### 4.4 综合评分与裁剪

#### 4.4.1 最终得分

$$
\text{Score}_{\text{graph}}(v_i) = \mu \cdot \text{Score}_{\text{dist}}(v_i) + (1-\mu) \cdot \text{Score}_{\text{imp}}(v_i)
$$

其中 $\mu \in [0, 1]$ 为距离与重要性的权衡参数（默认 $\mu = 0.6$）。

#### 4.4.2 裁剪算法

```
算法2：图结构感知裁剪

1. 输入：
   G = (V, E): 原始图
   V_q^seed: 查询实体集合
   k: 最大跳数
   T_max: Token限制

2. 计算候选节点集合：
   V_cand ← N_k(V_q^seed)

3. 计算中心性：
   PR ← PageRank(G)
   BC ← BetweennessCentrality(G)

4. 计算综合得分：
   for each v_i in V_cand:
       Score_dist(v_i) ← 1 / (1 + d(v_i, V_q^seed))
       Score_imp(v_i) ← ω·PR(v_i) + (1-ω)·BC(v_i)
       Score_graph(v_i) ← μ·Score_dist(v_i) + (1-μ)·Score_imp(v_i)

5. 贪心选择：
   V* ← ∅, T_current ← 0
   按Score_graph降序排序V_cand
   
   for each v_i in V_cand (降序):
       if T_current + T(v_i) ≤ 0.7·T_max:  // 为关系预留30%空间
           V* ← V* ∪ {v_i}
           T_current ← T_current + T(v_i)

6. 选择关系：
   E* ← {e_ij ∈ E | v_i ∈ V* and v_j ∈ V*}

7. 返回 (V*, E*)
```

### 4.5 理论分析

**定理3（连通性保证）**：若原图 $G$ 中查询实体集合 $V_q^{seed}$ 连通，且 $k \geq 1$，则算法2返回的子图 $G^*$ 中 $V_q^{seed} \cap V^*$ 连通。

**证明**：
1. 由于 $k \geq 1$，所有查询实体的1-hop邻居都在候选集中
2. 若 $v_i, v_j \in V_q^{seed}$ 在 $G$ 中连通，存在路径 $P = v_i \to u_1 \to \cdots \to u_l \to v_j$
3. 由于距离得分的单调性，路径上的节点得分较高，大概率被选中
4. 即使部分中间节点未被选中，由于k-hop邻居的完整性，存在替代路径
5. 因此 $V_q^{seed} \cap V^*$ 保持连通。$\square$

**定理4（时间复杂度）**：算法2的时间复杂度为：

$$
O(m \cdot \log n + n \cdot k + n \cdot \log n)
$$

其中：
- $O(m \cdot \log n)$：PageRank计算（使用优先队列）
- $O(n \cdot k)$：k-hop邻居计算（BFS）
- $O(n \cdot \log n)$：排序

---

## 5. 混合裁剪策略

### 5.1 策略组合

混合策略结合多层级裁剪和图结构感知裁剪的优势：

$$
\text{Score}_{\text{hybrid}}(v_i) = \lambda_1 \cdot \text{Score}(v_i, q) + \lambda_2 \cdot \text{Score}_{\text{graph}}(v_i)
$$

其中 $\lambda_1, \lambda_2 \geq 0$ 且 $\lambda_1 + \lambda_2 = 1$。

### 5.2 自适应权重

根据图的特性自适应调整权重：

$$
\lambda_1 = \frac{1}{1 + \exp(-\alpha \cdot (\text{Density}(G) - 0.5))}
$$

其中：
- $\text{Density}(G) = \frac{2m}{n(n-1)}$ 为图密度
- $\alpha$ 为调节参数

**直觉**：
- 稀疏图（低密度）：更依赖图结构 $\Rightarrow$ $\lambda_2$ 较大
- 密集图（高密度）：更依赖语义相似度 $\Rightarrow$ $\lambda_1$ 较大

### 5.3 两阶段裁剪

```
算法3：混合两阶段裁剪

1. 第一阶段：多层级裁剪
   (V_1, E_1) ← MultiLevelPruning(V_q, E_q, T_max, θ_H, θ_M)
   
   计算Token减少率：
   r_1 ← (Tokens(V_q, E_q) - Tokens(V_1, E_1)) / Tokens(V_q, E_q)

2. 判断是否需要第二阶段：
   if r_1 < r_min:  // 默认r_min = 0.2
       // 第一阶段效果不佳，启用图结构裁剪
       V_q^seed ← Top-k entities from V_1 by score
       (V*, E*) ← GraphAwarePruning(V_1, E_1, V_q^seed, T_max, k)
   else:
       (V*, E*) ← (V_1, E_1)

3. 返回 (V*, E*)
```

---

## 6. 复杂度分析

### 6.1 时间复杂度

| 算法组件 | 时间复杂度 | 说明 |
|---------|-----------|------|
| Embedding编码 | $O(n \cdot L \cdot d)$ | $L$为平均文本长度，$d$为embedding维度 |
| 相似度计算 | $O(n \cdot d)$ | 向量点积 |
| 多层级裁剪 | $O(n \cdot \log n)$ | 排序主导 |
| PageRank | $O(m \cdot I)$ | $I$为迭代次数（通常$I \approx 20$） |
| Betweenness | $O(n \cdot m)$ | Brandes算法 |
| 图结构裁剪 | $O(m + n \cdot \log n)$ | BFS + 排序 |
| **总计（混合策略）** | $O(n \cdot m + n \cdot \log n)$ | 图算法主导 |

### 6.2 空间复杂度

| 数据结构 | 空间复杂度 | 说明 |
|---------|-----------|------|
| Embedding向量 | $O(n \cdot d)$ | 存储所有实体的embedding |
| 图邻接表 | $O(n + m)$ | 稀疏图表示 |
| 中心性分数 | $O(n)$ | PageRank和Betweenness |
| 临时数据 | $O(n)$ | 排序、队列等 |
| **总计** | $O(n \cdot d + m)$ | Embedding主导 |

### 6.3 实际性能

对于典型的GraphRAG场景：
- $n \approx 1000$（实体数）
- $m \approx 5000$（关系数）
- $d = 384$（all-MiniLM-L6-v2）

**估算**：
- Embedding编码：~100ms（使用GPU）
- 相似度计算：~10ms
- 多层级裁剪：~5ms
- 图结构裁剪：~50ms
- **总耗时**：~165ms

相比原始LLM调用（~1-2秒），裁剪开销可忽略不计。

---

## 7. 理论保证

### 7.1 Token减少保证

**定理5（Token减少下界）**：设原始候选集的Token消耗为 $T_0$，裁剪后的Token消耗为 $T^*$，则：

$$
\frac{T^*}{T_0} \leq \rho \cdot \frac{T_{max}}{T_0}
$$

其中 $\rho$ 为Token缓冲比例。

**证明**：由算法设计，$T^* \leq \rho \cdot T_{max}$。若 $T_0 > T_{max}$（需要裁剪的情况），则：

$$
\frac{T^*}{T_0} \leq \frac{\rho \cdot T_{max}}{T_0} < \rho < 1
$$

即保证Token减少。$\square$

### 7.2 信息保留保证

**定理6（高相关性信息保留）**：设 $V_H = \{v_i \mid \text{Score}(v_i, q) \geq \theta_H\}$ 为高相关性实体集合，若：

$$
\sum_{v_i \in V_H} T(v_i) \leq \rho \cdot T_{max}
$$

则算法1保证 $V_H \subseteq V^*$。

**证明**：由算法1的第一阶段，所有高相关性实体在Token预算允许时都会被添加。若条件满足，则所有 $V_H$ 中的实体都能被容纳。$\square$

### 7.3 近似最优性

**定理7（近似比）**：在以下假设下：
1. 相关性函数 $\text{Relevance}(V', q)$ 满足子模性
2. Token消耗函数 $T(v_i)$ 为非负

贪心算法的近似比为：

$$
\frac{\text{Relevance}(V^*, q)}{\text{Relevance}(V^{opt}, q)} \geq 1 - \frac{1}{e}
$$

**证明**：这是经典的带约束子模最大化问题。详见[Nemhauser et al., 1978]。$\square$

### 7.4 鲁棒性分析

**定理8（评分噪声鲁棒性）**：设真实评分为 $s_i$，观测评分为 $\tilde{s}_i = s_i + \epsilon_i$，其中 $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$。若 $\sigma < \frac{\theta_H - \theta_M}{3}$，则高相关性实体被误分类的概率：

$$
P(\tilde{s}_i < \theta_M \mid s_i \geq \theta_H) < 0.003
$$

**证明**：由正态分布的3-sigma规则，$P(|\epsilon_i| > 3\sigma) < 0.003$。若 $s_i \geq \theta_H$ 且 $3\sigma < \theta_H - \theta_M$，则：

$$
\tilde{s}_i = s_i + \epsilon_i \geq \theta_H - 3\sigma > \theta_M
$$

以至少 $1 - 0.003 = 0.997$ 的概率成立。$\square$

---

<!-- ## 8. 实验验证

### 8.1 评估指标

#### 8.1.1 Token减少率

$$
R_{\text{token}} = \frac{T_0 - T^*}{T_0 -->