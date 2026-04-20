# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""子图裁剪器模块 - 用于裁剪召回的子图以减少Token消耗."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import networkx as nx

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


@dataclass
class PruningResult:
    """裁剪结果."""

    pruned_entities: list[Entity]
    """裁剪后的实体列表."""

    pruned_relationships: list[Relationship]
    """裁剪后的关系列表."""

    original_entity_count: int
    """原始实体数量."""

    original_relationship_count: int
    """原始关系数量."""

    tokens_before: int
    """裁剪前的token数量."""

    tokens_after: int
    """裁剪后的token数量."""

    tokens_saved: int
    """节省的token数量."""

    pruning_strategy: str
    """使用的裁剪策略."""

    metadata: dict[str, Any] | None = None
    """裁剪过程中的附加信息."""

    @property
    def token_reduction_rate(self) -> float:
        """Token减少率."""
        if self.tokens_before == 0:
            return 0.0
        return self.tokens_saved / self.tokens_before

    @property
    def entity_retention_rate(self) -> float:
        """实体保留率."""
        if self.original_entity_count == 0:
            return 0.0
        return len(self.pruned_entities) / self.original_entity_count

    @property
    def relationship_retention_rate(self) -> float:
        """关系保留率."""
        if self.original_relationship_count == 0:
            return 0.0
        return len(self.pruned_relationships) / self.original_relationship_count


class MultiLevelPruner:
    """
    多层级子图裁剪器.

    根据关联性分数进行分层裁剪：
    - 高相关性 (score >= high_threshold): 优先保留
    - 中等相关性 (medium_threshold <= score < high_threshold): 在token预算允许时保留
    - 低相关性 (score < medium_threshold): 丢弃
    """

    def __init__(
        self,
        high_relevance_threshold: float = 0.7,
        medium_relevance_threshold: float = 0.5,
        token_buffer_ratio: float = 0.9,
    ):
        """
        初始化多层级裁剪器.

        Parameters
        ----------
        high_relevance_threshold : float
            高相关性阈值
        medium_relevance_threshold : float
            中等相关性阈值
        token_buffer_ratio : float
            token预算使用比例，保留一定缓冲
        """
        self.high_threshold = high_relevance_threshold
        self.medium_threshold = medium_relevance_threshold
        self.token_buffer_ratio = token_buffer_ratio

    def prune_subgraph(
        self,
        scored_entities: list[tuple[Entity, float]],
        scored_relationships: list[tuple[Relationship, float]],
        max_context_tokens: int,
        tokenizer: Tokenizer,
        column_delimiter: str = "|",
    ) -> PruningResult:
        """
        执行多层级裁剪.

        Parameters
        ----------
        scored_entities : list[tuple[Entity, float]]
            评分后的实体列表
        scored_relationships : list[tuple[Relationship, float]]
            评分后的关系列表
        max_context_tokens : int
            最大上下文token数
        tokenizer : Tokenizer
            分词器
        column_delimiter : str
            列分隔符

        Returns
        -------
        PruningResult
            裁剪结果
        """
        # 记录原始数量
        original_entity_count = len(scored_entities)
        original_relationship_count = len(scored_relationships)

        # 估算原始token数
        all_entities = [e for e, _ in scored_entities]
        all_relationships = [r for r, _ in scored_relationships]
        tokens_before = self._estimate_tokens(
            all_entities, all_relationships, tokenizer, column_delimiter
        )

        # 按相关性分层
        high_entities = [e for e, s in scored_entities if s >= self.high_threshold]
        medium_entities = [
            e
            for e, s in scored_entities
            if self.medium_threshold <= s < self.high_threshold
        ]

        high_rels = [
            r for r, s in scored_relationships if s >= self.high_threshold
        ]
        medium_rels = [
            r
            for r, s in scored_relationships
            if self.medium_threshold <= s < self.high_threshold
        ]

        logger.info(
            f"Multi-level pruning: High entities={len(high_entities)}, "
            f"Medium entities={len(medium_entities)}, "
            f"High rels={len(high_rels)}, Medium rels={len(medium_rels)}"
        )

        # 初始化结果（优先保留高相关性元素）
        selected_entities = high_entities.copy()
        selected_rels = high_rels.copy()

        # 避免阈值过严导致实体上下文为空，至少保留一个最高分实体
        if not selected_entities and scored_entities:
            selected_entities = [scored_entities[0][0]]

        # 计算当前token使用
        current_tokens = self._estimate_tokens(
            selected_entities, selected_rels, tokenizer, column_delimiter
        )

        # 计算可用token预算
        available_tokens = int(max_context_tokens * self.token_buffer_ratio)

        logger.info(
            f"After high-relevance selection: {current_tokens} tokens "
            f"(budget: {available_tokens})"
        )

        # 贪心添加中等相关性实体
        for entity in medium_entities:
            temp_entities = selected_entities + [entity]
            temp_tokens = self._estimate_tokens(
                temp_entities, selected_rels, tokenizer, column_delimiter
            )

            if temp_tokens <= available_tokens:
                selected_entities.append(entity)
                current_tokens = temp_tokens
            else:
                break

        # 贪心添加中等相关性关系
        for rel in medium_rels:
            temp_rels = selected_rels + [rel]
            temp_tokens = self._estimate_tokens(
                selected_entities, temp_rels, tokenizer, column_delimiter
            )

            if temp_tokens <= available_tokens:
                selected_rels.append(rel)
                current_tokens = temp_tokens
            else:
                break

        # 计算最终token数
        tokens_after = self._estimate_tokens(
            selected_entities, selected_rels, tokenizer, column_delimiter
        )

        logger.info(
            f"Pruning complete: {len(selected_entities)}/{original_entity_count} entities, "
            f"{len(selected_rels)}/{original_relationship_count} relationships, "
            f"{tokens_after}/{tokens_before} tokens"
        )

        return PruningResult(
            pruned_entities=selected_entities,
            pruned_relationships=selected_rels,
            original_entity_count=original_entity_count,
            original_relationship_count=original_relationship_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            pruning_strategy="multi_level",
            metadata={
                "high_relevance_entity_count": len(high_entities),
                "medium_relevance_entity_count": len(medium_entities),
                "high_relevance_relationship_count": len(high_rels),
                "medium_relevance_relationship_count": len(medium_rels),
                "token_budget": available_tokens,
            },
        )

    def _estimate_tokens(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        tokenizer: Tokenizer,
        column_delimiter: str,
    ) -> int:
        """估算实体和关系的token数量."""
        total_tokens = 0

        # 估算实体tokens
        if entities:
            entity_text = "-----Entities-----\n"
            entity_text += f"id{column_delimiter}entity{column_delimiter}description\n"

            for entity in entities:
                entity_line = f"{entity.short_id or ''}{column_delimiter}"
                entity_line += f"{entity.title}{column_delimiter}"
                entity_line += f"{entity.description or ''}\n"
                entity_text += entity_line

            total_tokens += tokenizer.num_tokens(entity_text)

        # 估算关系tokens
        if relationships:
            rel_text = "-----Relationships-----\n"
            rel_text += f"id{column_delimiter}source{column_delimiter}target{column_delimiter}description\n"

            for rel in relationships:
                rel_line = f"{rel.short_id or ''}{column_delimiter}"
                rel_line += f"{rel.source}{column_delimiter}"
                rel_line += f"{rel.target}{column_delimiter}"
                rel_line += f"{rel.description or ''}\n"
                rel_text += rel_line

            total_tokens += tokenizer.num_tokens(rel_text)

        return total_tokens


class GraphAwarePruner:
    """
    图结构感知裁剪器.

    考虑图的拓扑结构进行智能裁剪：
    1. 保留与查询实体距离近的节点（k-hop邻居）
    2. 保留重要节点（基于PageRank、Betweenness等）
    3. 保留桥接节点（连接不同社区的关键节点）
    """

    def __init__(
        self,
        max_hops: int = 2,
        importance_weight: float = 0.5,
        distance_weight: float = 0.6,
        min_importance_score: float = 0.1,
    ):
        """
        初始化图结构感知裁剪器.

        Parameters
        ----------
        max_hops : int
            从查询实体出发的最大跳数
        importance_weight : float
            重要性权重（PageRank vs Betweenness）
        min_importance_score : float
            最小重要性分数阈值
        """
        self.max_hops = max_hops
        self.importance_weight = importance_weight
        self.distance_weight = distance_weight
        self.min_importance_score = min_importance_score

    def prune_subgraph(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        query_entities: list[str],
        max_context_tokens: int,
        tokenizer: Tokenizer,
        column_delimiter: str = "|",
    ) -> PruningResult:
        """
        基于图结构执行裁剪.

        Parameters
        ----------
        entities : list[Entity]
            实体列表
        relationships : list[Relationship]
            关系列表
        query_entities : list[str]
            查询中识别的实体名称列表
        max_context_tokens : int
            最大上下文token数
        tokenizer : Tokenizer
            分词器
        column_delimiter : str
            列分隔符

        Returns
        -------
        PruningResult
            裁剪结果
        """
        # 记录原始数量
        original_entity_count = len(entities)
        original_relationship_count = len(relationships)

        # 估算原始token数
        tokens_before = self._estimate_tokens(
            entities, relationships, tokenizer, column_delimiter
        )

        # 构建NetworkX图
        G = self._build_graph(entities, relationships)

        if len(G.nodes()) == 0:
            logger.warning("Empty graph, no pruning performed")
            return PruningResult(
                pruned_entities=entities,
                pruned_relationships=relationships,
                original_entity_count=original_entity_count,
                original_relationship_count=original_relationship_count,
                tokens_before=tokens_before,
                tokens_after=tokens_before,
                tokens_saved=0,
                pruning_strategy="graph_aware",
                metadata={"candidate_node_count": 0},
            )

        title_to_entity = {entity.title: entity for entity in entities}

        # 计算节点重要性
        try:
            pagerank = nx.pagerank(G)
            betweenness = nx.betweenness_centrality(G)
        except Exception as e:
            logger.warning(f"Failed to compute centrality metrics: {e}")
            pagerank = {node: 1.0 for node in G.nodes()}
            betweenness = {node: 0.0 for node in G.nodes()}

        normalized_pagerank = self._normalize_scores(pagerank)
        normalized_betweenness = self._normalize_scores(betweenness)

        # 找到查询实体的k-hop邻居
        candidate_nodes = set()
        for query_entity in query_entities:
            if query_entity in G:
                try:
                    neighbors = nx.single_source_shortest_path_length(
                        G, query_entity, cutoff=self.max_hops
                    )
                    candidate_nodes.update(neighbors.keys())
                except Exception as e:
                    logger.warning(f"Failed to find neighbors for {query_entity}: {e}")

        # 如果没有找到查询实体，使用所有节点
        if not candidate_nodes:
            logger.warning("No query entities found in graph, using all nodes")
            candidate_nodes = set(G.nodes())

        # 计算每个候选节点的综合分数
        node_scores = {}
        for node in candidate_nodes:
            # 距离分数（越近越好）
            min_distance = float("inf")
            for qe in query_entities:
                if qe in G:
                    try:
                        distance = nx.shortest_path_length(G, node, qe)
                        min_distance = min(min_distance, distance)
                    except nx.NetworkXNoPath:
                        continue

            if min_distance == float("inf"):
                distance_score = 0.0
            else:
                distance_score = 1.0 / (1.0 + min_distance)

            # 重要性分数
            pr_score = normalized_pagerank.get(node, 0.0)
            bc_score = normalized_betweenness.get(node, 0.0)
            importance_score = (
                self.importance_weight * pr_score
                + (1 - self.importance_weight) * bc_score
            )

            # 综合分数
            final_score = (
                self.distance_weight * distance_score
                + (1 - self.distance_weight) * importance_score
            )

            if final_score >= self.min_importance_score:
                node_scores[node] = final_score

        # 按分数排序并选择节点
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)

        # 贪心选择节点直到达到token限制
        selected_node_names = []
        selected_entities = []
        entity_token_budget = max_context_tokens * 0.7

        for node_name, _score in sorted_nodes:
            # 找到对应的实体
            entity = title_to_entity.get(node_name)
            if entity is None:
                continue

            # 估算添加此实体后的token数
            temp_entities = selected_entities + [entity]
            temp_tokens = self._estimate_tokens(
                temp_entities, [], tokenizer, column_delimiter
            )

            if temp_tokens <= entity_token_budget:  # 为关系预留30%空间
                selected_entities.append(entity)
                selected_node_names.append(node_name)
            else:
                break

        if not selected_entities and query_entities:
            for query_entity in query_entities:
                entity = title_to_entity.get(query_entity)
                if entity is not None:
                    selected_entities.append(entity)
                    selected_node_names.append(query_entity)
                    break

        # 选择连接这些节点的关系，并在剩余预算内继续贪心保留高价值边
        candidate_relationships = [
            r
            for r in relationships
            if r.source in selected_node_names and r.target in selected_node_names
        ]
        candidate_relationships.sort(
            key=lambda rel: (
                node_scores.get(rel.source, 0.0) + node_scores.get(rel.target, 0.0),
                rel.rank or 0,
                rel.weight or 0.0,
            ),
            reverse=True,
        )

        selected_relationships = []
        for relationship in candidate_relationships:
            temp_relationships = selected_relationships + [relationship]
            temp_tokens = self._estimate_tokens(
                selected_entities, temp_relationships, tokenizer, column_delimiter
            )
            if temp_tokens <= max_context_tokens:
                selected_relationships.append(relationship)

        # 计算最终token数
        tokens_after = self._estimate_tokens(
            selected_entities, selected_relationships, tokenizer, column_delimiter
        )

        logger.info(
            f"Graph-aware pruning: {len(selected_entities)}/{original_entity_count} entities, "
            f"{len(selected_relationships)}/{original_relationship_count} relationships, "
            f"{tokens_after}/{tokens_before} tokens"
        )

        return PruningResult(
            pruned_entities=selected_entities,
            pruned_relationships=selected_relationships,
            original_entity_count=original_entity_count,
            original_relationship_count=original_relationship_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            pruning_strategy="graph_aware",
            metadata={
                "candidate_node_count": len(candidate_nodes),
                "selected_node_count": len(selected_node_names),
                "distance_weight": self.distance_weight,
                "importance_weight": self.importance_weight,
            },
        )

    def _build_graph(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> nx.Graph[str]:
        """构建NetworkX图."""
        G: nx.Graph[str] = nx.Graph()

        # 添加节点
        for entity in entities:
            G.add_node(entity.title, data=entity)

        # 添加边
        for rel in relationships:
            if rel.source and rel.target:
                G.add_edge(rel.source, rel.target, data=rel, weight=rel.weight or 1.0)

        return G

    def _estimate_tokens(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        tokenizer: Tokenizer,
        column_delimiter: str,
    ) -> int:
        """估算实体和关系的token数量."""
        total_tokens = 0

        # 估算实体tokens
        if entities:
            entity_text = "-----Entities-----\n"
            entity_text += f"id{column_delimiter}entity{column_delimiter}description\n"

            for entity in entities:
                entity_line = f"{entity.short_id or ''}{column_delimiter}"
                entity_line += f"{entity.title}{column_delimiter}"
                entity_line += f"{entity.description or ''}\n"
                entity_text += entity_line

            total_tokens += tokenizer.num_tokens(entity_text)

        # 估算关系tokens
        if relationships:
            rel_text = "-----Relationships-----\n"
            rel_text += f"id{column_delimiter}source{column_delimiter}target{column_delimiter}description\n"

            for rel in relationships:
                rel_line = f"{rel.short_id or ''}{column_delimiter}"
                rel_line += f"{rel.source}{column_delimiter}"
                rel_line += f"{rel.target}{column_delimiter}"
                rel_line += f"{rel.description or ''}\n"
                rel_text += rel_line

            total_tokens += tokenizer.num_tokens(rel_text)

        return total_tokens

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        """将中心性分数归一化到[0, 1]区间."""
        if not scores:
            return {}

        max_score = max(scores.values())
        if max_score <= 0:
            return {node: 0.0 for node in scores}
        return {node: score / max_score for node, score in scores.items()}
