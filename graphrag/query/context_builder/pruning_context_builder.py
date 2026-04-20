# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""增强的上下文构建器，集成子图裁剪功能."""

import logging
from typing import Any

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship
from graphrag.query.input.retrieval.relationships import get_candidate_relationships
from graphrag.query.context_builder.relevance_scorer import RelevanceScorer
from graphrag.query.context_builder.subgraph_pruner import (
    GraphAwarePruner,
    MultiLevelPruner,
    PruningResult,
)
from graphrag.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def apply_subgraph_pruning(
    query: str,
    selected_entities: list[Entity],
    relationships: list[Relationship],
    max_context_tokens: int,
    tokenizer: Tokenizer,
    enable_pruning: bool = True,
    pruning_strategy: str = "multi_level",
    relevance_scoring_method: str = "embedding",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    high_relevance_threshold: float = 0.7,
    medium_relevance_threshold: float = 0.5,
    max_hops: int = 2,
    token_buffer_ratio: float = 0.9,
    importance_weight: float = 0.5,
    distance_weight: float = 0.6,
    min_hybrid_reduction_rate: float = 0.2,
    column_delimiter: str = "|",
    **kwargs: Any,
) -> tuple[list[Entity], list[Relationship], PruningResult | None]:
    """
    应用子图裁剪以减少Token消耗.

    Parameters
    ----------
    query : str
        用户查询
    selected_entities : list[Entity]
        已选择的实体列表
    relationships : list[Relationship]
        关系列表
    max_context_tokens : int
        最大上下文token数
    tokenizer : Tokenizer
        分词器
    enable_pruning : bool
        是否启用裁剪
    pruning_strategy : str
        裁剪策略：'multi_level', 'graph_aware', 'hybrid'
    relevance_scoring_method : str
        关联性评分方法：'embedding', 'hybrid', 'simple'
    embedding_model : str
        用于评分的embedding模型
    high_relevance_threshold : float
        高相关性阈值
    medium_relevance_threshold : float
        中等相关性阈值
    max_hops : int
        图结构裁剪的最大跳数
    token_buffer_ratio : float
        token预算使用比例
    column_delimiter : str
        列分隔符
    **kwargs : Any
        额外参数

    Returns
    -------
    tuple[list[Entity], list[Relationship], PruningResult | None]
        裁剪后的实体列表、关系列表和裁剪结果
    """
    if not enable_pruning:
        logger.info("Subgraph pruning is disabled")
        return selected_entities, relationships, None

    if not selected_entities:
        logger.warning("No entities to prune")
        return selected_entities, relationships, None

    candidate_relationships = get_candidate_relationships(
        selected_entities=selected_entities,
        relationships=relationships,
    )
    if candidate_relationships:
        relationships = candidate_relationships

    logger.info(
        f"Applying subgraph pruning: strategy={pruning_strategy}, "
        f"scoring_method={relevance_scoring_method}"
    )

    try:
        # 步骤1：关联性评分
        scorer = RelevanceScorer(
            scoring_method=relevance_scoring_method,
            embedding_model=embedding_model,
            score_threshold=0.0,  # 不在这里过滤，交给pruner处理
        )

        scored_entities = scorer.score_entities(query, selected_entities)
        scored_relationships = scorer.score_relationships(query, relationships)
        query_entity_names = [entity.title for entity, _ in scored_entities[:3]]

        logger.info(
            f"Scored {len(scored_entities)} entities and "
            f"{len(scored_relationships)} relationships"
        )

        # 步骤2：根据策略执行裁剪
        pruning_result = None

        if pruning_strategy == "multi_level":
            pruner = MultiLevelPruner(
                high_relevance_threshold=high_relevance_threshold,
                medium_relevance_threshold=medium_relevance_threshold,
                token_buffer_ratio=token_buffer_ratio,
            )
            pruning_result = pruner.prune_subgraph(
                scored_entities=scored_entities,
                scored_relationships=scored_relationships,
                max_context_tokens=max_context_tokens,
                tokenizer=tokenizer,
                column_delimiter=column_delimiter,
            )

        elif pruning_strategy == "graph_aware":
            pruner = GraphAwarePruner(
                max_hops=max_hops,
                importance_weight=importance_weight,
                distance_weight=distance_weight,
                min_importance_score=medium_relevance_threshold,
            )
            pruning_result = pruner.prune_subgraph(
                entities=selected_entities,
                relationships=relationships,
                query_entities=query_entity_names,
                max_context_tokens=max_context_tokens,
                tokenizer=tokenizer,
                column_delimiter=column_delimiter,
            )

        elif pruning_strategy == "hybrid":
            # 混合策略：先用multi_level裁剪，再用graph_aware优化
            pruner1 = MultiLevelPruner(
                high_relevance_threshold=high_relevance_threshold,
                medium_relevance_threshold=medium_relevance_threshold,
                token_buffer_ratio=token_buffer_ratio,
            )
            result1 = pruner1.prune_subgraph(
                scored_entities=scored_entities,
                scored_relationships=scored_relationships,
                max_context_tokens=max_context_tokens,
                tokenizer=tokenizer,
                column_delimiter=column_delimiter,
            )

            # 如果第一步裁剪效果不够好，使用graph_aware进一步优化
            if result1.token_reduction_rate < min_hybrid_reduction_rate:
                query_entity_names = [e.title for e in result1.pruned_entities[:3]]
                pruner2 = GraphAwarePruner(
                    max_hops=max_hops,
                    importance_weight=importance_weight,
                    distance_weight=distance_weight,
                    min_importance_score=medium_relevance_threshold,
                )
                pruning_result = pruner2.prune_subgraph(
                    entities=result1.pruned_entities,
                    relationships=result1.pruned_relationships,
                    query_entities=query_entity_names,
                    max_context_tokens=max_context_tokens,
                    tokenizer=tokenizer,
                    column_delimiter=column_delimiter,
                )
            else:
                pruning_result = result1

        else:
            logger.warning(f"Unknown pruning strategy: {pruning_strategy}")
            return selected_entities, relationships, None

        # 记录裁剪效果
        if pruning_result:
            pruning_metadata = pruning_result.metadata or {}
            pruning_metadata.update({
                "candidate_relationship_count": len(relationships),
                "importance_weight": importance_weight,
                "distance_weight": distance_weight,
                "min_hybrid_reduction_rate": min_hybrid_reduction_rate,
            })
            pruning_result.metadata = pruning_metadata
            logger.info(
                f"Pruning complete: "
                f"Entities: {pruning_result.original_entity_count} -> {len(pruning_result.pruned_entities)} "
                f"({pruning_result.entity_retention_rate:.1%} retained), "
                f"Relationships: {pruning_result.original_relationship_count} -> {len(pruning_result.pruned_relationships)} "
                f"({pruning_result.relationship_retention_rate:.1%} retained), "
                f"Tokens: {pruning_result.tokens_before} -> {pruning_result.tokens_after} "
                f"({pruning_result.token_reduction_rate:.1%} saved)"
            )

            return (
                pruning_result.pruned_entities,
                pruning_result.pruned_relationships,
                pruning_result,
            )

        return selected_entities, relationships, None

    except Exception as e:
        logger.error(f"Error during subgraph pruning: {e}", exc_info=True)
        logger.warning("Falling back to original entities and relationships")
        return selected_entities, relationships, None
