# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""子图关联性评分器模块 - 用于评估实体和关系与查询的相关性."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from graphrag.data_model.entity import Entity
from graphrag.data_model.relationship import Relationship

logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """评分结果."""

    scored_entities: list[tuple[Entity, float]]
    """评分后的实体列表 (实体, 分数)."""

    scored_relationships: list[tuple[Relationship, float]]
    """评分后的关系列表 (关系, 分数)."""

    scoring_method: str
    """使用的评分方法."""

    total_time: float = 0.0
    """评分总耗时（秒）."""


class RelevanceScorer:
    """
    子图关联性评分器.

    支持多种评分方法：
    1. embedding: 基于文本嵌入的余弦相似度
    2. hybrid: 结合多种特征的混合评分
    """

    def __init__(
        self,
        scoring_method: str = "embedding",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        score_threshold: float = 0.0,
        device: str = "cpu",
    ):
        """
        初始化关联性评分器.

        Parameters
        ----------
        scoring_method : str
            评分方法，可选 "embedding" 或 "hybrid"
        embedding_model : str
            HuggingFace模型名称，默认使用轻量级的all-MiniLM-L6-v2
        score_threshold : float
            分数阈值，低于此值的元素将被过滤
        device : str
            计算设备，"cpu" 或 "cuda"
        """
        self.scoring_method = scoring_method
        self.score_threshold = score_threshold
        self.device = device
        self.embedding_model_name = embedding_model

        # 初始化embedding模型
        if scoring_method in ["embedding", "hybrid"]:
            if SentenceTransformer is None:
                logger.warning(
                    "sentence-transformers is not installed, falling back to simple text matching"
                )
                self.embedding_model = None
                self.scoring_method = "simple"
            else:
                try:
                    logger.info(f"Loading embedding model: {embedding_model}")
                    self.embedding_model = SentenceTransformer(
                        embedding_model, device=device
                    )
                    logger.info("Embedding model loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load embedding model: {e}")
                    logger.warning("Falling back to simple text matching")
                    self.embedding_model = None
                    self.scoring_method = "simple"
        else:
            self.embedding_model = None

    def score_entities(
        self, query: str, entities: list[Entity], **kwargs: Any
    ) -> list[tuple[Entity, float]]:
        """
        为实体列表打分.

        Parameters
        ----------
        query : str
            用户查询
        entities : list[Entity]
            待评分的实体列表
        **kwargs : Any
            额外参数

        Returns
        -------
        list[tuple[Entity, float]]
            评分后的实体列表，按分数降序排列
        """
        if not entities:
            return []

        if self.scoring_method == "embedding":
            scored = self._score_by_embedding(query, entities)
        elif self.scoring_method == "hybrid":
            scored = self._score_hybrid(query, entities)
        else:
            scored = self._score_simple(query, entities)

        # 过滤低分实体
        scored = [(e, s) for e, s in scored if s >= self.score_threshold]

        # 按分数降序排序
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def score_relationships(
        self, query: str, relationships: list[Relationship], **kwargs: Any
    ) -> list[tuple[Relationship, float]]:
        """
        为关系列表打分.

        Parameters
        ----------
        query : str
            用户查询
        relationships : list[Relationship]
            待评分的关系列表
        **kwargs : Any
            额外参数

        Returns
        -------
        list[tuple[Relationship, float]]
            评分后的关系列表，按分数降序排列
        """
        if not relationships:
            return []

        if self.scoring_method == "embedding":
            scored = self._score_relationships_by_embedding(query, relationships)
        elif self.scoring_method == "hybrid":
            scored = self._score_relationships_hybrid(query, relationships)
        else:
            scored = self._score_relationships_simple(query, relationships)

        # 过滤低分关系
        scored = [(r, s) for r, s in scored if s >= self.score_threshold]

        # 按分数降序排序
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _score_by_embedding(
        self, query: str, entities: list[Entity]
    ) -> list[tuple[Entity, float]]:
        """基于Embedding相似度为实体打分."""
        if self.embedding_model is None:
            return self._score_simple(query, entities)

        try:
            # 获取query的embedding
            query_embedding = self.embedding_model.encode(
                query, convert_to_numpy=True, show_progress_bar=False
            )
            entity_texts = [self._build_entity_text(entity) for entity in entities]
            entity_embeddings = self.embedding_model.encode(
                entity_texts, convert_to_numpy=True, show_progress_bar=False
            )

            return [
                (entity, float(self._cosine_similarity(query_embedding, entity_embedding)))
                for entity, entity_embedding in zip(entities, entity_embeddings, strict=False)
            ]

        except Exception as e:
            logger.error(f"Error in embedding-based scoring: {e}")
            return self._score_simple(query, entities)

    def _score_relationships_by_embedding(
        self, query: str, relationships: list[Relationship]
    ) -> list[tuple[Relationship, float]]:
        """基于Embedding相似度为关系打分."""
        if self.embedding_model is None:
            return self._score_relationships_simple(query, relationships)

        try:
            # 获取query的embedding
            query_embedding = self.embedding_model.encode(
                query, convert_to_numpy=True, show_progress_bar=False
            )
            relationship_texts = [
                self._build_relationship_text(rel) for rel in relationships
            ]
            relationship_embeddings = self.embedding_model.encode(
                relationship_texts, convert_to_numpy=True, show_progress_bar=False
            )

            return [
                (rel, float(self._cosine_similarity(query_embedding, rel_embedding)))
                for rel, rel_embedding in zip(
                    relationships, relationship_embeddings, strict=False
                )
            ]

        except Exception as e:
            logger.error(f"Error in embedding-based relationship scoring: {e}")
            return self._score_relationships_simple(query, relationships)

    def _score_hybrid(
        self, query: str, entities: list[Entity]
    ) -> list[tuple[Entity, float]]:
        """
        混合评分策略.

        结合：
        1. Embedding相似度 (权重0.6)
        2. 文本匹配度 (权重0.2)
        3. 实体重要性 (rank) (权重0.2)
        """
        # 获取embedding分数
        embedding_scores = {
            entity.id: score for entity, score in self._score_by_embedding(query, entities)
        }

        # 获取文本匹配分数
        text_scores = {
            entity.id: score for entity, score in self._score_simple(query, entities)
        }

        # 归一化rank分数
        max_rank = max((e.rank or 1) for e in entities)
        rank_scores = {entity.id: (entity.rank or 1) / max_rank for entity in entities}

        # 混合评分
        scored_entities = []
        for entity in entities:
            embedding_score = embedding_scores.get(entity.id, 0.0)
            text_score = text_scores.get(entity.id, 0.0)
            rank_score = rank_scores.get(entity.id, 0.0)

            # 加权组合
            final_score = (
                0.6 * embedding_score + 0.2 * text_score + 0.2 * rank_score
            )

            scored_entities.append((entity, final_score))

        return scored_entities

    def _score_relationships_hybrid(
        self, query: str, relationships: list[Relationship]
    ) -> list[tuple[Relationship, float]]:
        """
        关系的混合评分策略.

        结合：
        1. Embedding相似度 (权重0.6)
        2. 文本匹配度 (权重0.2)
        3. 关系权重 (weight) (权重0.2)
        """
        # 获取embedding分数
        embedding_scores = {
            rel.id: score
            for rel, score in self._score_relationships_by_embedding(
                query, relationships
            )
        }

        # 获取文本匹配分数
        text_scores = {
            rel.id: score
            for rel, score in self._score_relationships_simple(query, relationships)
        }

        # 归一化weight分数
        max_weight = max((r.weight or 1.0) for r in relationships)
        weight_scores = {
            rel.id: (rel.weight or 1.0) / max_weight for rel in relationships
        }

        # 混合评分
        scored_relationships = []
        for rel in relationships:
            embedding_score = embedding_scores.get(rel.id, 0.0)
            text_score = text_scores.get(rel.id, 0.0)
            weight_score = weight_scores.get(rel.id, 0.0)

            # 加权组合
            final_score = (
                0.6 * embedding_score + 0.2 * text_score + 0.2 * weight_score
            )

            scored_relationships.append((rel, final_score))

        return scored_relationships

    def _score_simple(
        self, query: str, entities: list[Entity]
    ) -> list[tuple[Entity, float]]:
        """简单的文本匹配评分（后备方案）."""
        query_terms = self._tokenize_text(query)

        scored_entities = []
        for entity in entities:
            entity_terms = self._tokenize_text(self._build_entity_text(entity))

            # 计算Jaccard相似度
            if not query_terms or not entity_terms:
                score = 0.0
            else:
                intersection = len(query_terms & entity_terms)
                union = len(query_terms | entity_terms)
                score = intersection / union if union > 0 else 0.0

            scored_entities.append((entity, score))

        return scored_entities

    def _score_relationships_simple(
        self, query: str, relationships: list[Relationship]
    ) -> list[tuple[Relationship, float]]:
        """简单的关系文本匹配评分（后备方案）."""
        query_terms = self._tokenize_text(query)

        scored_relationships = []
        for rel in relationships:
            rel_terms = self._tokenize_text(self._build_relationship_text(rel))

            # 计算Jaccard相似度
            if not query_terms or not rel_terms:
                score = 0.0
            else:
                intersection = len(query_terms & rel_terms)
                union = len(query_terms | rel_terms)
                score = intersection / union if union > 0 else 0.0

            scored_relationships.append((rel, score))

        return scored_relationships

    @staticmethod
    def _cosine_similarity(vec1: NDArray[np.float64], vec2: NDArray[np.float64]) -> float:
        """计算两个向量的余弦相似度."""
        # 确保是一维向量
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()

        # 计算余弦相似度
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    @staticmethod
    def _build_entity_text(entity: Entity) -> str:
        """构建实体评分文本."""
        entity_text = entity.title
        if entity.description:
            entity_text += f": {entity.description}"
        return entity_text

    @staticmethod
    def _build_relationship_text(rel: Relationship) -> str:
        """构建关系评分文本."""
        rel_text = f"{rel.source} -> {rel.target}"
        if rel.description:
            rel_text += f": {rel.description}"
        return rel_text

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        """对文本进行轻量分词，避免大小写和多余空白影响匹配."""
        return {token for token in text.lower().split() if token}
