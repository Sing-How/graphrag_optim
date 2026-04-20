# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


class LocalSearchConfig(BaseModel):
    """The default configuration section for Cache."""

    prompt: str | None = Field(
        description="The local search prompt to use.",
        default=graphrag_config_defaults.local_search.prompt,
    )
    chat_model_id: str = Field(
        description="The model ID to use for local search.",
        default=graphrag_config_defaults.local_search.chat_model_id,
    )
    embedding_model_id: str = Field(
        description="The model ID to use for text embeddings.",
        default=graphrag_config_defaults.local_search.embedding_model_id,
    )
    text_unit_prop: float = Field(
        description="The text unit proportion.",
        default=graphrag_config_defaults.local_search.text_unit_prop,
    )
    community_prop: float = Field(
        description="The community proportion.",
        default=graphrag_config_defaults.local_search.community_prop,
    )
    conversation_history_max_turns: int = Field(
        description="The conversation history maximum turns.",
        default=graphrag_config_defaults.local_search.conversation_history_max_turns,
    )
    top_k_entities: int = Field(
        description="The top k mapped entities.",
        default=graphrag_config_defaults.local_search.top_k_entities,
    )
    top_k_relationships: int = Field(
        description="The top k mapped relations.",
        default=graphrag_config_defaults.local_search.top_k_relationships,
    )
    max_context_tokens: int = Field(
        description="The maximum tokens.",
        default=graphrag_config_defaults.local_search.max_context_tokens,
    )

    # 子图裁剪配置
    enable_subgraph_pruning: bool = Field(
        description="Enable subgraph pruning to reduce token consumption.",
        default=False,
    )
    pruning_strategy: str = Field(
        description="Pruning strategy: 'multi_level', 'graph_aware', or 'hybrid'.",
        default="multi_level",
    )
    relevance_scoring_method: str = Field(
        description="Relevance scoring method: 'embedding', 'hybrid', or 'simple'.",
        default="embedding",
    )
    embedding_model_for_scoring: str = Field(
        description="HuggingFace embedding model for relevance scoring.",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    high_relevance_threshold: float = Field(
        description="High relevance threshold for multi-level pruning.",
        default=0.7,
    )
    medium_relevance_threshold: float = Field(
        description="Medium relevance threshold for multi-level pruning.",
        default=0.5,
    )
    max_hops_for_graph_pruning: int = Field(
        description="Maximum hops from query entities for graph-aware pruning.",
        default=2,
    )
    graph_pruning_importance_weight: float = Field(
        description="Importance weight for PageRank vs betweenness in graph-aware pruning.",
        default=0.5,
    )
    graph_pruning_distance_weight: float = Field(
        description="Distance weight vs graph importance in graph-aware pruning.",
        default=0.6,
    )
    hybrid_min_reduction_rate: float = Field(
        description="Minimum reduction rate before hybrid strategy triggers the graph-aware stage.",
        default=0.2,
    )
    pruning_token_buffer_ratio: float = Field(
        description="Token buffer ratio for pruning (0.0-1.0).",
        default=0.9,
    )
