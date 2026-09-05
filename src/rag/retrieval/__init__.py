"""The retrieval pipeline: a query in, ranked context with provenance out.

Knows nothing about ingestion. It shares only the domain contracts, the
storage interface, and the model-loading helpers with the other pipeline.
"""

from rag.retrieval.interfaces import (
    BasePromptAugmenter,
    BaseQueryEmbedder,
    BaseRanker,
    BaseReranker,
)
from rag.retrieval.orchestrator import RetrievalOrchestrator
from rag.retrieval.prompting.augmenter import (
    PromptTemplateNotFoundError,
    TemplatePromptAugmenter,
)
from rag.retrieval.ranking.cosine import CosineSimilarityRanker
from rag.retrieval.reranking.cross_encoder import CrossEncoderReranker
from rag.retrieval.reranking.passthrough import PassthroughReranker

__all__ = [
    "BasePromptAugmenter",
    "BaseQueryEmbedder",
    "BaseRanker",
    "BaseReranker",
    "CosineSimilarityRanker",
    "CrossEncoderReranker",
    "PassthroughReranker",
    "PromptTemplateNotFoundError",
    "RetrievalOrchestrator",
    "TemplatePromptAugmenter",
]
