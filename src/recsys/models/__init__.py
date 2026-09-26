"""Модели рекомендаций с общим интерфейсом `Recommender`.

Popularity, Content-Based, Item-based CF, FunkSVD, PureSVD и гибрид CF и Content-Based
для cold start.
"""

from recsys.models.base import Recommender
from recsys.models.content import ContentBased
from recsys.models.hybrid import Hybrid
from recsys.models.item_cf import ItemCF
from recsys.models.popularity import Popularity
from recsys.models.svd import FunkSVD, PureSVD

__all__ = ["ContentBased", "FunkSVD", "Hybrid", "ItemCF", "Popularity", "PureSVD", "Recommender"]
