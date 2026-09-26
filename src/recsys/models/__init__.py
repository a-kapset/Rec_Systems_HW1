"""Модели рекомендаций с общим интерфейсом `Recommender`.

Popularity и Content-Based; Item-based CF, SVD и гибрид добавляются в
отдельных модулях пакета.
"""

from recsys.models.base import Recommender
from recsys.models.content import ContentBased
from recsys.models.popularity import Popularity

__all__ = ["ContentBased", "Popularity", "Recommender"]
