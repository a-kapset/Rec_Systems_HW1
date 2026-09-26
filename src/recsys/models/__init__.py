"""Модели рекомендаций с общим интерфейсом `Recommender`.

Popularity, Content-Based и Item-based CF; SVD и гибрид добавляются в
отдельных модулях пакета.
"""

from recsys.models.base import Recommender
from recsys.models.content import ContentBased
from recsys.models.item_cf import ItemCF
from recsys.models.popularity import Popularity

__all__ = ["ContentBased", "ItemCF", "Popularity", "Recommender"]
