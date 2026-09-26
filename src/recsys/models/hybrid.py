"""Гибридная модель для cold start: CF, Popularity и Content-Based.

Для пользователя скор книги — выпуклая комбинация нормированных скоров CF-модели
и Popularity с весом CF, растущим с числом понравившихся пользователю книг:
β(n) = n / (n + n0). При короткой истории выдача ближе к Popularity, при длинной —
к CF. Для книг без оценок в train CF-скор не определён и заменяется скором
Content-Based с понижающим весом γ (switching по книгам): нормированный
контентный скор распределён плотнее CF-скора, и без веса новые книги
вытесняют известные.
"""

from typing import Self

import numpy as np
import pandas as pd

from recsys import config
from recsys.models.base import Recommender


def normalize_rows(scores: np.ndarray) -> np.ndarray:
    """Нормирует неотрицательную часть скоров на максимум строки.

    Нечисловые и отрицательные скоры заменяются нулём; строка из нулей остаётся нулевой.

    Args:
        scores: Скоры формы (n_users, n_items).

    Returns:
        Скоры в [0, 1] float32.
    """
    scores = np.where(np.isfinite(scores), scores, 0.0).clip(min=0.0).astype(np.float32)
    row_max = scores.max(axis=1, keepdims=True)
    return np.divide(scores, row_max, out=np.zeros_like(scores), where=row_max > 0)


def min_max(scores: np.ndarray) -> np.ndarray:
    """Приводит конечные скоры к [0, 1] линейным преобразованием; нечисловые — к 0.

    Args:
        scores: Одномерный массив скоров.

    Returns:
        Скоры в [0, 1] float32.
    """
    finite = np.isfinite(scores)
    if not finite.any():
        return np.zeros(scores.shape, dtype=np.float32)
    low, high = scores[finite].min(), scores[finite].max()
    scaled = (scores - low) / (high - low) if high > low else np.ones_like(scores)
    return np.where(finite, scaled, 0.0).astype(np.float32)


class Hybrid(Recommender):
    """Гибрид CF и Popularity по длине истории с переключением на Content-Based для новых книг.

    score(u, i) = β_u · X(u, i) + (1 − β_u) · Pop(i), где X — нормированный скор CF
    для книг с не менее чем `min_item_ratings` оценками в train и нормированный скор
    Content-Based, умноженный на γ = `new_item_weight`, для остальных, Pop —
    популярность, приведённая к [0, 1], β_u = n_u / (n_u + n0), n_u — число книг
    пользователя с оценкой не ниже порога релевантности в train. При n_u = 0
    выдача совпадает с Popularity.

    Attributes:
        cf: CF-модель с неотрицательными скорами (Item-based CF).
        content: Content-Based модель для новых книг.
        popularity: Модель Popularity.
        n0: Число понравившихся книг, при котором веса CF и Popularity равны.
        min_item_ratings: Минимальное число оценок книги в train для CF-скора.
        new_item_weight: Вес γ контентного скора новых книг.
        n_liked: Число понравившихся книг каждого пользователя в train.
        item_counts: Число оценок каждой книги в train.
    """

    def __init__(
        self,
        cf: Recommender,
        content: Recommender,
        popularity: Recommender,
        n0: float = config.HYBRID_N0,
        min_item_ratings: int = config.HYBRID_MIN_ITEM_RATINGS,
        new_item_weight: float = config.HYBRID_NEW_ITEM_WEIGHT,
    ) -> None:
        """Задаёт компоненты и параметры гибрида.

        Args:
            cf: CF-модель с неотрицательными скорами.
            content: Content-Based модель для новых книг.
            popularity: Модель Popularity.
            n0: Параметр веса β(n) = n / (n + n0); 0 — только CF при n > 0.
            min_item_ratings: Минимальное число оценок книги в train для CF-скора.
            new_item_weight: Вес γ контентного скора новых книг.
        """
        self.cf = cf
        self.content = content
        self.popularity = popularity
        self.n0 = n0
        self.min_item_ratings = min_item_ratings
        self.new_item_weight = new_item_weight
        self.n_liked: np.ndarray | None = None
        self.item_counts: np.ndarray | None = None

    @property
    def name(self) -> str:
        """Название модели с параметрами."""
        return f"Hybrid[n0={self.n0:g}, γ={self.new_item_weight:g}]"

    def fit(self, train: pd.DataFrame, fit_components: bool = True) -> Self:
        """Обучает компоненты и считает активность пользователей и популярность книг.

        Args:
            train: Обучающие оценки `user_id, book_id, rating`.
            fit_components: Обучать ли компоненты; False — компоненты уже обучены на `train`.

        Returns:
            Обученная модель.
        """
        if fit_components:
            for model in (self.cf, self.content, self.popularity):
                model.fit(train)
        liked = train.loc[train["rating"] >= config.RELEVANCE_THRESHOLD, "user_id"].to_numpy()
        self.n_liked = np.bincount(liked - 1, minlength=config.N_USERS)
        self.item_counts = np.bincount(train["book_id"].to_numpy() - 1, minlength=config.N_BOOKS)
        return self

    def set_params(self, n0: float | None = None, new_item_weight: float | None = None) -> Self:
        """Меняет параметры весов без переобучения компонентов.

        Args:
            n0: Новое значение n0; None — без изменений.
            new_item_weight: Новое значение γ; None — без изменений.

        Returns:
            Модель с новыми параметрами.
        """
        if n0 is not None:
            self.n0 = n0
        if new_item_weight is not None:
            self.new_item_weight = new_item_weight
        return self

    def cf_weight(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает вес CF β_u = n_u / (n_u + n0).

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Веса формы (len(user_ids),); при n_u = 0 — 0.
        """
        n = self.n_liked[np.asarray(user_ids) - 1].astype(np.float32)
        return np.divide(n, n + self.n0, out=np.zeros_like(n), where=n > 0)

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Считает скоры гибрида для батча пользователей.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS) float32; `-inf` — новая книга с нулевым
            контентным сходством с профилем пользователя.
        """
        new_items = self.item_counts < self.min_item_ratings
        personal = normalize_rows(self.cf.score(user_ids))
        if new_items.any():
            content = normalize_rows(self.content.score(user_ids))
            personal[:, new_items] = self.new_item_weight * content[:, new_items]
        popularity = min_max(np.asarray(self.popularity.score(np.asarray(user_ids)[:1]))[0])
        beta = self.cf_weight(user_ids)[:, None]
        scores = beta * personal + (1.0 - beta) * popularity
        return np.where((scores > 0) | ~new_items, scores, -np.inf).astype(np.float32)
