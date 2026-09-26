"""Оценка top-N моделей на отложенной выборке.

Единый для всех моделей набор оценки: фиксированная случайная выборка
пользователей с хотя бы одной релевантной книгой (оценка ≥ порога) в отложенной
части, матрица релевантных книг, матрица train для исключения уже оценённых книг
и популярность книг в train. Рекомендации формируются батчами, метрики
считаются векторизованно и усредняются по пользователям (macro).
"""

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse

from recsys import config, metrics
from recsys.data import interaction_matrix
from recsys.models.base import Recommender


@dataclass(frozen=True)
class EvalSet:
    """Данные для оценки top-N моделей.

    Attributes:
        user_ids: Идентификаторы оцениваемых пользователей.
        relevant: Бинарная матрица (len(user_ids), N_BOOKS) релевантных книг.
        seen: Матрица (N_USERS, N_BOOKS) оценок train, исключаемых из выдачи.
        item_counts: Число оценок каждой книги в train.
    """

    user_ids: np.ndarray
    relevant: sparse.csr_matrix
    seen: sparse.csr_matrix
    item_counts: np.ndarray

    @property
    def n_relevant(self) -> np.ndarray:
        """Число релевантных книг каждого оцениваемого пользователя."""
        return np.diff(self.relevant.indptr)

    @property
    def n_catalog(self) -> int:
        """Число книг, имеющих оценки в train."""
        return int((self.item_counts > 0).sum())


def build_eval_set(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    n_users: int | None = config.EVAL_N_USERS,
    threshold: int = config.RELEVANCE_THRESHOLD,
    random_state: int = config.RANDOM_STATE,
) -> EvalSet:
    """Формирует набор оценки по разбиению train / holdout.

    Args:
        train: Обучающие оценки.
        holdout: Отложенные оценки (valid или test).
        n_users: Размер случайной выборки пользователей; None — все пользователи.
        threshold: Порог релевантности оценки.
        random_state: Seed выборки пользователей.

    Returns:
        Набор оценки `EvalSet`.
    """
    relevant_all = interaction_matrix(holdout[holdout["rating"] >= threshold], binary=True)
    candidates = np.flatnonzero(np.diff(relevant_all.indptr) > 0) + 1
    if n_users is not None and n_users < candidates.size:
        rng = np.random.default_rng(random_state)
        candidates = np.sort(rng.choice(candidates, size=n_users, replace=False))
    item_counts = np.bincount(train["book_id"].to_numpy() - 1, minlength=config.N_BOOKS)
    return EvalSet(
        user_ids=candidates,
        relevant=relevant_all[candidates - 1],
        seen=interaction_matrix(train, binary=True),
        item_counts=item_counts,
    )


def ranking_metrics(
    recommendations: np.ndarray, eval_set: EvalSet, ks: tuple[int, ...] = config.K_VALUES
) -> pd.DataFrame:
    """Считает метрики top-N по готовым спискам рекомендаций.

    Args:
        recommendations: `book_id` формы (n_users, ≥ max(ks)), -1 — пустая позиция;
            строки выровнены с `eval_set.user_ids`.
        eval_set: Набор оценки.
        ks: Длины списков.

    Returns:
        DataFrame с индексом K и колонками метрик; `short_lists` — доля
        списков короче K.
    """
    items = np.where(recommendations > 0, recommendations - 1, -1)
    hits = metrics.hit_matrix(items, eval_set.relevant)
    n_relevant = eval_set.n_relevant
    rows = {}
    for k in ks:
        rows[k] = {
            "precision": metrics.precision_at_k(hits, k).mean(),
            "recall": metrics.recall_at_k(hits, n_relevant, k).mean(),
            "ndcg": metrics.ndcg_at_k(hits, n_relevant, k).mean(),
            "map": metrics.average_precision_at_k(hits, n_relevant, k).mean(),
            "hit_rate": metrics.hit_rate_at_k(hits, k).mean(),
            "coverage": metrics.coverage_at_k(items, eval_set.n_catalog, k),
            "novelty": metrics.novelty_at_k(items, eval_set.item_counts, k).mean(),
            "mean_popularity": metrics.mean_popularity_at_k(items, eval_set.item_counts, k).mean(),
            "short_lists": (items[:, k - 1] < 0).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("k")


def evaluate(
    model: Recommender, eval_set: EvalSet, ks: tuple[int, ...] = config.K_VALUES
) -> pd.DataFrame:
    """Формирует рекомендации обученной модели и считает метрики top-N.

    Args:
        model: Обученная модель.
        eval_set: Набор оценки.
        ks: Длины списков.

    Returns:
        DataFrame с индексом (model, k), колонками метрик и временем
        формирования рекомендаций `seconds`.
    """
    start = time.perf_counter()
    recommendations = model.recommend(eval_set.user_ids, max(ks), exclude=eval_set.seen)
    seconds = time.perf_counter() - start
    result = ranking_metrics(recommendations, eval_set, ks).assign(seconds=seconds)
    return pd.concat({model.name: result}, names=["model"])


def compare(
    models: list[Recommender], eval_set: EvalSet, ks: tuple[int, ...] = config.K_VALUES
) -> pd.DataFrame:
    """Оценивает несколько обученных моделей на одном наборе.

    Args:
        models: Обученные модели.
        eval_set: Набор оценки.
        ks: Длины списков.

    Returns:
        Объединённая таблица `evaluate` по всем моделям.
    """
    return pd.concat([evaluate(model, eval_set, ks) for model in models])
