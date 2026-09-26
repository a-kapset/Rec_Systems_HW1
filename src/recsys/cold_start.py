"""Симуляция cold start пользователей и книг на обучающей выборке.

Cold start пользователя моделируется усечением истории: у выбранных пользователей
в train остаются первые n оценок по времени, остальные удаляются, и модели
обучаются на усечённых данных. Cold start книги — удаление всех оценок части
каталога из train: такие книги неизвестны CF-моделям, но имеют метаданные.
"""

import dataclasses

import numpy as np
import pandas as pd

from recsys import config, evaluation, metrics
from recsys.evaluation import EvalSet
from recsys.models import ContentBased, Hybrid, ItemCF, Popularity, PureSVD, Recommender


def sample_users(
    user_ids: np.ndarray,
    n_users: int = config.COLD_N_USERS,
    random_state: int = config.RANDOM_STATE,
) -> np.ndarray:
    """Выбирает случайное подмножество пользователей.

    Args:
        user_ids: Идентификаторы-кандидаты.
        n_users: Размер выборки.
        random_state: Seed выборки.

    Returns:
        Отсортированные идентификаторы.
    """
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(user_ids, size=min(n_users, len(user_ids)), replace=False))


def truncate_history(train: pd.DataFrame, user_ids: np.ndarray, n: int) -> pd.DataFrame:
    """Оставляет у выбранных пользователей только первые n оценок train.

    Args:
        train: Обучающие оценки в хронологическом порядке строк.
        user_ids: Пользователи с усекаемой историей.
        n: Длина оставляемой истории.

    Returns:
        Оценки train с усечённой историей выбранных пользователей.
    """
    position = train.groupby("user_id", sort=False).cumcount().to_numpy()
    keep = ~train["user_id"].isin(user_ids).to_numpy() | (position < n)
    return train[keep]


def hide_items(
    train: pd.DataFrame,
    share: float = config.COLD_ITEM_SHARE,
    random_state: int = config.RANDOM_STATE,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Удаляет из train все оценки случайной доли книг каталога.

    Args:
        train: Обучающие оценки.
        share: Доля скрываемых книг.
        random_state: Seed выборки книг.

    Returns:
        Кортеж: оценки train без скрытых книг и отсортированные `book_id` скрытых книг.
    """
    rng = np.random.default_rng(random_state)
    hidden = np.sort(
        rng.choice(
            np.arange(1, config.N_BOOKS + 1), size=int(share * config.N_BOOKS), replace=False
        )
    )
    return train[~train["book_id"].isin(hidden)], hidden


def with_train(eval_set: EvalSet, train: pd.DataFrame) -> EvalSet:
    """Заменяет популярность книг набора оценки на рассчитанную по другой обучающей выборке.

    Исключаемые из выдачи книги (`seen`) и релевантные книги не меняются: цель
    оценки остаётся той же, что и без симуляции.

    Args:
        eval_set: Исходный набор оценки.
        train: Обучающие оценки симуляции.

    Returns:
        Набор оценки с `item_counts` по `train`.
    """
    counts = np.bincount(train["book_id"].to_numpy() - 1, minlength=config.N_BOOKS)
    return dataclasses.replace(eval_set, item_counts=counts)


def new_item_metrics(
    recommendations: np.ndarray, eval_set: EvalSet, new_books: np.ndarray, k: int = 10
) -> pd.Series:
    """Считает качество рекомендаций скрытых (новых) книг.

    Args:
        recommendations: Списки `book_id`, строки выровнены с `eval_set.user_ids`.
        eval_set: Набор оценки.
        new_books: `book_id` новых книг.
        k: Длина списка.

    Returns:
        Series: `exposure` — доля позиций top-K с новыми книгами; `recall_new` —
        средний Recall@K по новым книгам среди пользователей, у которых есть
        релевантные новые книги; `hit_rate_new` — доля таких пользователей
        с попаданием в новую книгу; `users_with_new` — их число.
    """
    top = recommendations[:, :k]
    is_new = np.isin(top, new_books)
    new_relevant = eval_set.relevant[:, new_books - 1]
    n_new_relevant = np.diff(new_relevant.indptr)
    users = n_new_relevant > 0
    items = np.where(is_new, top - 1, -1)
    hits = metrics.hit_matrix(items[users], eval_set.relevant[users])
    return pd.Series(
        {
            "exposure": is_new.sum() / max((top > 0).sum(), 1),
            "recall_new": metrics.recall_at_k(hits, n_new_relevant[users], k).mean(),
            "hit_rate_new": metrics.hit_rate_at_k(hits, k).mean(),
            "users_with_new": int(users.sum()),
        }
    )


def fit_models(
    train: pd.DataFrame, books: pd.DataFrame, book_tags: pd.DataFrame, pure_svd: bool = True
) -> dict[str, Recommender]:
    """Обучает модели сравнения cold start с параметрами из `config`.

    Args:
        train: Обучающие оценки.
        books: Метаданные книг.
        book_tags: Очищенные теги книг.
        pure_svd: Обучать ли PureSVD.

    Returns:
        Модели по отображаемому названию: Popularity (weighted), Content-Based,
        Item-based CF, PureSVD (опционально), Hybrid на тех же компонентах.
    """
    models: dict[str, Recommender] = {
        "Popularity (weighted)": Popularity("weighted", config.POPULARITY_WR_M).fit(train),
        "Content-Based": ContentBased(books, book_tags).fit(train),
        "Item-based CF": ItemCF().fit(train),
    }
    if pure_svd:
        models["PureSVD"] = PureSVD(config.PURE_SVD_N_FACTORS).fit(train)
    models["Hybrid"] = Hybrid(
        models["Item-based CF"], models["Content-Based"], models["Popularity (weighted)"]
    ).fit(train, fit_components=False)
    return models


def evaluate_models(
    models: dict[str, Recommender], eval_set: EvalSet, n0_grid: tuple[float, ...] = (), k: int = 10
) -> pd.DataFrame:
    """Считает метрики top-K моделей; гибрид — при каждом n0 из сетки.

    Args:
        models: Обученные модели; гибрид — под ключом `Hybrid`.
        eval_set: Набор оценки.
        n0_grid: Значения n0 гибрида; пусто — только текущее значение.
        k: Длина списка.

    Returns:
        DataFrame метрик при K с индексом `model`; варианты гибрида подписаны
        `Hybrid (n0=…)`.
    """
    variants = dict(models)
    hybrid = variants.pop("Hybrid", None)
    recommendations, _ = evaluation.recommend_all(variants, eval_set, k)
    if hybrid is not None:
        initial_n0 = hybrid.n0
        for n0 in n0_grid or (initial_n0,):
            label = f"Hybrid (n0={n0:g})" if n0_grid else "Hybrid"
            recommendations[label] = hybrid.set_params(n0=n0).recommend(
                eval_set.user_ids, k, exclude=eval_set.seen
            )
        hybrid.set_params(n0=initial_n0)
    table = evaluation.compare_recommendations(recommendations, eval_set, ks=(k,))
    return table.xs(k, level="k")


def evaluate_histories(
    train: pd.DataFrame,
    eval_set: EvalSet,
    user_ids: np.ndarray,
    books: pd.DataFrame,
    book_tags: pd.DataFrame,
    lengths: tuple[int, ...] = config.COLD_HISTORY_LENGTHS,
    n0_grid: tuple[float, ...] = (),
    pure_svd: bool = True,
) -> pd.DataFrame:
    """Оценивает модели при усечённой до n оценок истории выбранных пользователей.

    Для каждого n модели обучаются заново на train с усечённой историей; цель
    оценки — релевантные книги отложенной части тех же пользователей.

    Args:
        train: Полные обучающие оценки.
        eval_set: Набор оценки, содержащий `user_ids`.
        user_ids: Пользователи с усекаемой историей.
        books: Метаданные книг.
        book_tags: Очищенные теги книг.
        lengths: Длины истории.
        n0_grid: Значения n0 гибрида.
        pure_svd: Включать ли PureSVD.

    Returns:
        DataFrame метрик при K = 10 с индексом (model, history).
    """
    base = evaluation.subset(eval_set, user_ids)
    tables = {}
    for n in lengths:
        truncated = truncate_history(train, user_ids, n)
        models = fit_models(truncated, books, book_tags, pure_svd)
        tables[n] = evaluate_models(models, with_train(base, truncated), n0_grid)
    table = pd.concat(tables, names=["history"])
    return table.swaplevel()


def evaluate_new_item_weights(
    models: dict[str, Recommender],
    eval_set: EvalSet,
    new_books: np.ndarray,
    weights: tuple[float, ...],
    k: int = 10,
) -> pd.DataFrame:
    """Считает общие метрики и метрики новых книг; гибрид — при каждом весе γ.

    Args:
        models: Модели, обученные на train без оценок новых книг; гибрид — под ключом `Hybrid`.
        eval_set: Набор оценки; релевантные книги включают новые.
        new_books: `book_id` новых книг.
        weights: Значения веса γ контентного скора новых книг.
        k: Длина списка.

    Returns:
        DataFrame с индексом `model`: ndcg, recall, hit_rate, coverage и метрики
        `new_item_metrics`; варианты гибрида подписаны `Hybrid (γ=…)`.
    """
    variants = dict(models)
    hybrid = variants.pop("Hybrid")
    recommendations, _ = evaluation.recommend_all(variants, eval_set, k)
    initial = hybrid.new_item_weight
    for weight in weights:
        recommendations[f"Hybrid (γ={weight:g})"] = hybrid.set_params(
            new_item_weight=weight
        ).recommend(eval_set.user_ids, k, exclude=eval_set.seen)
    hybrid.set_params(new_item_weight=initial)
    overall = evaluation.compare_recommendations(recommendations, eval_set, ks=(k,)).xs(
        k, level="k"
    )
    new = pd.DataFrame(
        {
            name: new_item_metrics(recs, eval_set, new_books, k)
            for name, recs in recommendations.items()
        }
    ).T
    return pd.concat([overall[["ndcg", "recall", "hit_rate", "coverage"]], new], axis=1)
