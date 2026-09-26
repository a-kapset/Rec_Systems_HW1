"""Статистики разведочного анализа данных goodbooks-10k.

Расчёт сводных характеристик матрицы оценок, распределения оценок, активности
пользователей, популярности книг (long tail, кривая Лоренца, коэффициент Джини),
частоты тегов и проверки хронологического порядка строк. Все расчёты
векторизованы (`numpy.bincount`, групповые агрегаты pandas).
"""

import numpy as np
import pandas as pd

from recsys import config


def summary(ratings: pd.DataFrame) -> pd.Series:
    """Считает общие характеристики матрицы оценок.

    Args:
        ratings: Оценки `user_id, book_id, rating`.

    Returns:
        Series: число оценок, пользователей, книг, заполненность и sparsity
        матрицы «пользователь × книга», среднее и медиана оценки, доля оценок ≥ 4.
    """
    n_ratings = len(ratings)
    n_users = ratings["user_id"].nunique()
    n_books = ratings["book_id"].nunique()
    density = n_ratings / (n_users * n_books)
    return pd.Series(
        {
            "n_ratings": n_ratings,
            "n_users": n_users,
            "n_books": n_books,
            "density": density,
            "sparsity": 1.0 - density,
            "rating_mean": ratings["rating"].mean(),
            "rating_median": ratings["rating"].median(),
            "share_relevant": (ratings["rating"] >= config.RELEVANCE_THRESHOLD).mean(),
        }
    )


def rating_distribution(ratings: pd.DataFrame) -> pd.DataFrame:
    """Считает распределение оценок.

    Args:
        ratings: Оценки `user_id, book_id, rating`.

    Returns:
        DataFrame с индексом `rating` (1–5) и колонками `count`, `share`.
    """
    counts = np.bincount(ratings["rating"], minlength=config.RATING_MAX + 1)
    counts = counts[config.RATING_MIN :]
    index = pd.Index(range(config.RATING_MIN, config.RATING_MAX + 1), name="rating")
    return pd.DataFrame({"count": counts, "share": counts / counts.sum()}, index=index)


def user_counts(ratings: pd.DataFrame) -> pd.Series:
    """Считает число оценок каждого пользователя.

    Args:
        ratings: Оценки `user_id, book_id, rating`.

    Returns:
        Series с индексом `user_id` и числом оценок.
    """
    return ratings.groupby("user_id").size().rename("n_ratings")


def book_counts(ratings: pd.DataFrame) -> pd.Series:
    """Считает число оценок каждой книги.

    Args:
        ratings: Оценки `user_id, book_id, rating`.

    Returns:
        Series с индексом `book_id` и числом оценок.
    """
    return ratings.groupby("book_id").size().rename("n_ratings")


def describe_counts(counts: pd.Series) -> pd.Series:
    """Считает квантили распределения числа оценок.

    Args:
        counts: Число оценок на пользователя или книгу.

    Returns:
        Series: минимум, квантили 1–99 %, максимум, среднее.
    """
    quantiles = counts.quantile([0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    quantiles.index = ["min", "q01", "q10", "q25", "median", "q75", "q90", "q99", "max"]
    return pd.concat([quantiles, pd.Series({"mean": counts.mean()})])


def activity_segment(ratings: pd.DataFrame) -> pd.Series:
    """Определяет сегмент активности каждого пользователя по числу оценок.

    Границы сегментов задаются `config.USER_ACTIVITY_BINS`.

    Args:
        ratings: Оценки `user_id, book_id, rating`.

    Returns:
        Categorical Series с индексом `user_id` и метками `config.USER_ACTIVITY_LABELS`.
    """
    return pd.cut(
        ratings.groupby("user_id").size(),
        bins=list(config.USER_ACTIVITY_BINS),
        labels=list(config.USER_ACTIVITY_LABELS),
        right=False,
    ).rename("segment")


def user_segments(ratings: pd.DataFrame) -> pd.DataFrame:
    """Разбивает пользователей на сегменты по активности.

    Границы сегментов задаются `config.USER_ACTIVITY_BINS`.

    Args:
        ratings: Оценки `user_id, book_id, rating`.

    Returns:
        DataFrame по сегментам: число и доля пользователей, доля оценок,
        средняя оценка.
    """
    per_user = ratings.groupby("user_id")["rating"].agg(["size", "mean"])
    segment = activity_segment(ratings)
    grouped = per_user.groupby(segment, observed=False)
    table = pd.DataFrame(
        {
            "n_users": grouped.size(),
            "rating_sum": grouped["size"].sum(),
            "mean_rating": grouped["mean"].mean(),
        }
    )
    table["share_users"] = table["n_users"] / table["n_users"].sum()
    table["share_ratings"] = table["rating_sum"] / table["rating_sum"].sum()
    table.index.name = "segment"
    return table[["n_users", "share_users", "share_ratings", "mean_rating"]]


def lorenz_curve(counts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Строит кривую Лоренца для распределения числа оценок.

    Args:
        counts: Число оценок на объект.

    Returns:
        Кортеж (доля объектов, накопленная доля оценок), объекты упорядочены
        по возрастанию числа оценок; обе оси начинаются с 0.
    """
    values = np.sort(counts.to_numpy())
    x = np.arange(len(values) + 1) / len(values)
    y = np.concatenate([[0.0], np.cumsum(values) / values.sum()])
    return x, y


def gini(counts: pd.Series) -> float:
    """Считает коэффициент Джини распределения числа оценок.

    Args:
        counts: Число оценок на объект.

    Returns:
        Коэффициент Джини: 0 — равномерное распределение, 1 — все оценки у одного объекта.
    """
    values = np.sort(counts.to_numpy()).astype(np.float64)
    n = len(values)
    ranks = np.arange(1, n + 1)
    return float((2 * ranks - n - 1) @ values / (n * values.sum()))


def popularity_segments(counts: pd.Series) -> pd.Series:
    """Разбивает книги на head, mid и long tail по накопленной доле оценок.

    Head — наиболее популярные книги, вместе дающие `config.HEAD_RATING_SHARE`
    оценок; tail — книги за пределами первых `config.TAIL_RATING_SHARE` оценок.

    Args:
        counts: Число оценок на книгу.

    Returns:
        Categorical Series с индексом `book_id` и значениями `head`, `mid`, `tail`.
    """
    ordered = counts.sort_values(ascending=False, kind="stable")
    share_before = (ordered.cumsum() - ordered) / ordered.sum()
    segment = np.select(
        [share_before < config.HEAD_RATING_SHARE, share_before < config.TAIL_RATING_SHARE],
        ["head", "mid"],
        default="tail",
    )
    result = pd.Series(segment, index=ordered.index, name="segment")
    return result.astype(pd.CategoricalDtype(["head", "mid", "tail"], ordered=True))


def popularity_table(counts: pd.Series) -> pd.DataFrame:
    """Считает сводку по сегментам популярности книг.

    Args:
        counts: Число оценок на книгу.

    Returns:
        DataFrame по сегментам: число и доля книг, доля оценок, диапазон
        числа оценок на книгу.
    """
    segment = popularity_segments(counts)
    grouped = counts.groupby(segment.reindex(counts.index), observed=False)
    table = pd.DataFrame(
        {
            "n_books": grouped.size(),
            "share_books": grouped.size() / len(counts),
            "share_ratings": grouped.sum() / counts.sum(),
            "min_ratings": grouped.min(),
            "max_ratings": grouped.max(),
        }
    )
    table.index.name = "segment"
    return table


def top_share(counts: pd.Series, fractions: tuple[float, ...] = (0.01, 0.1, 0.2)) -> pd.Series:
    """Считает долю оценок, приходящуюся на самые популярные объекты.

    Args:
        counts: Число оценок на объект.
        fractions: Доли самых популярных объектов.

    Returns:
        Series: доля оценок у топ-`fraction` объектов.
    """
    values = np.sort(counts.to_numpy())[::-1]
    cumulative = np.cumsum(values) / values.sum()
    top_n = np.maximum(np.round(np.array(fractions) * len(values)).astype(int), 1)
    return pd.Series(cumulative[top_n - 1], index=[f"top_{f:.0%}" for f in fractions])


def activity_vs_rating(ratings: pd.DataFrame, entity: str, n_bins: int = 10) -> pd.DataFrame:
    """Считает среднюю оценку в квантильных группах по числу оценок.

    Args:
        ratings: Оценки `user_id, book_id, rating`.
        entity: `user_id` или `book_id`.
        n_bins: Число квантильных групп.

    Returns:
        DataFrame по группам: медиана числа оценок и средняя оценка объекта.
    """
    per_entity = ratings.groupby(entity)["rating"].agg(["size", "mean"])
    group = pd.qcut(per_entity["size"], q=n_bins, duplicates="drop")
    grouped = per_entity.groupby(group, observed=True)
    return pd.DataFrame(
        {"median_ratings": grouped["size"].median(), "mean_rating": grouped["mean"].mean()}
    ).reset_index(drop=True)


def chronology_check(ratings: pd.DataFrame) -> pd.Series:
    """Проверяет, что оценки пользователей перемежаются по всему файлу.

    Если строки упорядочены по времени глобально, оценки одного пользователя
    распределены по файлу, а не сгруппированы в один непрерывный блок.

    Args:
        ratings: Оценки в исходном порядке строк.

    Returns:
        Series: признак сортировки по `user_id`, среднее число непрерывных
        блоков на пользователя, медиана размаха позиций оценок пользователя
        в долях от длины файла.
    """
    users = ratings["user_id"].to_numpy()
    n_blocks = int(np.count_nonzero(users[1:] != users[:-1])) + 1
    positions = pd.Series(np.arange(len(users)), index=users)
    bounds = positions.groupby(level=0).agg(["min", "max"])
    span = bounds["max"] - bounds["min"]
    return pd.Series(
        {
            "sorted_by_user": bool(np.all(np.diff(users) >= 0)),
            "blocks_per_user": n_blocks / ratings["user_id"].nunique(),
            "median_span_share": float(span.median() / len(users)),
        }
    )


def tag_frequency(book_tags: pd.DataFrame) -> pd.DataFrame:
    """Считает частоту тегов по книгам.

    Args:
        book_tags: Таблица `book_id, tag_name, count`.

    Returns:
        DataFrame с индексом `tag_name`: `n_books` — число книг с тегом,
        `total_count` — суммарное число присвоений тега; сортировка по
        убыванию `total_count`.
    """
    grouped = book_tags.groupby("tag_name", observed=True)
    table = pd.DataFrame(
        {"n_books": grouped["book_id"].nunique(), "total_count": grouped["count"].sum()}
    )
    return table.sort_values("total_count", ascending=False)
