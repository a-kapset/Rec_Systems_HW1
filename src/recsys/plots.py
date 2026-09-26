"""Визуализация результатов EDA и сравнения моделей.

Единый стиль графиков (палитра, сетка, подписи), функции построения отдельных
графиков и сохранение в `reports/figures`. Функции принимают заранее
рассчитанные статистики из `recsys.eda` и возвращают `matplotlib.figure.Figure`.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter

from recsys import config, eda

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#a3a29c"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def set_style() -> None:
    """Устанавливает единый стиль графиков matplotlib."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "text.color": TEXT_PRIMARY,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def save_figure(fig: Figure, name: str, figures_dir: Path = config.FIGURES_DIR) -> Path:
    """Сохраняет график в PNG.

    Args:
        fig: График.
        name: Имя файла без расширения.
        figures_dir: Каталог для сохранения.

    Returns:
        Путь к сохранённому файлу.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    return path


def _thousands(value: float, _: int) -> str:
    """Форматирует число с пробелом в качестве разделителя разрядов.

    Args:
        value: Значение деления оси.
        _: Позиция деления (не используется).

    Returns:
        Строковое представление числа.
    """
    return f"{value:,.0f}".replace(",", " ")


def plot_rating_distribution(distribution: pd.DataFrame) -> Figure:
    """Строит столбчатую диаграмму распределения оценок.

    Args:
        distribution: Результат `eda.rating_distribution`.

    Returns:
        График.
    """
    fig, ax = plt.subplots(figsize=(7, 4.2))
    relevant = distribution.index >= config.RELEVANCE_THRESHOLD
    colors = np.where(relevant, SERIES[0], NEUTRAL)
    bars = ax.bar(distribution.index, distribution["count"], color=colors, width=0.7)
    for bar, share in zip(bars, distribution["share"], strict=True):
        ax.annotate(
            f"{share:.1%}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            color=TEXT_SECONDARY,
        )
    ax.bar(0, 0, color=SERIES[0], label=f"релевантные (≥ {config.RELEVANCE_THRESHOLD})")
    ax.bar(0, 0, color=NEUTRAL, label=f"нерелевантные (< {config.RELEVANCE_THRESHOLD})")
    ax.set_xticks(distribution.index)
    ax.set_xlim(0.4, 5.6)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("Оценка")
    ax.set_ylabel("Число оценок")
    ax.set_title("Распределение оценок")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
    return fig


def plot_user_activity(counts: pd.Series) -> Figure:
    """Строит гистограмму числа оценок на пользователя с границами сегментов.

    Args:
        counts: Число оценок на пользователя (`eda.user_counts`).

    Returns:
        График.
    """
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.arange(0, counts.max() + 6, 5)
    ax.hist(counts, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.8)
    for edge in config.USER_ACTIVITY_BINS[1:-1]:
        ax.axvline(edge, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    median = counts.median()
    ax.axvline(median, color=SERIES[1], linewidth=1.5, label=f"медиана = {median:.0f}")
    edges = [*config.USER_ACTIVITY_BINS[:-1], counts.max() + 1]
    y_top = ax.get_ylim()[1]
    for left, right, label in zip(edges[:-1], edges[1:], config.USER_ACTIVITY_LABELS, strict=True):
        share = counts.between(left, right - 1).mean()
        ax.text(
            (left + right) / 2,
            y_top * 0.96,
            f"{label}\n{share:.1%}",
            ha="center",
            va="top",
            color=TEXT_SECONDARY,
            fontsize=9,
        )
    ax.set_ylim(0, y_top * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax.set_xlabel("Число оценок пользователя")
    ax.set_ylabel("Число пользователей")
    ax.set_title("Активность пользователей: число пользователей vs число оценок")
    ax.legend(loc="center right")
    return fig


def plot_book_popularity(counts: pd.Series) -> Figure:
    """Строит распределение популярности книг и ранговый график long tail.

    Левая панель — число книг в зависимости от числа оценок (лог-бины,
    log-log); правая — число оценок книги по рангу популярности с границами
    сегментов head / mid / tail.

    Args:
        counts: Число оценок на книгу (`eda.book_counts`).

    Returns:
        График.
    """
    fig, (ax_hist, ax_rank) = plt.subplots(1, 2, figsize=(12, 4.4))

    bins = np.logspace(np.log10(counts.min()), np.log10(counts.max() + 1), 40)
    ax_hist.hist(counts, bins=bins, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
    ax_hist.set_xscale("log")
    ax_hist.set_yscale("log")
    ax_hist.set_xlabel("Число оценок книги (лог. шкала)")
    ax_hist.set_ylabel("Число книг (лог. шкала)")
    ax_hist.set_title("Число книг vs число оценок")

    ordered = np.sort(counts.to_numpy())[::-1]
    ranks = np.arange(1, len(ordered) + 1)
    ax_rank.plot(ranks, ordered, color=SERIES[0])
    ax_rank.set_xscale("log")
    ax_rank.set_yscale("log")
    segments = eda.popularity_segments(counts).value_counts().reindex(["head", "mid", "tail"])
    bounds = np.cumsum(segments.to_numpy())
    starts = np.concatenate([[1], bounds[:-1] + 1])
    for bound in bounds[:-1]:
        ax_rank.axvline(bound, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
    y_text = ordered.max()
    for start, end, name in zip(starts, bounds, segments.index, strict=True):
        ax_rank.text(
            np.sqrt(start * end), y_text, name, ha="center", va="top", color=TEXT_SECONDARY
        )
    shares = eda.popularity_table(counts)
    lines = [
        f"{name}: {_thousands(row.n_books, 0)} книг ({row.share_books:.0%}), "
        f"{row.share_ratings:.0%} оценок"
        for name, row in shares.iterrows()
    ]
    ax_rank.text(
        0.03, 0.04, "\n".join(lines),
        transform=ax_rank.transAxes, va="bottom", color=TEXT_SECONDARY, fontsize=9,
    )  # fmt: skip
    ax_rank.set_xlabel("Ранг книги по числу оценок (лог. шкала)")
    ax_rank.set_ylabel("Число оценок (лог. шкала)")
    ax_rank.set_title("Long tail: популярность книг по рангу")
    fig.tight_layout()
    return fig


def plot_lorenz(book_counts: pd.Series, user_counts: pd.Series) -> Figure:
    """Строит кривые Лоренца для книг и пользователей.

    Args:
        book_counts: Число оценок на книгу.
        user_counts: Число оценок на пользователя.

    Returns:
        График.
    """
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, linestyle="--", linewidth=1, label="равномерное")
    for counts, name, color in (
        (book_counts, "книги", SERIES[0]),
        (user_counts, "пользователи", SERIES[1]),
    ):
        x, y = eda.lorenz_curve(counts)
        ax.plot(x, y, color=color, label=f"{name}, Gini = {eda.gini(counts):.2f}")
    top10 = eda.top_share(book_counts, (0.1,)).iloc[0]
    ax.plot(0.9, 1 - top10, "o", color=SERIES[0], markersize=8, markeredgecolor=SURFACE)
    ax.annotate(
        "", (0.93, 1.0), xytext=(0.93, 1 - top10),
        arrowprops={"arrowstyle": "<->", "color": TEXT_SECONDARY, "linewidth": 0.8},
    )  # fmt: skip
    ax.text(
        0.91, 1 - top10 / 2, f"топ-10 % книг —\n{top10:.0%} оценок",
        ha="right", va="center", color=TEXT_SECONDARY,
        bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1},
    )  # fmt: skip
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Доля объектов (по возрастанию числа оценок)")
    ax.set_ylabel("Накопленная доля оценок")
    ax.set_title("Кривая Лоренца: концентрация оценок")
    ax.legend(loc="upper left")
    return fig


def plot_top_tags(
    raw_frequency: pd.DataFrame,
    service_mask: pd.Series,
    clean_frequency: pd.DataFrame,
    n_raw: int = 20,
    n_clean: int = 30,
) -> Figure:
    """Строит частоту тегов до и после очистки.

    Args:
        raw_frequency: Частота исходных тегов (`eda.tag_frequency`).
        service_mask: Признак служебного тега, индекс — `tag_name`.
        clean_frequency: Частота тегов после очистки.
        n_raw: Число исходных тегов на левой панели.
        n_clean: Число очищенных тегов на правой панели.

    Returns:
        График.
    """
    fig, (ax_raw, ax_clean) = plt.subplots(
        1, 2, figsize=(13, 8), gridspec_kw={"width_ratios": [1, 1.1]}
    )
    raw = raw_frequency.head(n_raw).iloc[::-1]
    colors = np.where(service_mask.reindex(raw.index).to_numpy(), SERIES[1], SERIES[0])
    ax_raw.barh(raw.index, raw["total_count"], color=colors, height=0.7)
    ax_raw.barh([raw.index[0]], [0], color=SERIES[1], label="служебный тег")
    ax_raw.barh([raw.index[0]], [0], color=SERIES[0], label="содержательный тег")
    ax_raw.set_xscale("log")
    ax_raw.set_xlabel("Суммарное число присвоений (лог. шкала)")
    ax_raw.set_title(f"Топ-{n_raw} исходных тегов")
    ax_raw.legend(loc="lower right")

    clean = clean_frequency.head(n_clean).iloc[::-1]
    ax_clean.barh(clean.index, clean["total_count"], color=SERIES[0], height=0.7)
    ax_clean.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}"))
    ax_clean.set_xlabel("Суммарное число присвоений, млн")
    ax_clean.set_title(f"Топ-{n_clean} тегов после очистки")
    for ax in (ax_raw, ax_clean):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    return fig


def plot_activity_vs_rating(users: pd.DataFrame, books: pd.DataFrame) -> Figure:
    """Строит зависимость средней оценки от активности пользователя и популярности книги.

    Args:
        users: Результат `eda.activity_vs_rating` для пользователей.
        books: Результат `eda.activity_vs_rating` для книг.

    Returns:
        График.
    """
    fig, (ax_users, ax_books) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    ax_users.plot(users["median_ratings"], users["mean_rating"], marker="o", markersize=7)
    ax_users.set_xlabel("Медиана числа оценок пользователя в децильной группе")
    ax_users.set_ylabel("Средняя оценка")
    ax_users.set_title("Средняя оценка пользователя vs активность")
    ax_books.plot(books["median_ratings"], books["mean_rating"], marker="o", markersize=7)
    ax_books.set_xscale("log")
    ax_books.set_xlabel("Медиана числа оценок книги в децильной группе (лог. шкала)")
    ax_books.set_title("Средняя оценка книги vs популярность")
    fig.tight_layout()
    return fig


def plot_popularity_threshold(grid: pd.DataFrame, count_ndcg: float, chosen: int) -> Figure:
    """Строит зависимость nDCG@10 модели Popularity от порога числа оценок.

    Левая панель — nDCG@10 вариантов `mean` и `weighted` и уровень варианта
    `count`; правая — число книг, допущенных к рекомендации вариантом `mean`.

    Args:
        grid: Таблица `method`, `min_ratings`, `ndcg`, `n_candidates`.
        count_ndcg: nDCG@10 варианта `count`.
        chosen: Выбранный порог варианта `mean`.

    Returns:
        График.
    """
    fig, (ax_metric, ax_pool) = plt.subplots(1, 2, figsize=(12, 4.2))
    labels = {"mean": "средний рейтинг с порогом m", "weighted": "weighted rating, параметр m"}
    for (method, label), color in zip(labels.items(), SERIES, strict=False):
        part = grid[(grid["method"] == method) & (grid["min_ratings"] > 0)]
        ax_metric.plot(part["min_ratings"], part["ndcg"], marker="o", color=color, label=label)
    ax_metric.axhline(count_ndcg, color=NEUTRAL, linestyle="--", label="число оценок (count)")
    ax_metric.axvline(chosen, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    ax_metric.set_xscale("log")
    ax_metric.set_xlabel("m (лог. шкала)")
    ax_metric.set_ylabel("nDCG@10 на valid")
    ax_metric.set_title("Popularity: качество vs порог")
    ax_metric.legend(loc="lower right")

    mean = grid[(grid["method"] == "mean") & (grid["min_ratings"] > 0)]
    ax_pool.plot(mean["min_ratings"], mean["n_candidates"], marker="o", color=SERIES[0])
    ax_pool.axvline(chosen, color=TEXT_SECONDARY, linestyle=":", linewidth=1)
    ax_pool.set_xscale("log")
    ax_pool.set_yscale("log")
    ax_pool.yaxis.set_major_formatter(FuncFormatter(_thousands))
    ax_pool.set_xlabel("Порог m (лог. шкала)")
    ax_pool.set_ylabel("Число книг с ≥ m оценками (лог. шкала)")
    ax_pool.set_title("Размер пула кандидатов")
    fig.tight_layout()
    return fig


def plot_item_cf_grid(grid: pd.DataFrame, shrinkage: float) -> Figure:
    """Строит зависимость RMSE и nDCG@10 Item-based CF от числа соседей K.

    Линии — меры схожести при фиксированном λ.

    Args:
        grid: Таблица `similarity`, `shrinkage`, `k`, `rmse`, `ndcg`.
        shrinkage: Значение λ для отображения.

    Returns:
        График.
    """
    fig, (ax_rmse, ax_ndcg) = plt.subplots(1, 2, figsize=(12, 4.2))
    part = grid[grid["shrinkage"] == shrinkage]
    for (similarity, group), color in zip(part.groupby("similarity"), SERIES, strict=False):
        group = group.sort_values("k")
        ax_rmse.plot(group["k"], group["rmse"], marker="o", color=color, label=similarity)
        ax_ndcg.plot(group["k"], group["ndcg"], marker="o", color=color, label=similarity)
    for ax, ylabel, title in (
        (ax_rmse, "RMSE на valid", "Ошибка предсказания оценки"),
        (ax_ndcg, "nDCG@10 на valid", "Качество top-10"),
    ):
        ax.set_xscale("log")
        ax.set_xticks(sorted(part["k"].unique()))
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.set_xlabel("Число соседей K (лог. шкала)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}, λ = {shrinkage:g}")
        ax.legend()
    fig.tight_layout()
    return fig
