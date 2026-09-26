"""Проверка гибрида и симуляции cold start на игрушечных данных."""

from typing import Self

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from recsys import cold_start, config, evaluation
from recsys.models.base import Recommender
from recsys.models.hybrid import Hybrid, normalize_rows


class FixedScores(Recommender):
    """Модель с заданными скорами книг, одинаковыми для всех пользователей."""

    def __init__(self, scores: dict[int, float]) -> None:
        """Задаёт скоры.

        Args:
            scores: Скор по `book_id`; остальные книги — 0.
        """
        self.name = "fixed"
        self._scores = np.zeros(config.N_BOOKS, dtype=np.float32)
        for book_id, value in scores.items():
            self._scores[book_id - 1] = value

    def fit(self, train: pd.DataFrame) -> Self:
        """Не обучается.

        Args:
            train: Не используется.

        Returns:
            Модель.
        """
        return self

    def score(self, user_ids: np.ndarray) -> np.ndarray:
        """Возвращает заданные скоры.

        Args:
            user_ids: Идентификаторы пользователей.

        Returns:
            Матрица (len(user_ids), N_BOOKS).
        """
        return np.tile(self._scores, (len(user_ids), 1))


def test_normalize_rows() -> None:
    scores = np.array([[2.0, -1.0, np.inf, 1.0], [0.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(normalize_rows(scores), [[1.0, 0.0, 0.0, 0.5], [0.0] * 4])


@pytest.fixture
def hybrid() -> Hybrid:
    """Пользователь 1 — 3 понравившиеся книги, пользователь 2 — ни одной; книги ≥ 5 новые."""
    train = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 3, 3, 3],
            "book_id": [1, 2, 3, 1, 2, 3, 4],
            "rating": [5, 4, 4, 2, 5, 5, 5],
        }
    )
    cf = FixedScores({1: 4.0, 2: 2.0, 3: 1.0, 4: 0.5})
    content = FixedScores({4: 1.0, 5: 2.0})
    popularity = FixedScores({3: 10.0, 2: 5.0, 1: 1.0})
    return Hybrid(cf, content, popularity, n0=1.0, new_item_weight=0.5).fit(train)


def test_hybrid_weights_and_switching(hybrid: Hybrid) -> None:
    np.testing.assert_allclose(hybrid.cf_weight(np.array([1, 2, 3])), [0.75, 0.0, 0.75])
    scores = hybrid.score(np.array([1]))[0]
    cf = np.array([1.0, 0.5, 0.25, 0.125])
    popularity = np.array([0.1, 0.5, 1.0, 0.0])
    # Книги 1–4: β · CF + (1 − β) · Pop; книга 5 (новая): β · γ · CB; остальные новые без сходства.
    np.testing.assert_allclose(scores[:4], 0.75 * cf + 0.25 * popularity, rtol=1e-5)
    assert scores[4] == pytest.approx(0.75 * 0.5)
    assert np.isneginf(scores[5:]).all()


def test_hybrid_falls_back_to_popularity(hybrid: Hybrid) -> None:
    top = hybrid.recommend(np.array([2]), k=3)
    np.testing.assert_array_equal(top, [[3, 2, 1]])


def test_truncate_history_and_hide_items() -> None:
    train = pd.DataFrame(
        {"user_id": [1, 2, 1, 1, 2], "book_id": [1, 2, 3, 4, 5], "rating": [5, 4, 3, 2, 1]}
    )
    truncated = cold_start.truncate_history(train, np.array([1]), n=2)
    assert truncated["book_id"].tolist() == [1, 2, 3, 5]
    ratings = pd.DataFrame({"user_id": 1, "book_id": np.arange(1, config.N_BOOKS + 1), "rating": 4})
    visible, hidden = cold_start.hide_items(ratings, share=0.01)
    assert hidden.size == 100
    assert not visible["book_id"].isin(hidden).any()
    assert len(visible) == config.N_BOOKS - 100


def test_new_item_metrics() -> None:
    relevant = sparse.csr_matrix(
        (np.ones(4), ([0, 0, 1, 2], [0, 4, 5, 1])), shape=(3, config.N_BOOKS)
    )
    eval_set = evaluation.EvalSet(
        np.array([1, 2, 3]), relevant, relevant, np.ones(config.N_BOOKS, dtype=np.int64)
    )
    recs = np.array([[5, 1], [7, 9], [2, -1]])
    result = cold_start.new_item_metrics(recs, eval_set, np.array([5, 6, 7]), k=2)
    assert result["users_with_new"] == 2
    assert result["exposure"] == pytest.approx(2 / 5)
    assert result["recall_new"] == pytest.approx(0.5)
    assert result["hit_rate_new"] == pytest.approx(0.5)
