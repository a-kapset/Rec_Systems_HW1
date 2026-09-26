"""Конфигурация проекта: пути, источник данных и глобальные константы.

Единая точка задания путей к данным и отчётам, параметров воспроизводимости
и протокола оценки. Остальные модули импортируют значения отсюда.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
ARTIFACTS_DIR: Path = PROJECT_ROOT / "artifacts"
FIGURES_DIR: Path = PROJECT_ROOT / "reports" / "figures"

RANDOM_STATE: int = 42

DATASET_REPO_URL: str = "https://github.com/zygmuntz/goodbooks-10k"
DATASET_RAW_URL: str = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"

# Ожидаемое число строк (без заголовка) в актуальной версии датасета.
DATASET_FILES: dict[str, int] = {
    "ratings.csv": 5_976_479,
    "books.csv": 10_000,
    "tags.csv": 34_252,
    "book_tags.csv": 999_912,
}

RATING_MIN: int = 1
RATING_MAX: int = 5
N_BOOKS: int = 10_000
N_USERS: int = 53_424

RELEVANCE_THRESHOLD: int = 4
TEST_FRACTION: float = 0.2
K_VALUES: tuple[int, ...] = (5, 10, 20)
