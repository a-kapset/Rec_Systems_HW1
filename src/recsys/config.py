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

# Сегменты пользователей по числу оценок: границы [left, right).
USER_ACTIVITY_BINS: tuple[int, ...] = (0, 50, 100, 150, 10_000)
USER_ACTIVITY_LABELS: tuple[str, ...] = ("<50", "50–99", "100–149", "≥150")

# Сегменты книг по популярности: head — книги, дающие первые 50 % оценок,
# tail — книги за пределами первых 80 % оценок, mid — остальные.
HEAD_RATING_SHARE: float = 0.5
TAIL_RATING_SHARE: float = 0.8

# Служебные теги (статус чтения, владение, формат, личные оценки и списки).
STOP_TAGS: frozenset[str] = frozenset(
    {
        "abandoned", "all-time-favorites", "audible", "audio", "bookclub", "book-club",
        "books", "bookshelf", "borrowed", "calibre", "currently-reading", "default",
        "did-not-finish", "didn-t-finish", "dnf", "english", "favorite", "favorites",
        "favourite", "favourites", "finished", "have", "i-own", "library", "maybe",
        "must-read", "nook", "on-hold", "own", "owned", "own-it", "paperback",
        "hardcover", "re-read", "reread", "recommended", "reviewed", "series",
        "tbr", "unfinished", "unread", "wish-list", "wishlist", "part-of-a-series",
    }
)  # fmt: skip

# Шаблоны служебных тегов: годы, личные полки, форматы, статусы, звёзды.
STOP_TAG_PATTERNS: tuple[str, ...] = (
    r"(?:19|20)\d\d",
    r"^(?:to|want|need|wanna)-",
    r"(?:^|-)(?:my|i|own|owned|have|mine)(?:-|$)",
    r"read(?:ing)?-(?:in|20|19)|(?:^|-)read-\d|^read$|-reads?$|^reads?-",
    r"fav",
    r"(?:^|-)(?:kindle|e-?books?|audio-?books?|audio|audible|nook|pdf|epub|ibooks?)(?:-|$)",
    r"(?:^|-)(?:library|shelf|shelves|bookshelf|home|wishlist|wish-list)(?:-|$)",
    r"\d-stars?|^stars?$",
    r"shelfari|goodreads|calibre|default|book-?club|dnf|not-finish|unfinish",
)

# Нормализация синонимичных тегов.
TAG_SYNONYMS: dict[str, str] = {
    "science-fiction": "sci-fi",
    "scifi": "sci-fi",
    "ya": "young-adult",
    "ya-fiction": "young-adult",
    "young-adult-fiction": "young-adult",
    "nonfiction": "non-fiction",
    "classic": "classics",
    "classic-literature": "classics",
    "mysteries": "mystery",
    "thrillers": "thriller",
    "novel": "novels",
    "humour": "humor",
    "childrens": "children",
    "children-s": "children",
    "kids": "children",
    "childrens-books": "children",
    "graphic-novel": "graphic-novels",
    "memoirs": "memoir",
    "dystopia": "dystopian",
    "vampire": "vampires",
    "children-s-books": "children",
    "biographies": "biography",
}
