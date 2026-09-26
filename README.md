# Прототип книжного рекомендательного сервиса (goodbooks-10k)

## Задача

Построение и сравнение рекомендательных моделей для сервиса подбора книг на датасете goodbooks-10k:

1. Разведочный анализ данных (EDA): распределение оценок, активность пользователей, популярность книг и long tail, теги, проблемы данных (sparsity, popularity bias, cold start).
2. Базовые модели: неперсонализированная Popularity (Top-N по среднему рейтингу с порогом числа оценок) и Content-Based (TF-IDF по названию и тегам, cosine similarity).
3. Item-based Collaborative Filtering на разреженной матрице «пользователь × книга».
4. Matrix Factorization: SVD, оценка ошибки предсказания RMSE.
5. Сравнение top-N моделей по Precision@K, Recall@K, nDCG@K на отложенной выборке.
6. Гибридная стратегия для cold start, выводы и пути улучшения.

## Данные

Согласно инструкции на странице датасета на Kaggle (https://www.kaggle.com/datasets/zygmunt/goodbooks-10k), используется актуальная версия датасета из репозитория автора на GitHub: https://github.com/zygmuntz/goodbooks-10k.

Причина: версия на Kaggle устарела. Она содержит дубликаты оценок и около 980 тыс. оценок, отобранных примерно по 100 на книгу, что искажает распределение популярности книг и активности пользователей. Актуальная версия содержит все оценки без дубликатов, отсортированные по времени; порядок строк позволяет построить временное разбиение на обучающую и тестовую выборки. Статистики со страницы Kaggle относятся к устаревшей версии и в работе не используются.

Лицензия датасета: CC BY-SA 4.0.

| Файл | Строк | Содержимое |
|---|---:|---|
| `ratings.csv` | 5 976 479 | Оценки `user_id, book_id, rating` (1–5), строки упорядочены по времени |
| `books.csv` | 10 000 | Метаданные книг: `book_id`, `goodreads_book_id`, авторы, `original_title`, `title`, год, агрегаты рейтинга |
| `tags.csv` | 34 252 | Справочник тегов `tag_id, tag_name` |
| `book_tags.csv` | 999 912 | Теги книг `goodreads_book_id, tag_id, count` |

Идентификаторы непрерывны: книги 1–10 000, пользователи 1–53 424. Теги связываются с книгами через `books.goodreads_book_id`. Результаты проверки целостности: дубликатов пар `(user_id, book_id)` и пропусков в оценках нет; все ссылки между таблицами корректны; в `book_tags.csv` 8 повторяющихся пар `(goodreads_book_id, tag_id)` и 6 записей с неположительным `count`; у 585 книг отсутствует `original_title`.

## Структура репозитория

```
Rec_Systems_HW1/
├── README.md
├── requirements.txt          # закреплённые версии зависимостей
├── pyproject.toml            # пакет recsys, настройки ruff и pytest
├── scripts/
│   └── download_data.py      # загрузка CSV с GitHub в data/raw/
├── src/recsys/
│   ├── config.py             # пути, RANDOM_STATE, параметры протокола оценки
│   └── data.py               # загрузка и валидация данных
├── reports/figures/          # графики для README
└── data/raw/                 # исходные CSV (не входят в репозиторий)
```

## Воспроизведение

Требования: Python 3.14 (Windows, x86-64); проверено на Python 3.14.3.

```bash
git clone https://github.com/a-kapset/Rec_Systems_HW1.git
cd Rec_Systems_HW1
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Получение данных (файлы сохраняются в `data/raw/`):

- автоматически: `python scripts/download_data.py` — загрузка `ratings.csv`, `books.csv`, `tags.csv`, `book_tags.csv` из репозитория https://github.com/zygmuntz/goodbooks-10k с проверкой числа строк; уже загруженные файлы пропускаются;
- вручную: скачать те же четыре файла из репозитория https://github.com/zygmuntz/goodbooks-10k и поместить в каталог `data/raw/`.

Проверка загрузки и целостности данных:

```bash
python -c "from recsys import data; print(data.validate(data.load_all()))"
```
