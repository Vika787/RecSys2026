# ДЗ1

Смелова Виктория Евгеньевна

**Датасет:** Amazon Reviews 2018 — Musical Instruments (5-core).

**Источник:** https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/

**Файлы:**

- https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFilesSmall/Musical_Instruments_5.json.gz
- https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/meta_Musical_Instruments.json.gz

Взаимодействие — отзыв пользователя о товаре с числовой оценкой и датой. Признаки товаров включают название, бренд и категории. Релевантным для оценки Top-10 считается товар с рейтингом не ниже 4.

Исходный набор содержит 231 392 отзывов, 27 530 пользователей и 10 620 товаров. После очистки, ограничения каталога и формирования warm-start выборок используется 126 057 взаимодействий: train — 115 300, validation — 8 046, test — 2 711.

### Воспроизведение эксперимента

Для запуска работы необходимо:

1. Установить Python 3.10 или новее.
2. Установить зависимости из `requirements.txt` командой `pip install -r requirements.txt`.
3. Открыть `experiments.ipynb` из директории `hw1`.

Данные Amazon Musical Instruments автоматически загружаются из открытого источника при первом запуске и сохраняются в локальную директорию `data/`. Повторная загрузка не требуется, если файлы уже существуют.

Все эксперименты используют фиксированное случайное зерно `SEED = 42`. Гиперпараметры выбираются только на validation, после чего модели переобучаются на train + validation и оцениваются на test.
