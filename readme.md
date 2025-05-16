# Financial Transactions Anomaly Detection Pipeline

## Код для предобработки, генерации признаков, обучения автоэнкодера для детекции аномалий, кластеризации гео-данных и визуализации результатов на карте.

````markdown
## 📂 Структура проекта

```text
├── config/
│   └── pipeline_config.json    # Конфиг с гиперпараметрами и путями
├── data/
│   ├── raw/                    # Исходные CSV (например, transactions.csv)
│   └── processed/              # Результаты: full_output.csv, anomalies.csv, map.html, модель
├── scripts/
│   └── run_pipeline.py         # Точка входа для запуска пайплайна
├── src/
│   ├── data_loader.py          # Загрузка и базовая предобработка
│   ├── features.py             # Генерация rolling- и geo-признаков
│   ├── clustering.py           # HDBSCAN-кластеризация
│   ├── model.py                # Определение, обучение и сохранение автоэнкодера
│   └── utils.py                # Вспомогательные функции (логирование, сохранение)
├── tests/
│   └── ...                     # Unit-тесты для модулей
├── requirements.txt            # Список зависимостей
└── README.md                   # Этот файл
````

---

## ⚙️ Пререквизиты

* Python 3.8+
* Git
* Рекомендуется создать и активировать виртуальное окружение:

  ```bash
  python -m venv .venv
  source .venv/bin/activate    # Linux / macOS
  .venv\Scripts\activate       # Windows PowerShell
  ```

---

## 📥 Установка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/K0RTEK/financial_transactions.git
   cd financial_transactions
   ```
2. Установите зависимости и пакет в режиме разработки:

   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```
3. (Опционально) Сгенерируйте актуальный `requirements.txt`:

   ```bash
   pip freeze > requirements.txt
   ```

---

## 🔧 Конфигурация

В файле `config/pipeline_config.json` укажите:

* `time_col`: название колонки с меткой времени (например, `"transactiontime"`).
* `group_cols`: список ключей группировки (например, `["carduid"]`).
* `rolling_windows`: массив окон в минутах (например, `[5,10,20,30]`).
* `rolling_stats`: список статистик (обычно `["count"]`).
* `lat_col`, `lon_col`: названия колонок широты/долготы.
* `feature_cols`: финальный список признаков (см. шаблон).
* Параметры автоэнкодера в разделе `autoencoder`.
* Пороги и гиперпараметры HDBSCAN.

Пример минимального конфига см. в `config/pipeline_config.json`.

---

## 🚀 Запуск пайплайна

```bash
python scripts/run_pipeline.py \
  --config config/pipeline_config.json \
  --input data/raw/transactions.csv \
  --output data/processed/full_output.csv \
  --map data/processed/map.html \
  --model data/processed/model
```

* `--config`  — путь к JSON-конфигу.
* `--input`   — CSV с сырыми данными.
* `--output`  — куда сохранить весь результат (full\_output.csv).
* `--map`     — путь для HTML-карты (map.html).
* `--model`   — папка/файл для сохранения обученной модели.

После выполнения будет:

* `data/processed/full_output.csv` — полный фрейм с фичами и метками.
* `data/processed/anomalies.csv` — только записи, признанные аномалиями.
* `data/processed/map.html` — интерактивная карта с кластерами.
* `data/processed/model/` — файлы автоэнкодера.

---

## ✅ Тестирование

Запустите все unit-тесты через `pytest`:

```bash
pytest --maxfail=1 --disable-warnings -q
```

---

## 📚 Примеры

1. Построение фичей за разные окна:

   ```bash
   python scripts/run_pipeline.py \
     --config config/pipeline_config.json \
     --input data/raw/transactions.csv \
     --output data/processed/full_output.csv
   ```
2. Только сохранить модель:

   ```bash
   python scripts/run_pipeline.py \
     --config config/pipeline_config.json \
     --input data/raw/transactions.csv \
     --model data/processed/model
   ```