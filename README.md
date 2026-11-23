# sourcematch

## Документация

- [Бизнес-требования](docs/business-requirements.md)
- [Приёмка и оценка качества](docs/acceptance-and-quality.md)
- [Риски и рабочие допущения](docs/risks-and-assumptions.md)
- [Техническая архитектура и ADR](docs/technical-architecture.md)

## Запуск

Понадобятся Rust, файловые выгрузки и доступные локальные API моделей. Профиль поиска задаёт способ подключения. Программа проверяет API перед прогоном. Серверы и активные модели управляются отдельно. Команда ниже воспроизводит проверенный результат с эмбеддингами и BM25.

```sh
cargo run --release -- run --no-rerank \
  --listings 'input/listings (1).csv' \
  --offers 'input/offers (1).csv' \
  --catalog 'input/ru_catalog (1).csv' \
  --template 'input/submission_template (1).csv' \
  --out submission.csv
```

На новых данных подставьте пути к CSV. Без шаблона результат охватит все карточки. Каталог закупок опционален. Правила рынка переопределяются через `--rules`, подключение моделей — через `--search-profile`. Локальный кеш, журнал SQLite и манифесты прогонов сохраняются в `runs/`.

Для переименованных колонок передайте `--input-profile` с TOML-таблицами `listings`, `offers`, `catalog` или `template`. Ключ — имя колонки выгрузки, значение — имя колонки из ТЗ. Например, `[listings]` и `market_id = "listing_id"`.

`prepare` проверяет подключение к обеим моделям. Для полного режима используйте профиль с доступным API реранжирования и вызовите `run` без `--no-rerank`. Манифест показывает, какой режим использован. Для `evaluate` нужны прогнозы и отдельные эталонные строки. Прогон по ним можно получить, указав `--template 'input/labeled (1).csv'` и отдельный выходной файл.

```sh
cargo test
cargo run --release -- simulate --seed 42 --out-dir runs/synthetic
cargo run --release -- evaluate --predictions runs/labeled.csv \
  --labels 'input/labeled (1).csv'
```

- [Отчёт о реализации и проверках](docs/report.md)
- [Первые 90 дней](docs/first-90-days.md)
- [Контекст использования ИИ](context.md)
