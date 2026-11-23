# sourcematch

## Документация

- [Бизнес-требования](docs/business-requirements.md)
- [Приёмка и оценка качества](docs/acceptance-and-quality.md)
- [Риски и рабочие допущения](docs/risks-and-assumptions.md)
- [Техническая архитектура и ADR](docs/technical-architecture.md)
- [Настройка скоринга и дообучение](docs/scoring-calibration.md)

## Запуск

Понадобятся Rust, файловые выгрузки и доступные локальные API моделей. Профиль поиска задаёт способ подключения. Программа проверяет API перед прогоном. Серверы и активные модели управляются отдельно. Команда ниже воспроизводит проверенный результат с эмбеддингами и BM25.

```sh
cargo run --release -- run --no-rerank \
  --listings 'input/listings (1).csv' \
  --offers 'input/offers (1).csv' \
  --catalog 'input/ru_catalog (1).csv' \
  --template 'input/submission_template (1).csv' \
  --scoring-profile profiles/scoring-strict.toml \
  --out submission.csv
```

На новых данных подставьте пути к CSV. Без шаблона результат охватит все карточки. Каталог закупок опционален. Правила рынка переопределяются через `--rules`, подключение моделей — через `--search-profile`. Локальный кеш, журнал SQLite и манифесты прогонов сохраняются в `runs/`.

Для переименованных колонок передайте `--input-profile` с TOML-таблицами `listings`, `offers`, `catalog` или `template`. Ключ — имя колонки выгрузки, значение — имя колонки из ТЗ. Например, `[listings]` и `market_id = "listing_id"`.

`prepare` проверяет подключение к обеим моделям и порядок двух контрольных предложений. Локальный профиль получает оценки реранжера через `/v1/chat/completions` в LM Studio. Для полного режима вызовите `run` без `--no-rerank`. Манифест показывает модель и протокол реранжирования. Для `evaluate` нужны прогнозы и отдельные эталонные строки. Прогон по ним можно получить, указав `--template 'input/labeled (1).csv'` и отдельный выходной файл.

Профиль `profiles/scoring-strict.toml` создаёт отдельную очередь запусков и ручных проверок по финансовому приоритету. Путь к очереди указан в поле `portfolio` манифеста. Профиль `profiles/scoring-logistics.toml` включает восстановление логистики по сопоставимым поставщикам. `profiles/scoring-pilot.toml` также включает волну по спросу. Оба кандидата запускайте с отдельным `--out` для сравнения. [ADR волн скоринга](docs/adr/0016-portfolio-scoring.md)

```sh
cargo test
cargo run --release -- evaluate --predictions runs/labeled.csv \
  --labels 'input/labeled (1).csv'
```

Каждый синтетический сценарий содержит входные файлы, свой профиль рынка и `gold.csv`. Для полной сверки всех сценариев используйте обычный конвейер и строгий режим эвала.

```sh
set -e
cargo build --release
target/release/sourcematch simulate --seed 42 --out-dir runs/synthetic
for scenario in runs/synthetic/*/; do
  target/release/sourcematch run --no-rerank \
    --listings "${scenario}listings.csv" --offers "${scenario}offers.csv" \
    --rules "${scenario}rules.toml" --out "${scenario}result.csv"
  target/release/sourcematch evaluate --verify-synthetic \
    --predictions "${scenario}result.csv" --labels "${scenario}gold.csv"
done
```

Для финансового сравнения включите один из экспериментальных профилей на тех же сценариях и запустите обычный `evaluate` без `--verify-synthetic`. В блоке `synthetic_finance` показаны вклад продаж, закупленный капитал, остаток и число неподходящих запусков по скрытым фактам. Базовый строгий эвал сверяет ответы по доступным конвейеру входам.

- [Отчёт о реализации и проверках](docs/report.md)
- [Первые 90 дней](docs/first-90-days.md)
- [Контекст использования ИИ](context.md)
