# 🤖 AI Analytics Agent

Аналитический продукт на базе **GigaChat** с агентным подходом: ИИ самостоятельно пишет и выполняет Python-код для анализа данных через безопасный интерпретатор.

---

## Возможности

| Функция | Описание |
|---------|----------|
| **Интеграция с GigaChat** | Прямая работа с API через SDK, без ручного веб-интерфейса |
| **Агентный цикл** | Function calling → генерация кода → выполнение → интерпретация |
| **Code Interpreter** | Безопасное выполнение pandas/plotly-кода в изолированном процессе |
| **Защита от injection** | Валидация кода, blacklist опасных операций, санитайзинг ввода |
| **Streamlit UI** | Темная тема, загрузка CSV/Excel, история диалогов, превью данных |
| **Docker-поддержка** | Готовые Dockerfile и docker-compose.yml для деплоя |
| **Кэширование** | Сохранение истории диалогов с автоматической очисткой |

---

## Требования

| Компонент | Версия | Примечание |
|-----------|--------|------------|
| Python | 3.9–3.11 | Рекомендуется 3.10 |
| pip | ≥21.0 | Для установки зависимостей |
| ОЗУ | ≥4 ГБ | Для работы с датасетами до 100 МБ |
| Интернет | ✅ | Для доступа к GigaChat API |

### Зависимости

Все зависимости указаны в `requirements.txt`:

```txt
streamlit>=1.32.0
gigachat>=0.1.15
pandas>=2.0.0
plotly>=5.18.0
python-dotenv>=1.0.0
pydantic>=2.0.0
aiohttp>=3.9.0
bleach>=6.1.0
tabulate>=0.9.0
openpyxl>=3.1.0
```

---

## Установка

### Локально

```bash
# 1. Клонирование репозитория
git clone <your-repo-url>
cd llm_analytics_product

# 2. Создание виртуального окружения
python -m venv venv

# 3. Активация
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Установка зависимостей
pip install --upgrade pip
pip install -r requirements.txt
```

### Через Docker

```bash
# 1. Сборка и запуск
docker-compose up --build

# 2. Приложение доступно на http://localhost:8501
```

---

## Настройка

### 1. Получение API-ключа GigaChat

1. Перейдите на [developers.sber.ru](https://developers.sber.ru/portal)
2. Создайте проект → подключите **GigaChat**
3. Во вкладке "Учётные данные" создайте ключ типа `GIGACHAT_API_CORP`
4. Скопируйте ключ (начинается с `eyJ...`)

> Ключ показывается только один раз — сохраните его надёжно!

### 2. Установка сертификата (Windows)

GigaChat использует российский сертификат НУЦ Минцифры:

```bash
# Скачайте сертификат
curl -O https://storage.googleapis.com/gigachat-certificates/rootCA.pem

# Установите через двойной клик:
# 1. "Установить сертификат" → "Локальный компьютер"
# 2. "Поместить все сертификаты в: Доверенные корневые центры"
# 3. Завершите мастер
```

### 3. Настройка `.env`

```bash
cp .env.example .env
```

Заполните файл:

```ini
# === GigaChat API ===
LLM_PROVIDER_LABEL=GigaChat
LLM_API_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
LLM_CHAT_COMPLETIONS_PATH=/chat/completions
LLM_MODEL=GigaChat-Pro
LLM_API_KEY=your_api_key...

# === Параметры ===
LLM_TIMEOUT_SECONDS=90
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=1200
LLM_MAX_CONTEXT_ROWS=30
LLM_MAX_CHARTS=3
LLM_AGENT_MAX_STEPS=5
LLM_CODE_TIMEOUT_SECONDS=12

# === Безопасность ===
ALLOWED_PANDAS_METHODS=describe,head,tail,info,corr,groupby,mean,sum,count,value_counts,plot
FORBIDDEN_KEYWORDS=os.system,subprocess,eval,exec,__import__,open(,rm ,del ,shutil
```
---

## Запуск

```bash
# Активируйте окружение
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Запустите Streamlit
streamlit run app.py --server.port 8501 --server.address localhost
```

Откройте в браузере: **`http://localhost:8501`**

---

## Использование

### 1. Загрузка данных
- Перетащите CSV или Excel-файл в зону загрузки
- Поддерживаются файлы до 200 МБ
- Автоматическое определение кодировки и форматов

### 2. Примеры запросов

```
"Рассчитай средние продажи по регионам"
"Найди топ-5 продуктов по выручке"
"Есть ли аномалии в данных?"
"Сравни средние значения по категориям"
```

### 3. Результат

Агент возвращает:
- Текстовый ответ с выводами на русском
- Сгенерированный Python-код (можно просмотреть)
- Таблицы с результатами вычислений
- Информацию об ошибках (если возникли)

### 4. История диалогов
- Последние 5 запросов отображаются внизу
- Кнопка "🗑️ Очистить всё" в сайдбаре сбрасывает сессию

---

## Структура проекта

```
llm_analytics_product/
├── app.py                 # Streamlit UI: чат, загрузка, визуализация
├── llm_client.py          # Клиент GigaChat + агентный цикл с function calling
├── llm_stream.py          # SSE streaming ответов (опционально)
├── analytics_core.py      # Валидация, защита от injection, спецификации графиков
├── code_interpreter.py    # Безопасный Python-интерпретатор с изоляцией
├── chat_cache.py          # Кэширование истории диалогов
├── requirements.txt       # Зависимости Python
├── Dockerfile            # Образ для контейнеризации
├── docker-compose.yml    # Оркестрация сервисов
├── .env.example          # Шаблон конфигурации
└── README.md             
```

---

## Безопасность

### Защита от prompt-injection

```python
# analytics_core.py
class SecurityGuard:
    FORBIDDEN_PATTERNS = [
        r'os\.(system|popen|exec)',
        r'subprocess\.(call|run|Popen)',
        r'__import__\s*\(',
        r'eval\s*\(',
        r'exec\s*\(',
        # ... и другие опасные паттерны
    ]
```

### Изоляция выполнения кода

- Код выполняется в **отдельном процессе** с таймаутом
- Запрещены импорты системных модулей (`os`, `sys`, `subprocess`)
- Разрешены только безопасные методы pandas (настраивается в `.env`)
- Все данные передаются через Base64-кодирование

---

## 🐛 Устранение неполадок

| Ошибка | Решение |
|--------|---------|
| `SSL: CERTIFICATE_VERIFY_FAILED` | Установите rootCA.pem (см. [Настройка](#-настройка)) |
| `401 Unauthorized` | Проверьте `LLM_API_KEY` в `.env` |
| `Timeout: код выполнялся дольше 12с` | Увеличьте `LLM_CODE_TIMEOUT_SECONDS` в `.env` |

### Режим отладки

Включите вывод очищенного кода в консоли:

```python
# code_interpreter.py, метод execute_python
print("=" * 60)
print("ОЧИЩЕННЫЙ КОД:")
print(cleaned_code)
print("=" * 60)
```

---

## Лицензия
Проект распространяется под лицензией **MIT**. См. файл `LICENSE` для деталей.

---

## Контакты

По вопросам и предложениям: [danamottueva@gmail.com]
Моттуева Уруйдана 
Б9123-01.03.02ии
