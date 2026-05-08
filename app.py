# app.py - AI Analytics Agent с реальным выполнением кода (Code Interpreter)
import ssl
import os
import re
import json
import time
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import traceback
import base64
import signal
from io import StringIO, BytesIO
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

# Отключаем SSL предупреждения
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ["CURL_CA_BUNDLE"] = ""

px.defaults.template = "plotly_white"

# ============================================================================
# БЕЗОПАСНЫЙ CODE INTERPRETER (Вариант A: локальное выполнение с ограничениями)
# ============================================================================

class SafeCodeInterpreter:
    """
    Безопасный интерпретатор кода для выполнения аналитики на данных.
    - Изолированная среда выполнения
    - Ограниченный набор доступных модулей
    - Защита от опасных операций
    - Таймаут выполнения
    """
    
    ALLOWED_MODULES = {'pd': pd, 'pandas': pd, 'np': np, 'numpy': np, 'math': __import__('math'), 'statistics': __import__('statistics')}
    ALLOWED_BUILTINS = {'print': print, 'len': len, 'str': str, 'int': int, 'float': float, 'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'sum': sum, 'min': min, 'max': max, 'abs': abs, 'round': round}
    
    FORBIDDEN_PATTERNS = [
        r'__import__', r'import\s+os', r'import\s+sys', r'import\s+subprocess',
        r'import\s+shutil', r'import\s+socket', r'import\s+urllib', r'import\s+requests',
        r'os\.', r'sys\.', r'subprocess\.', r'shutil\.', r'socket\.', r'urllib\.',
        r'eval\s*\(', r'exec\s*\(', r'compile\s*\(', r'open\s*\(', r'input\s*\(',
        r'breakpoint\s*\(', r'pickle', r'marshal', r'shelve', r'__builtins__',
        r'__class__', r'__globals__', r'__closure__', r'__code__', r'__func__',
        r'__self__', r'__mro__', r'__subclasses__', r'__import__', r'__loader__'
    ]
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout = timeout_seconds
        
    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Проверка кода на наличие опасных паттернов"""
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"⚠️ Запрещённая операция: {pattern}"
        return True, ""
    
    @contextmanager
    def _timeout_handler(self):
        """Контекстный менеджер для таймаута выполнения (только Unix)"""
        if os.name != 'nt':  # Не работает на Windows
            def handler(signum, frame):
                raise TimeoutError("⏱️ Превышено время выполнения кода")
            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.timeout)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # На Windows просто выполняем без таймаута (альтернатива: использовать multiprocessing)
            yield
    
    def execute(self, code: str, df: pd.DataFrame) -> dict:
        """
        Выполняет код аналитики в безопасной среде.
        
        Returns:
            dict с результатами: {'success': bool, 'result': Any, 'error': str, 'output': str}
        """
        # Валидация кода
        is_safe, error_msg = self._validate_code(code)
        if not is_safe:
            return {'success': False, 'error': error_msg, 'result': None, 'output': ''}
        
        # Подготовка изолированной среды
        safe_globals = {
            '__builtins__': self.ALLOWED_BUILTINS,
            'df': df.copy(),  # Копия для защиты оригинала
            **self.ALLOWED_MODULES
        }
        safe_locals = {}
        
        # Перехват вывода print()
        output_buffer = StringIO()
        
        try:
            with self._timeout_handler():
                # Выполнение кода
                result = None
                # Разделяем код на строки для обработки последней как выражения
                lines = [line.strip() for line in code.strip().split('\n') if line.strip() and not line.strip().startswith('#')]
                
                if not lines:
                    return {'success': False, 'error': '❌ Пустой код', 'result': None, 'output': ''}
                
                # Выполняем все строки кроме последней как утверждения
                for line in lines[:-1]:
                    exec(line, safe_globals, safe_locals)
                
                # Последнюю строку пытаемся выполнить как выражение для получения результата
                last_line = lines[-1]
                try:
                    # Пробуем как выражение
                    result = eval(last_line, safe_globals, safe_locals)
                except:
                    # Если не вышло - выполняем как утверждение
                    exec(last_line, safe_globals, safe_locals)
                    result = safe_locals.get('result', safe_locals.get('output', None))
                
                return {
                    'success': True, 
                    'result': result, 
                    'error': None, 
                    'output': output_buffer.getvalue()
                }
                
        except TimeoutError as e:
            return {'success': False, 'error': str(e), 'result': None, 'output': ''}
        except SyntaxError as e:
            return {'success': False, 'error': f"❌ Синтаксическая ошибка: {e}", 'result': None, 'output': ''}
        except NameError as e:
            return {'success': False, 'error': f"❌ Неизвестная переменная: {e}", 'result': None, 'output': ''}
        except AttributeError as e:
            return {'success': False, 'error': f"❌ Ошибка атрибута: {e}", 'result': None, 'output': ''}
        except KeyError as e:
            return {'success': False, 'error': f"❌ Ключ не найден: {e}", 'result': None, 'output': ''}
        except Exception as e:
            return {'success': False, 'error': f"❌ {type(e).__name__}: {e}", 'result': None, 'output': ''}


# ============================================================================
# КЛАСС АГЕНТА: LLM генерирует код -> Code Interpreter выполняет -> Возврат реальных результатов
# ============================================================================

class QwenAnalyticsAgent:
    """Агент: LLM генерирует Python-код, который выполняется локально на реальных данных"""
    
    def __init__(self, api_key: str, model: str = "qwen-3-6-plus"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.gen-api.ru/api/v1/networks/qwen-3-6-plus"
        self.interpreter = SafeCodeInterpreter(timeout_seconds=30)
        
    def test_connection(self) -> dict:
        """Тест подключения к API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        input_data = {"messages": [{"role": "user", "content": "Hi"}], "is_sync": True, "max_tokens": 10}
        
        try:
            response = requests.post(self.base_url, headers=headers, json=input_data, timeout=30)
            if response.status_code == 200:
                return {'success': True, 'error': None, 'details': {'status': 'OK'}}
            elif response.status_code == 401:
                return {'success': False, 'error': "❌ Неверный токен", 'details': {'status_code': 401}}
            else:
                return {'success': False, 'error': f"❌ Ошибка {response.status_code}", 'details': response.text}
        except Exception as e:
            return {'success': False, 'error': f"❌ {type(e).__name__}", 'details': str(e)}

    def _get_dataset_summary(self, df: pd.DataFrame) -> str:
        """
        Генерирует компактное, но репрезентативное описание датасета для LLM.
        Вместо первых 100 строк — статистические сводки + примеры значений.
        """
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        summary = []
        summary.append(f"📊 ДАТАСЕТ: {df.shape[0]} строк × {df.shape[1]} столбцов")
        summary.append(f"\n🔢 Числовые столбцы ({len(numeric_cols)}): {numeric_cols[:10]}{'...' if len(numeric_cols) > 10 else ''}")
        summary.append(f"🏷️ Категориальные столбцы ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols) > 10 else ''}")
        
        if numeric_cols:
            summary.append("\n📈 Описательная статистика (числовые):")
            stats = df[numeric_cols].describe(include='all', datetime_is_numeric=True).T
            for col in numeric_cols[:5]:  # Первые 5 числовых
                if col in stats.index:
                    row = stats.loc[col]
                    summary.append(f"  • {col}: mean={row.get('mean', 'N/A'):.2f}, std={row.get('std', 'N/A'):.2f}, min={row.get('min', 'N/A')}, max={row.get('max', 'N/A')}")
        
        if cat_cols:
            summary.append("\n📦 Примеры категориальных значений:")
            for col in cat_cols[:3]:  # Первые 3 категориальных
                unique_vals = df[col].dropna().unique()[:5]
                summary.append(f"  • {col}: {list(unique_vals)} (уникальных: {df[col].nunique()})")
        
        # Добавляем 3-5 реальных строк как пример (не 100!)
        sample = df.head(3).to_dict('records')
        summary.append(f"\n📋 Примеры строк (3 из {len(df)}):")
        for i, row in enumerate(sample, 1):
            summary.append(f"  {i}. {dict(list(row.items())[:5])}")  # Первые 5 полей
        
        return "\n".join(summary)

    def _generate_code_prompt(self, user_query: str, dataset_summary: str, auto_mode: bool) -> str:
        """Формирует промпт для генерации кода, а не выполнения"""
        
        system_instruction = """Ты эксперт по анализу данных на Python (pandas, numpy).
Твоя задача — ГЕНЕРИРОВАТЬ рабочий Python-код для анализа данных.

ПРАВИЛА:
1. Верни ТОЛЬКО валидный JSON в формате:
{
  "thought": "краткое объяснение подхода",
  "code": "рабочий Python-код (одна строка или несколько, без ```)",
  "expected_output_type": "число|строка|DataFrame|Series|dict|list|plot_config"
}

2. Код должен:
   - Использовать переменную df (уже загруженный pandas DataFrame)
   - Возвращать результат через последнюю строку (как выражение) ИЛИ присваивать в переменную result/output
   - Быть безопасным: без import os, exec, eval, open, и т.д.
   - Обрабатывать возможные ошибки (используй try-except если нужно)

3. Примеры кода:
   - Для среднего: df['salary'].mean()
   - Для корреляции: df['col1'].corr(df['col2'])
   - Для группировки: df.groupby('dept')['salary'].mean().to_dict()
   - Для описательной статистики: df.describe().to_dict()
   - Для фильтрации: df[df['age'] > 30]['salary'].mean()

4. НЕ выполняй код сам — только генерируй его!"""

        if auto_mode:
            user_content = f"""{dataset_summary}

ЗАДАЧА: Автоматический анализ датасета.
Сгенерируй код, который вычислит:
1. Описательную статистику числовых переменных
2. Распределения категориальных переменных (топ-5 значений)
3. Корреляционную матрицу для числовых столбцов
4. Процент пропущенных значений по столбцам
5. Выбросы через IQR для ключевых числовых переменных

Верни код, который возвращает dict с этими метриками."""
        else:
            user_content = f"""{dataset_summary}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ: "{user_query}"

Сгенерируй рабочий Python-код для ответа на этот запрос.
Код должен работать с переменной df и возвращать конкретный результат."""

        return system_instruction, user_content

    def _parse_llm_response(self, raw_response: str) -> dict:
        """Парсит ответ LLM и извлекает JSON с кодом"""
        try:
            # Пробуем найти JSON в ответе
            json_match = re.search(r'\{[\s\S]*\}', raw_response)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                if 'code' in parsed:
                    return {'success': True, 'data': parsed}
            # Если не получилось — пробуем распарсить весь ответ
            parsed = json.loads(raw_response)
            if 'code' in parsed:
                return {'success': True, 'data': parsed}
            return {'success': False, 'error': '❌ В ответе нет поля "code"'}
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f'❌ Ошибка парсинга JSON: {e}'}
        except Exception as e:
            return {'success': False, 'error': f'❌ {type(e).__name__}: {e}'}

    def generate_and_execute(self, user_query: str, df: pd.DataFrame, auto_mode: bool) -> dict:
        """
        Основной цикл: 
        1. LLM генерирует код на основе описания данных
        2. Локальный SafeCodeInterpreter выполняет код на РЕАЛЬНЫХ данных
        3. Возвращаются фактические результаты
        """
        
        # Шаг 1: Подготовка контекста для LLM
        dataset_summary = self._get_dataset_summary(df)
        system_prompt, user_prompt = self._generate_code_prompt(user_query, dataset_summary, auto_mode)
        
        # Шаг 2: Запрос к LLM для генерации кода
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        input_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "is_sync": True,
            "temperature": 0.1,  # Низкая температура для стабильного кода
            "top_p": 0.95,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=input_data,
                timeout=60
            )
            
            if response.status_code != 200:
                return {'success': False, 'error': f'❌ Ошибка API: {response.status_code}', 'details': response.text[:300]}
            
            result = response.json()
            
            # Надёжный парсинг ответа gen-api.ru
            raw_text = None
            if "response" in result and isinstance(result["response"], list) and result["response"]:
                raw_text = result["response"][0]
            elif "output" in result:
                raw_text = result["output"] if isinstance(result["output"], str) else json.dumps(result["output"])
            elif isinstance(result, str):
                raw_text = result
            else:
                raw_text = json.dumps(result)
            
            # Парсим код из ответа LLM
            code_result = self._parse_llm_response(raw_text)
            if not code_result['success']:
                return {'success': False, 'error': code_result['error'], 'llm_raw': raw_text[:500]}
            
            generated_code = code_result['data']['code']
            thought = code_result['data'].get('thought', '')
            
            # Шаг 3: ВЫПОЛНЕНИЕ КОДА НА РЕАЛЬНЫХ ДАННЫХ (ключевое исправление!)
            execution = self.interpreter.execute(generated_code, df)
            
            if not execution['success']:
                return {
                    'success': False, 
                    'error': f"🔧 Код сгенерирован, но ошибка выполнения: {execution['error']}",
                    'generated_code': generated_code,
                    'thought': thought
                }
            
            # Шаг 4: Форматирование результата
            result_value = execution['result']
            
            # Конвертируем сложные типы в сериализуемые
            if isinstance(result_value, (pd.DataFrame, pd.Series)):
                metrics_text = result_value.to_string() if len(result_value) <= 20 else result_value.head(10).to_string() + "\n..."
                result_serializable = result_value.to_dict() if isinstance(result_value, pd.Series) else result_value.head(10).to_dict('records')
            elif isinstance(result_value, (np.ndarray, np.generic)):
                metrics_text = str(result_value)
                result_serializable = result_value.tolist() if hasattr(result_value, 'tolist') else float(result_value)
            elif isinstance(result_value, dict):
                metrics_text = json.dumps(result_value, indent=2, default=str)[:1000]
                result_serializable = result_value
            else:
                metrics_text = str(result_value)
                result_serializable = result_value
            
            return {
                'success': True,
                'thought': thought,
                'generated_code': generated_code,
                'metrics_text': metrics_text,
                'result': result_serializable,
                'output': execution['output']
            }
            
        except requests.exceptions.Timeout:
            return {'success': False, 'error': '⏱️ Таймаут запроса к LLM'}
        except Exception as e:
            return {'success': False, 'error': f'❌ {type(e).__name__}: {str(e)}', 'details': traceback.format_exc()[:500]}


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def check_safety(query: str) -> tuple:
    """Проверка запроса на безопасность"""
    FORBIDDEN_PATTERNS = [
        r'eval\s*\(', r'exec\s*\(', r'compile\s*\(',
        r'import\s+os', r'import\s+sys', r'import\s+subprocess',
        r'os\.system', r'subprocess\.', r'shutil\.',
        r'__import__', r'__builtins__', r'__class__',
        r'open\s*\(', r'read\s*\(', r'write\s*\(',
        r'input\s*\(', r'breakpoint\s*\(',
        r'pickle', r'marshal', r'shelve'
    ]
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "⚠️ Обнаружен подозрительный паттерн"
    if '<script' in query.lower() or 'javascript:' in query.lower():
        return False, "⚠️ XSS попытка"
    return True, "✅ Безопасно"


def format_result_for_display(result: Any) -> str:
    """Форматирует результат для отображения в UI"""
    if result is None:
        return "Нет результата"
    if isinstance(result, (dict, list)):
        return json.dumps(result, indent=2, default=str, ensure_ascii=False)[:2000]
    return str(result)


# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(page_title="AI Analytics Agent", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stPlotlyChart {border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .insight-tag {display: inline-block; background: #e3f2fd; padding: 5px 12px; border-radius: 20px; margin: 4px 4px 4px 0; font-size: 0.9em;}
    .code-block {background: #1e1e1e; color: #d4d4d4; padding: 12px; border-radius: 6px; font-family: 'Fira Code', monospace; font-size: 0.85em; overflow-x: auto;}
    .metrics-block {background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; font-family: monospace; white-space: pre-wrap;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Analytics Agent")
st.markdown("*LLM генерирует код → Локальный интерпретатор выполняет → Реальные результаты*")

if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input("🔑 API Token", type="password", value=st.session_state.api_key)
    st.session_state.api_key = api_key
    
    if api_key and st.button("🔌 Проверить подключение"):
        with st.spinner("Тест..."):
            test = QwenAnalyticsAgent(api_key=api_key).test_connection()
            if test['success']:
                st.success("✅ Подключение успешно!")
            else:
                st.error(test['error'])
    
    model = st.selectbox("🧠 Модель", ["qwen-3-6-plus", "qwen-3-5-plus", "qwen-max"], index=0)
    
    st.divider()
    st.info("🔐 Безопасность:\n- Код выполняется локально в изолированной среде\n- Запрещены опасные операции (os, sys, exec)\n- Таймаут выполнения: 30 сек")
    
    st.divider()
    st.info("📁 Поддерживаемые форматы: CSV, Excel")

uploaded_file = st.file_uploader("📁 Загрузите датасет", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Загружен: **{uploaded_file.name}** ({df.shape[0]:,} строк × {df.shape[1]} столбцов)")
        
        # Метрики датасета
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📊 Строк", f"{df.shape[0]:,}")
        c2.metric("📐 Столбцов", df.shape[1])
        c3.metric("🔢 Числовых", len(df.select_dtypes(include='number').columns))
        c4.metric("🏷️ Категориальных", len(df.select_dtypes(include=['object', 'category', 'bool']).columns))
        c5.metric("⚠️ Пропусков", f"{df.isnull().sum().sum():,}")
        
        with st.expander("📋 Превью данных (первые 5 строк)"):
            st.dataframe(df.head(), use_container_width=True)
        
        st.markdown("---")
        
        # Запрос пользователя
        st.subheader("💬 Запрос к аналитику")
        user_query = st.text_area(
            "Опишите задачу или оставьте пустым для автоматического анализа",
            placeholder="Примеры:\n• 'Сравни среднюю зарплату по отделам'\n• 'Найди корреляции между переменными'\n• 'Есть ли выбросы в возрасте?'",
            height=90
        )
        
        run_btn = st.button("🚀 Запустить анализ", type="primary", disabled=not api_key, use_container_width=True)
        
        if run_btn:
            if not api_key or len(api_key) < 10:
                st.error("❌ Введите валидный API токен")
                st.stop()
            
            is_auto = not user_query.strip()
            is_safe, msg = check_safety(user_query if not is_auto else "auto")
            if not is_safe:
                st.warning(msg)
                st.stop()
            
            # Инфо о процессе
            with st.expander("🔄 Как работает анализ", expanded=False):
                st.markdown("""
                1. **LLM получает описание данных** (не сырые данные, а статистику + примеры)
                2. **Генерирует Python-код** для решения задачи
                3. **Локальный интерпретатор выполняет код** на ваших реальных данных
                4. **Возвращаются фактические результаты** (не выдуманные!)
                """)
            
            with st.spinner("🤖 Генерация и выполнение кода..."):
                progress = st.progress(0)
                status = st.empty()
                
                status.text("Подготовка контекста...")
                progress.progress(20)
                
                agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                
                status.text("LLM генерирует код...")
                progress.progress(50)
                
                result = agent.generate_and_execute(user_query, df, auto_mode=is_auto)
                
                progress.progress(100)
                status.empty()
            
            st.markdown("---")
            
            if not result['success']:
                st.error(f"❌ {result.get('error', 'Ошибка анализа')}")
                if result.get('details'):
                    with st.expander("🔍 Детали ошибки"):
                        st.code(result['details'])
                if result.get('generated_code'):
                    with st.expander("📜 Сгенерированный код"):
                        st.markdown(f"<div class='code-block'>{result['generated_code']}</div>", unsafe_allow_html=True)
                if result.get('llm_raw'):
                    with st.expander("📄 Сырой ответ LLM"):
                        st.text(result['llm_raw'][:1000])
            else:
                # === ОТЧЁТ ===
                st.subheader("📋 Аналитический отчёт")
                
                if result.get('thought'):
                    with st.expander("💭 Логика анализа", expanded=True):
                        st.markdown(result['thought'])
                
                # Показываем сгенерированный код
                st.markdown("**🐍 Выполненный код:**")
                st.markdown(f"<div class='code-block'>{result['generated_code']}</div>", unsafe_allow_html=True)
                
                # Реальные метрики
                if result.get('metrics_text'):
                    st.markdown("**📊 Реальные результаты:**")
                    st.markdown(f"<div class='metrics-block'>{result['metrics_text']}</div>", unsafe_allow_html=True)
                
                if result.get('output'):
                    st.caption(f"📤 Вывод: {result['output']}")
                
                # === ВИЗУАЛИЗАЦИЯ (Streamlit строит на реальных данных) ===
                st.markdown("---")
                st.subheader("📈 Интерактивная визуализация")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                
                if numeric_cols:
                    tab1, tab2, tab3 = st.tabs(["📊 Гистограмма", "🔗 Корреляции", "📦 Box Plot"])
                    
                    with tab1:
                        col = st.selectbox("Выберите столбец", numeric_cols, key="hist_col")
                        fig = px.histogram(df, x=col, title=f"Распределение: {col}", marginal="box")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        if len(numeric_cols) >= 2:
                            cols_to_corr = st.multiselect("Столбцы для корреляции", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
                            if cols_to_corr:
                                corr = df[cols_to_corr].corr(numeric_only=True)
                                fig = px.imshow(corr, text_auto=True, title="Корреляционная матрица", color_continuous_scale="RdBu_r")
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with tab3:
                        if cat_cols and len(numeric_cols) >= 1:
                            c1, c2 = st.columns(2)
                            cat_col = c1.selectbox("Категория", cat_cols, key="box_cat")
                            num_col = c2.selectbox("Значение", numeric_cols, key="box_num")
                            if df[cat_col].nunique() <= 20:
                                fig = px.box(df, x=cat_col, y=num_col, title=f"{num_col} по {cat_col}", points="outliers")
                                st.plotly_chart(fig, use_container_width=True)
                
                if cat_cols:
                    with st.expander("🏷️ Анализ категориальных переменных"):
                        for col in cat_cols[:5]:
                            if df[col].nunique() <= 15:
                                counts = df[col].value_counts()
                                fig = px.bar(x=counts.index, y=counts.values, title=f"{col}", labels={'x': col, 'y': 'Count'})
                                st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Ошибка загрузки: {type(e).__name__}: {e}")
        with st.expander("🔍 Traceback"):
            st.code(traceback.format_exc())
else:
    st.info("👆 **Загрузите CSV или Excel файл для начала анализа**")
    
st.markdown("---")
st.caption("🤖 AI Analytics Agent | Qwen3.6 • gen-api.ru • Code Interpreter v1.0")
