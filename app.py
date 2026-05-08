import ssl
import os
import re
import json
import base64
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO, BytesIO
import traceback
import ast
from contextlib import redirect_stdout, redirect_stderr
import time
from typing import Optional, Union, List
from datetime import datetime

# OpenAI-compatible client для Qwen
from openai import OpenAI, APIError, AuthenticationError, RateLimitError

# Отключаем предупреждения SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ["CURL_CA_BUNDLE"] = ""

# Настройки Plotly
px.defaults.template = "plotly_white"

# Запрещённые модули
FORBIDDEN_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'httpx',
    'urllib', 'ftplib', 'smtplib', 'paramiko', 'pickle', 'marshal', 'eval',
    'exec', 'compile', 'open', 'input', 'breakpoint', 'import'
}
FORBIDDEN_BUILTINS = {'eval', 'exec', 'compile', 'import', 'open', 'input'}


class SafeCodeExecutor:
    """Безопасный исполнитель кода"""
    
    def __init__(self, df: pd.DataFrame, timeout: int = 30, theme: str = 'plotly_white'):
        self.df = df.copy()
        self.timeout = timeout
        self.output = []
        self.figures = []
        self.theme = theme
        px.defaults.template = theme

    def _safe_globals(self):
        """Безопасное окружение"""
        allowed_modules = {
            'pd': pd, 'pandas': pd,
            'np': np, 'numpy': np,
            'px': px, 'plotly': __import__('plotly'), 'go': go,
            'make_subplots': make_subplots,
            'math': __import__('math'), 're': __import__('re'),
            'json': __import__('json'), 'datetime': __import__('datetime'),
        }

        builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        safe_builtins = {k: v for k, v in builtins_dict.items() if k not in FORBIDDEN_BUILTINS}

        return {
            '__builtins__': safe_builtins,
            'df': self.df,
            'print': lambda *args: self.output.append(' '.join(map(str, args))),
            'save_fig': self._save_fig_callback,
            'display_fig': self._display_fig_callback,
            'create_dashboard': self._create_dashboard_callback,
            'theme': self.theme,
            **allowed_modules
        }

    def _normalize_figure(self, fig) -> go.Figure:
        """Приведение фигуры к стандартному формату"""
        if isinstance(fig, go.Figure):
            fig.update_layout(template=self.theme, height=500, margin=dict(l=40, r=40, t=40, b=40))
            return fig
        elif hasattr(fig, 'to_plotly_json'):
            return go.Figure(fig)
        else:
            try:
                return go.Figure(data=fig)
            except:
                return None

    def _save_fig_callback(self, fig, filename: str = "plot", title: str = None, **layout_kwargs):
        """Сохранение графика"""
        try:
            normalized = self._normalize_figure(fig)
            if normalized:
                if title:
                    normalized.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
                if layout_kwargs:
                    normalized.update_layout(**layout_kwargs)
                self.figures.append({'fig': normalized, 'name': filename, 'title': title})
                return f"✅ График '{filename}' создан"
            return "❌ Ошибка создания графика"
        except Exception as e:
            return f"❌ Ошибка при сохранении графика: {str(e)}"

    def _display_fig_callback(self, fig, title: str = None):
        """Алиас для save_fig"""
        return self._save_fig_callback(fig, filename=f"viz_{len(self.figures)+1}", title=title)

    def _create_dashboard_callback(self, figs: List, titles: List[str] = None, 
                                  subplot_titles: List[str] = None, rows: int = None, cols: int = None):
        """Создание дашборда"""
        if not figs:
            return "❌ Нет графиков для дашборда"
        
        n = len(figs)
        if rows and cols:
            pass
        elif n == 1:
            rows, cols = 1, 1
        elif n <= 2:
            rows, cols = 1, 2
        elif n <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = (n + 1) // 2, 2
        
        dashboard = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles or titles)
        
        for idx, fig in enumerate(figs[:rows*cols]):
            normalized = self._normalize_figure(fig)
            if normalized:
                row = idx // cols + 1
                col = idx % cols + 1
                for trace in normalized.data:
                    dashboard.add_trace(trace, row=row, col=col)
        
        dashboard.update_layout(
            template=self.theme, 
            height=500 * rows,
            title={'text': "Аналитический дашборд", 'x': 0.5, 'xanchor': 'center'},
            showlegend=True
        )
        
        return self._save_fig_callback(dashboard, filename="dashboard", title="Аналитический дашборд")

    def _validate_code(self, code: str) -> tuple:
        """Проверка кода"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {e}"

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = node.module if isinstance(node, ast.ImportFrom) else None
                for alias in (node.names if isinstance(node, ast.Import) else []):
                    mod_name = alias.name.split('.')[0] if isinstance(node, ast.Import) else module.split('.')[0] if module else None
                    if mod_name in FORBIDDEN_MODULES:
                        return False, f"Запрещённый модуль: {mod_name}"
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                if func_name in FORBIDDEN_BUILTINS:
                    return False, f"Запрещённая функция: {func_name}"
        return True, "OK"

    def execute(self, code: str) -> dict:
        """Выполнение кода"""
        result = {'success': False, 'output': [], 'error': None, 'figures': [], 'data_result': None, 'debug_info': {}}

        is_valid, message = self._validate_code(code)
        if not is_valid:
            result['error'] = f"Код отклонён: {message}"
            return result

        try:
            safe_globals = self._safe_globals()
            local_vars = {}

            stdout_capture = StringIO()
            stderr_capture = StringIO()
            
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals, local_vars)

            result['output'] = self.output
            result['figures'] = [item['fig'] if isinstance(item, dict) and 'fig' in item else item 
                                for item in self.figures if isinstance(item, (go.Figure, dict))]
            
            if 'result' in local_vars:
                result['data_result'] = local_vars['result']
            elif 'df_result' in local_vars:
                result['data_result'] = local_vars['df_result']
            
            result['debug_info']['stdout'] = stdout_capture.getvalue()
            result['debug_info']['stderr'] = stderr_capture.getvalue()
            result['success'] = True

        except Exception as e:
            result['error'] = f"Ошибка выполнения: {type(e).__name__}: {str(e)}"
            result['traceback'] = traceback.format_exc()
            result['debug_info']['exception_type'] = type(e).__name__
            result['debug_info']['exception_msg'] = str(e)

        return result


class QwenAnalyticsAgent:
    """Агент с улучшенной обработкой ошибок API"""
    
    def __init__(self, api_key: str, model: str = "qwen3.6-plus"):
        self.api_key = api_key
        self.model = model
        self.df_info = None
        self.api_errors = []
        self.request_history = []
        
        # Инициализация клиента с проверкой
        try:
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                timeout=30.0  # Таймаут 30 секунд
            )
            self.client_initialized = True
        except Exception as e:
            self.client = None
            self.client_initialized = False
            self.init_error = str(e)

    def test_connection(self) -> dict:
        """Тестирование подключения к API"""
        if not self.client_initialized:
            return {
                'success': False,
                'error': f"Не удалось инициализировать клиент: {getattr(self, 'init_error', 'Неизвестная ошибка')}",
                'details': {
                    'api_key_set': bool(self.api_key),
                    'api_key_length': len(self.api_key) if self.api_key else 0,
                    'client_initialized': False
                }
            }
        
        try:
            # Пробный запрос с минимальными токенами
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                temperature=0
            )
            return {
                'success': True,
                'error': None,
                'details': {
                    'api_key_set': bool(self.api_key),
                    'api_key_length': len(self.api_key) if self.api_key else 0,
                    'client_initialized': True,
                    'model': self.model,
                    'response_received': True,
                    'timestamp': datetime.now().isoformat()
                }
            }
        except AuthenticationError as e:
            return {
                'success': False,
                'error': "❌ Ошибка аутентификации",
                'details': {
                    'error_type': 'AuthenticationError',
                    'message': str(e),
                    'possible_causes': [
                        "Неверный API ключ",
                        "Ключ истёк",
                        "Ключ не активирован",
                        "Неправильный формат ключа"
                    ]
                }
            }
        except RateLimitError as e:
            return {
                'success': False,
                'error': "⏱ Превышен лимит запросов",
                'details': {
                    'error_type': 'RateLimitError',
                    'message': str(e),
                    'possible_causes': [
                        "Превышен дневной лимит",
                        "Превышен лимит запросов в минуту",
                        "Недостаточно токенов на счёте"
                    ]
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"❌ Ошибка подключения: {type(e).__name__}",
                'details': {
                    'error_type': type(e).__name__,
                    'message': str(e),
                    'traceback': traceback.format_exc(),
                    'possible_causes': [
                        "Проблемы с интернет-соединением",
                        "API сервер недоступен",
                        "Неверный URL API",
                        "Брандмауэр блокирует соединение"
                    ]
                }
            }

    def prepare_dataset_context(self, df: pd.DataFrame):
        """Подготовка контекста данных"""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_cols': numeric_cols[:10],  # Первые 10
            'categorical_cols': categorical_cols[:10],
            'sample': df.head(2).to_dict(orient='records'),
            'missing': {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
        }

    def generate_code(self, user_query: str, auto_eda: bool = False) -> dict:
        """Генерация кода с полной информацией об ошибках"""
        system_prompt = """Ты эксперт по визуализации данных. Отвечай ТОЛЬКО JSON:
{
"thought": "краткое рассуждение",
"code": "Python код. УЖЕ импортированы: pd, np, px, go, make_subplots. НЕ пиши import!",
"explanation": "что покажут графики"
}

ПРАВИЛА:
1. Всегда вызывай save_fig(fig, 'unique_name', title='...')
2. Используй px для быстрых графиков, go для кастомных
3. result = df.groupby(...).agg(...) для табличных результатов"""

        context = f"Данные: {json.dumps(self.df_info, ensure_ascii=False)}\nЗадача: {user_query}"

        request_info = {
            'timestamp': datetime.now().isoformat(),
            'query': user_query,
            'model': self.model,
            'auto_eda': auto_eda
        }

        try:
            if not self.client_initialized:
                raise Exception(f"Клиент не инициализирован: {getattr(self, 'init_error', 'Unknown')}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": context}
                ],
                temperature=0.2 if auto_eda else 0.3,
                max_tokens=2500,
                response_format={"type": "json_object"}
            )
            
            request_info['response_tokens'] = response.usage.completion_tokens if hasattr(response, 'usage') else None
            request_info['success'] = True
            
            self.request_history.append(request_info)
            
            return {
                'success': True,
                'content': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens if hasattr(response, 'usage') else None,
                    'completion_tokens': response.usage.completion_tokens if hasattr(response, 'usage') else None
                }
            }
            
        except AuthenticationError as e:
            error_info = {
                'type': 'AuthenticationError',
                'message': str(e),
                'details': 'Неверный API ключ или ключ не активен'
            }
            request_info['error'] = error_info
            self.request_history.append(request_info)
            self.api_errors.append(error_info)
            
            return {
                'success': False,
                'error_type': 'AuthenticationError',
                'message': 'Ошибка аутентификации. Проверьте API ключ.',
                'details': str(e),
                'raw_response': None
            }
            
        except RateLimitError as e:
            error_info = {
                'type': 'RateLimitError',
                'message': str(e),
                'details': 'Превышен лимит запросов'
            }
            request_info['error'] = error_info
            self.request_history.append(request_info)
            self.api_errors.append(error_info)
            
            return {
                'success': False,
                'error_type': 'RateLimitError',
                'message': 'Превышен лимит запросов. Подождите немного.',
                'details': str(e),
                'raw_response': None
            }
            
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'traceback': traceback.format_exc()
            }
            request_info['error'] = error_info
            self.request_history.append(request_info)
            self.api_errors.append(error_info)
            
            return {
                'success': False,
                'error_type': type(e).__name__,
                'message': f'Ошибка API: {type(e).__name__}',
                'details': str(e),
                'traceback': traceback.format_exc(),
                'raw_response': None
            }

    def run_analysis(self, user_query: str, df: pd.DataFrame, auto_eda: bool = False, theme: str = 'plotly_white') -> dict:
        """Запуск анализа"""
        self.prepare_dataset_context(df)
        
        # Генерация кода
        gen_result = self.generate_code(user_query, auto_eda)
        
        if not gen_result['success']:
            return {
                'success': False,
                'stage': 'api_call',
                'error': gen_result['message'],
                'error_type': gen_result.get('error_type'),
                'details': gen_result.get('details'),
                'traceback': gen_result.get('traceback'),
                'api_diagnostics': {
                    'model': self.model,
                    'api_key_length': len(self.api_key) if self.api_key else 0,
                    'client_initialized': self.client_initialized
                }
            }
        
        # Парсинг ответа
        try:
            plan = json.loads(gen_result['content'])
        except json.JSONDecodeError as e:
            match = re.search(r'\{[\s\S]*\}', gen_result['content'])
            if match:
                try:
                    plan = json.loads(match.group())
                except:
                    plan = {"error": "Parse error", "code": "print('Error parsing response')"}
            else:
                plan = {"error": "No JSON found", "code": "print('Error')"}
        except Exception as e:
            plan = {"error": f"Parse error: {str(e)}", "code": "print('Error')"}

        code = plan.get('code', '')
        executor = SafeCodeExecutor(df, theme=theme)
        exec_result = executor.execute(code)

        return {
            'success': exec_result['success'],
            'stage': 'execution',
            'thought': plan.get('thought', ''),
            'explanation': plan.get('explanation', ''),
            'code': code,
            'output': exec_result['output'],
            'figures': exec_result['figures'],
            'data_result': exec_result['data_result'],
            'error': exec_result.get('error'),
            'traceback': exec_result.get('traceback'),
            'debug_info': exec_result.get('debug_info', {}),
            'api_usage': gen_result.get('usage'),
            'full_llm_response': gen_result['content']  # Полный ответ от LLM
        }


def check_safety(query: str) -> tuple:
    """Проверка безопасности"""
    forbidden = [r'eval\s*\(', r'exec\s*\(', r'import\s+', r'os\.system', r'__']
    for pattern in forbidden:
        if re.search(pattern, query, re.IGNORECASE):
            return False, f"⚠️ Подозрительный паттерн: {pattern}"
    return True, "✅ Безопасно"


# ================= UI =================
st.set_page_config(page_title="AI Analytics Pro", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .stPlotlyChart {border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .error-box {background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 4px solid #f44336;}
    .warning-box {background-color: #fff3e0; padding: 10px; border-radius: 5px; border-left: 4px solid #ff9800;}
    .success-box {background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50;}
</style>
""", unsafe_allow_html=True)

st.title("AI Analytics Agent Pro")
st.markdown("*🤖 Интеллектуальный анализ данных с детальной диагностикой*")

# Инициализация session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'api_errors_log' not in st.session_state:
    st.session_state.api_errors_log = []

with st.sidebar:
    st.header("⚙️ Настройки")
    
    api_key = st.text_input("🔑 DashScope API Key", type="password", 
                           help="Получите ключ на dashscope.aliyun.com")
    
    if api_key:
        # Кнопка тестирования подключения
        if st.button("🔌 Проверить подключение к API", type="secondary"):
            with st.spinner("Тестирование..."):
                test_agent = QwenAnalyticsAgent(api_key=api_key, model=st.session_state.get('selected_model', 'qwen3.6-plus'))
                test_result = test_agent.test_connection()
                
                if test_result['success']:
                    st.success("✅ Подключение успешно!")
                    st.json(test_result['details'])
                else:
                    st.error(test_result['error'])
                    st.warning("🔍 Детали:")
                    st.json(test_result.get('details', {}))
                    
                    if 'possible_causes' in test_result.get('details', {}):
                        st.info("💡 Возможные причины:")
                        for cause in test_result['details']['possible_causes']:
                            st.markdown(f"• {cause}")
        
        if len(api_key) > 10:
            st.success("✅ Ключ введён")
    
    model = st.selectbox("🧠 Модель", ["qwen3.6-plus", "qwen3.5-plus", "qwen-max"], 
                        index=0, key='selected_model')
    
    st.divider()
    
    # Настройки визуализации
    st.subheader("🎨 Визуализация")
    theme = st.selectbox("Тема графиков", 
                        ['plotly_white', 'plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_dark'],
                        index=0)
    
    st.divider()
    
    # Диагностика
    st.subheader("🔍 Диагностика")
    if st.button("📋 Показать историю ошибок API"):
        if st.session_state.api_errors_log:
            st.error(f"Всего ошибок: {len(st.session_state.api_errors_log)}")
            for idx, err in enumerate(st.session_state.api_errors_log[-5:], 1):
                with st.expander(f"Ошибка #{len(st.session_state.api_errors_log) - 5 + idx}"):
                    st.json(err)
        else:
            st.success("Ошибок не зафиксировано")
    
    st.info("📁 Поддерживаемые форматы: CSV, Excel (.xlsx)")

# Загрузка данных
uploaded_file = st.file_uploader("📁 Загрузите файл с данными", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Загружен: **{uploaded_file.name}**")
        
        # Быстрая статистика
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📊 Строк", f"{df.shape[0]:,}")
        col2.metric("📐 Столбцов", df.shape[1])
        col3.metric("🔢 Числовых", len(df.select_dtypes(include='number').columns))
        col4.metric("🏷️ Категориальных", len(df.select_dtypes(include=['object', 'category', 'bool']).columns))
        col5.metric("⚠️ Пропусков", df.isnull().sum().sum())
        
        with st.expander("📋 Превью данных", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Запрос
        query = st.text_area("💬 Ваш запрос к данным", 
                           placeholder="Примеры:\n• Построй гистограмму распределения столбца 'age'\n• Сравни средние значения 'salary' по группам 'department'", 
                           height=100)
        
        if st.button("🚀 Выполнить анализ", type="primary", 
                    disabled=not (api_key and query)):
            
            if not api_key:
                st.error("❌ Введите API ключ")
                st.stop()
            
            if len(api_key) < 10:
                st.error("❌ API ключ слишком короткий")
                st.stop()
            
            is_safe, msg = check_safety(query)
            if not is_safe:
                st.warning(msg)
                st.stop()
            
            with st.spinner("🤖 Генерация кода и выполнение анализа..."):
                try:
                    agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                    
                    # Тест подключения перед основным запросом
                    test_result = agent.test_connection()
                    if not test_result['success']:
                        st.error("❌ Проблемы с подключением к API:")
                        st.markdown(f"<div class='error-box'>{test_result['error']}</div>", unsafe_allow_html=True)
                        
                        if 'details' in test_result:
                            with st.expander("🔍 Детали ошибки"):
                                st.json(test_result['details'])
                                
                                if 'possible_causes' in test_result['details']:
                                    st.warning("💡 Возможные причины:")
                                    for cause in test_result['details']['possible_causes']:
                                        st.markdown(f"• {cause}")
                                    
                                    st.info("🔧 Что делать:")
                                    st.markdown("""
                                    1. **Проверьте API ключ** - скопируйте его заново из DashScope
                                    2. **Убедитесь, что ключ активен** - проверьте баланс и лимиты
                                    3. **Проверьте интернет-соединение**
                                    4. **Попробуйте другую модель** - возможно, текущая недоступна
                                    """)
                        st.stop()
                    
                    result = agent.run_analysis(query, df, auto_eda=False, theme=theme)
                    
                    # Сохранение в историю
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'query': query,
                        'success': result['success'],
                        'error': result.get('error')
                    })
                    
                    # Логирование ошибок API
                    if not result['success'] and result.get('stage') == 'api_call':
                        st.session_state.api_errors_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'query': query,
                            'error_type': result.get('error_type'),
                            'error': result.get('error'),
                            'details': result.get('details')
                        })
                    
                    st.markdown("---")
                    
                    # Обработка результатов
                    if not result['success']:
                        if result.get('stage') == 'api_call':
                            st.error(f"❌ {result.get('error')}")
                            
                            if result.get('error_type') == 'AuthenticationError':
                                st.markdown("""
                                <div class='error-box'>
                                <strong>🔑 Проблема с аутентификацией</strong><br>
                                Проверьте:
                                <ul>
                                <li>API ключ введён правильно (без пробелов)</li>
                                <li>Ключ активен и не истёк</li>
                                <li>На счёте есть токены</li>
                                <li>Ключ имеет доступ к выбранной модели</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif result.get('error_type') == 'RateLimitError':
                                st.warning("⏱ **Превышен лимит запросов**<br>Подождите несколько минут и попробуйте снова.")
                            
                            if result.get('details'):
                                with st.expander("🔍 Технические детали"):
                                    st.json(result.get('details', {}))
                            
                            if result.get('traceback'):
                                with st.expander("📄 Full Traceback"):
                                    st.code(result['traceback'], language='python')
                            
                            # Диагностическая информация
                            with st.expander("📊 Диагностика API"):
                                st.json(result.get('api_diagnostics', {}))
                        else:
                            st.error(f"❌ Ошибка выполнения: {result.get('error')}")
                            if result.get('traceback'):
                                with st.expander("📄 Stack trace"):
                                    st.code(result['traceback'], language='python')
                    else:
                        # Успешный результат
                        if result.get('thought'):
                            with st.expander("💭 Ход рассуждений агента", expanded=True):
                                st.markdown(result['thought'])
                        
                        with st.expander("📝 Сгенерированный код", expanded=False):
                            st.code(result['code'], language='python')
                        
                        if result.get('output'):
                            with st.expander("📤 Консольный вывод", expanded=True):
                                for line in result['output']:
                                    st.text(line)
                        
                        if result.get('data_result') is not None:
                            st.markdown("### 📋 Результаты вычислений")
                            if isinstance(result['data_result'], pd.DataFrame):
                                st.dataframe(result['data_result'], use_container_width=True)
                            else:
                                st.write(result['data_result'])
                        
                        if result.get('figures'):
                            st.markdown("### 📈 Визуализации")
                            for idx, fig in enumerate(result['figures']):
                                st.plotly_chart(fig, use_container_width=True, key=f"fig_{idx}")
                        
                        if result.get('explanation'):
                            st.info(f"💡 **Интерпретация:** {result['explanation']}")
                        
                        # Показ статистики использования API
                        if result.get('api_usage'):
                            with st.expander("📊 Статистика API"):
                                st.json(result['api_usage'])
                
                except Exception as e:
                    st.error(f"❌ Неожиданная ошибка: {type(e).__name__}: {str(e)}")
                    with st.expander("🔍 Stack trace"):
                        st.code(traceback.format_exc(), language='python')
    
    except Exception as e:
        st.error(f"❌ Ошибка загрузки файла: {type(e).__name__}: {e}")
        with st.expander("🔍 Stack trace"):
            st.code(traceback.format_exc(), language='python')
else:
    st.info("👆 **Загрузите CSV или Excel файл** для начала анализа")
    
    st.markdown("### 💡 Примеры запросов:")
    examples = [
        "Построй гистограмму распределения возраста",
        "Сравни среднюю зарплату по отделам",
        "Найди корреляции между числовыми переменными"
    ]
    for ex in examples:
        st.markdown(f"• `{ex}`")

# Футер
st.markdown("---")
st.caption("🤖 AI Analytics Agent Pro | Диагностика включена")
