# app.py - AI Analytics Agent для gen-api.ru Qwen3.6
import ssl
import os
import re
import json
import base64
import requests
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
from datetime import datetime
from typing import Optional, Union, List

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

# Запрещённые модули для безопасного выполнения
FORBIDDEN_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'httpx',
    'urllib', 'ftplib', 'smtplib', 'paramiko', 'pickle', 'marshal', 'eval',
    'exec', 'compile', 'open', 'input', 'breakpoint', 'import'
}
FORBIDDEN_BUILTINS = {'eval', 'exec', 'compile', 'import', 'open', 'input'}


class SafeCodeExecutor:
    """Безопасный исполнитель кода с поддержкой визуализаций"""
    
    def __init__(self, df: pd.DataFrame, timeout: int = 30, theme: str = 'plotly_white'):
        self.df = df.copy()
        self.timeout = timeout
        self.output = []
        self.figures = []
        self.theme = theme
        px.defaults.template = theme

    def _safe_globals(self):
        """Безопасное окружение для exec()"""
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
        """Приведение фигуры к стандартному формату Plotly"""
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
        """Сохранение графика в коллекцию"""
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
        """Создание дашборда из нескольких графиков"""
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
        """Проверка кода на запрещённые операции"""
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
        """Выполнение кода в безопасном окружении"""
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
    """Агент для gen-api.ru Qwen3.6"""
    
    def __init__(self, api_key: str, model: str = "qwen-3-6-plus"):
        self.api_key = api_key
        self.model = model
        self.df_info = None
        self.base_url = "https://api.gen-api.ru/api/v1/networks/qwen-3-6-plus"
        
    def test_connection(self) -> dict:
        """Тестирование подключения к API"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        input_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "is_sync": True,
            "max_tokens": 10
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=input_data,
                timeout=30
            )
            
            if response.status_code == 200:
                return {
                    'success': True,
                    'error': None,
                    'details': {
                        'status_code': response.status_code,
                        'api_key_valid': True,
                        'model': self.model,
                        'timestamp': datetime.now().isoformat()
                    }
                }
            elif response.status_code == 401:
                return {
                    'success': False,
                    'error': "❌ Неверный API токен",
                    'details': {
                        'status_code': response.status_code,
                        'response': response.json() if response.text else "Empty response",
                        'possible_causes': [
                            "Неверный токен",
                            "Токен не активирован",
                            "Токен истёк",
                            "Неправильный формат токена"
                        ]
                    }
                }
            elif response.status_code == 429:
                return {
                    'success': False,
                    'error': "⏱ Превышен лимит запросов",
                    'details': {
                        'status_code': response.status_code,
                        'response': response.json() if response.text else "Rate limited",
                        'possible_causes': [
                            "Превышен лимит в минуту",
                            "Превышен дневной лимит",
                            "Недостаточно токенов на счёте"
                        ]
                    }
                }
            else:
                return {
                    'success': False,
                    'error': f"❌ Ошибка API: {response.status_code}",
                    'details': {
                        'status_code': response.status_code,
                        'response': response.json() if response.text else response.text
                    }
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error': "❌ Нет подключения к интернету",
                'details': {
                    'error_type': 'ConnectionError',
                    'possible_causes': [
                        "Проверьте интернет-соединение",
                        "Возможно, API сервер временно недоступен",
                        "Брандмауэр может блокировать соединение"
                    ]
                }
            }
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': "⏱ Таймаут подключения",
                'details': {
                    'error_type': 'Timeout',
                    'possible_causes': [
                        "Медленное интернет-соединение",
                        "Сервер перегружен",
                        "Попробуйте увеличить таймаут"
                    ]
                }
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"❌ Ошибка: {type(e).__name__}",
                'details': {
                    'error_type': type(e).__name__, 
                    'message': str(e),
                    'traceback': traceback.format_exc()
                }
            }

    def prepare_dataset_context(self, df: pd.DataFrame):
        """Подготовка контекста данных для LLM"""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_cols': numeric_cols[:10],
            'categorical_cols': categorical_cols[:10],
            'sample': df.head(2).to_dict(orient='records'),
            'missing': {k: int(v) for k, v in df.isnull().sum().items() if v > 0}
        }

    def generate_code(self, user_query: str, auto_eda: bool = False) -> dict:
        """Генерация кода через gen-api.ru"""
        
        system_prompt = """Ты эксперт по визуализации данных. Отвечай ТОЛЬКО валидным JSON:
{
"thought": "краткое рассуждение о подходе к задаче",
"code": "Python код. УЖЕ импортированы: pd, np, px, go, make_subplots. НЕ пиши import! Используй: df, save_fig(fig, 'name', title='...'), print(), result",
"explanation": "что покажут графики и как интерпретировать результаты"
}

ПРАВИЛА для кода:
1. Всегда вызывай save_fig(fig, 'unique_name', title='Заголовок') для каждого графика
2. Используй px для быстрых графиков, go для кастомных
3. Добавляй подписи осей: labels={'x': '...', 'y': '...'}
4. Для нескольких графиков: create_dashboard([fig1, fig2], titles=['A', 'B'])
5. result = df.groupby(...).agg(...) для табличных результатов
6. Не используй plt.show() - графики отображаются через save_fig()"""

        context = f"Данные: {json.dumps(self.df_info, ensure_ascii=False)}\nЗадача: {user_query}"
        
        if auto_eda:
            context += "\n\nСоздай комплексную визуализацию: 1) гистограммы числовых переменных, 2) корреляционную матрицу если >1 числового столбца, 3) бар-чарты для категориальных, 4) scatter plot для пар числовых"

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        input_data = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": context}
            ],
            "is_sync": True,
            "temperature": 0.2 if auto_eda else 0.3,
            "top_p": 0.9,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=input_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Извлекаем ответ из структуры gen-api.ru
                if 'output' in result:
                    content = result['output']
                elif 'choices' in result and len(result['choices']) > 0:
                    content = result['choices'][0].get('message', {}).get('content', '')
                elif 'result' in result:
                    content = result['result']
                else:
                    content = json.dumps(result)
                
                return {
                    'success': True,
                    'content': content,
                    'raw_response': result
                }
            else:
                error_details = response.text
                try:
                    error_details = response.json()
                except:
                    pass
                    
                return {
                    'success': False,
                    'error_type': f'HTTP_{response.status_code}',
                    'message': f'Ошибка API: {response.status_code}',
                    'details': error_details,
                    'status_code': response.status_code
                }
                
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error_type': 'Timeout',
                'message': 'Превышено время ожидания ответа от API',
                'details': 'Попробуйте повторить запрос или упростите задачу'
            }
        except requests.exceptions.ConnectionError:
            return {
                'success': False,
                'error_type': 'ConnectionError',
                'message': 'Нет подключения к API серверу',
                'details': 'Проверьте интернет-соединение'
            }
        except Exception as e:
            return {
                'success': False,
                'error_type': type(e).__name__,
                'message': f'Ошибка: {str(e)}',
                'details': traceback.format_exc()
            }

    def run_analysis(self, user_query: str, df: pd.DataFrame, auto_eda: bool = False, theme: str = 'plotly_white') -> dict:
        """Запуск полного анализа"""
        self.prepare_dataset_context(df)
        
        # Генерация кода
        gen_result = self.generate_code(user_query, auto_eda)
        
        if not gen_result['success']:
            return {
                'success': False,
                'stage': 'api_call',
                'error': gen_result.get('message', 'Неизвестная ошибка'),
                'error_type': gen_result.get('error_type'),
                'details': gen_result.get('details'),
                'status_code': gen_result.get('status_code')
            }
        
        # Парсинг JSON ответа
        try:
            plan = json.loads(gen_result['content'])
        except json.JSONDecodeError:
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
            'full_llm_response': gen_result['content']
        }


def check_safety(query: str) -> tuple:
    """Проверка безопасности запроса"""
    forbidden = [r'eval\s*\(', r'exec\s*\(', r'import\s+', r'os\.system', r'__']
    for pattern in forbidden:
        if re.search(pattern, query, re.IGNORECASE):
            return False, f"⚠️ Подозрительный паттерн: {pattern}"
    return True, "✅ Безопасно"


def download_plotly_fig(fig, filename: str, format: str = 'png'):
    """Конвертация Plotly фигуры для скачивания"""
    try:
        if format == 'png':
            img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            return base64.b64encode(img_bytes).decode()
        elif format == 'html':
            html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            return base64.b64encode(html.encode()).decode()
    except Exception as e:
        st.warning(f"⚠️ Не удалось экспортировать график: {e}")
    return None


# ================= UI =================
st.set_page_config(page_title="AI Analytics Pro", page_icon="🤖", layout="wide")

# Кастомные стили
st.markdown("""
<style>
    .stPlotlyChart {border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .error-box {background-color: #ffebee; padding: 10px; border-radius: 5px; border-left: 4px solid #f44336;}
    .warning-box {background-color: #fff3e0; padding: 10px; border-radius: 5px; border-left: 4px solid #ff9800;}
    .success-box {background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Analytics Agent Pro")
st.markdown("*Интеллектуальный анализ данных на Qwen3.6-plus")

# Инициализация session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'api_errors_log' not in st.session_state:
    st.session_state.api_errors_log = []

with st.sidebar:
    st.header("⚙️ Настройки")
    
    api_key = st.text_input("🔑 Gen-API.ru Token", type="password", 
                           help="Получите токен в личном кабинете gen-api.ru")
    
    if api_key:
        if st.button("🔌 Проверить подключение", type="secondary"):
            with st.spinner("Тестирование..."):
                test_agent = QwenAnalyticsAgent(api_key=api_key, model=st.session_state.get('selected_model', 'qwen-3-6-plus'))
                test_result = test_agent.test_connection()
                
                if test_result['success']:
                    st.success("✅ Подключение успешно!")
                    st.json(test_result['details'])
                else:
                    st.error(test_result['error'])
                    if 'details' in test_result:
                        with st.expander("🔍 Детали"):
                            st.json(test_result['details'])
                        if 'possible_causes' in test_result['details']:
                            st.info("💡 Возможные причины:")
                            for cause in test_result['details']['possible_causes']:
                                st.markdown(f"• {cause}")
        
        if len(api_key) > 10:
            st.success("✅ Токен введён")
    
    model = st.selectbox("🧠 Модель", ["qwen-3-6-plus", "qwen-3-5-plus", "qwen-max"], 
                        index=0, key='selected_model')
    
    st.divider()
    
    st.subheader("🎨 Визуализация")
    theme = st.selectbox("Тема графиков", 
                        ['plotly_white', 'plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_dark'],
                        index=0)
    
    st.divider()
    
    st.subheader("🔍 Диагностика")
    if st.button("📋 История ошибок"):
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
                st.error("❌ Введите API токен")
                st.stop()
            
            if len(api_key) < 10:
                st.error("❌ Токен слишком короткий")
                st.stop()
            
            is_safe, msg = check_safety(query)
            if not is_safe:
                st.warning(msg)
                st.stop()
            
            with st.spinner("🤖 Генерация кода и выполнение анализа..."):
                try:
                    agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                    
                    # Тест подключения
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
                                    1. **Проверьте токен** - скопируйте заново из gen-api.ru
                                    2. **Убедитесь, что токен активен** - проверьте личный кабинет
                                    3. **Проверьте баланс** - достаточно ли токенов
                                    4. **Попробуйте другую модель** - возможно, текущая недоступна
                                    """)
                        st.stop()
                    
                    result = agent.run_analysis(query, df, auto_eda=False, theme=theme)
                    
                    # Логирование
                    st.session_state.analysis_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'query': query,
                        'success': result['success'],
                        'error': result.get('error')
                    })
                    
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
                            
                            if result.get('error_type') == 'HTTP_401':
                                st.markdown("""
                                <div class='error-box'>
                                <strong>🔑 Проблема с аутентификацией</strong><br>
                                Проверьте:
                                <ul>
                                <li>Токен введён правильно (без пробелов)</li>
                                <li>Токен активен в личном кабинете gen-api.ru</li>
                                <li>На счёте есть токены</li>
                                <li>Токен имеет доступ к выбранной модели</li>
                                </ul>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            elif result.get('error_type') == 'HTTP_429':
                                st.warning("⏱ **Превышен лимит запросов**<br>Подождите немного и попробуйте снова.")
                            
                            if result.get('details'):
                                with st.expander("🔍 Технические детали"):
                                    st.json(result.get('details', {}))
                            
                            if result.get('traceback'):
                                with st.expander("📄 Full Traceback"):
                                    st.code(result['traceback'], language='python')
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
        "Найди корреляции между числовыми переменными",
        "Покажи box plot для выявления выбросов",
        "Визуализируй динамику продаж по месяцам"
    ]
