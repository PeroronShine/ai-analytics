# app.py - AI Analytics Agent для gen-api.ru Qwen3.6
import ssl
import os
import re
import json
import base64
import requests
import time
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
        """Нормализация фигуры Plotly"""
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
            return f"❌ Ошибка: {str(e)}"

    def _display_fig_callback(self, fig, title: str = None):
        return self._save_fig_callback(fig, filename=f"viz_{len(self.figures)+1}", title=title)

    def _create_dashboard_callback(self, figs: List, titles: List[str] = None, 
                                  subplot_titles: List[str] = None, rows: int = None, cols: int = None):
        """Создание дашборда"""
        if not figs:
            return "❌ Нет графиков"
        
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
        """Валидация кода"""
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
            result['error'] = f"Ошибка: {type(e).__name__}: {str(e)}"
            result['traceback'] = traceback.format_exc()
            result['debug_info']['exception_type'] = type(e).__name__
            result['debug_info']['exception_msg'] = str(e)

        return result


class QwenAnalyticsAgent:
    """Агент для gen-api.ru с авто-анализом"""
    
    def __init__(self, api_key: str, model: str = "qwen-3-6-plus"):
        self.api_key = api_key
        self.model = model
        self.df_info = None
        self.base_url = "https://api.gen-api.ru/api/v1/networks/qwen-3-6-plus"
        
    def test_connection(self) -> dict:
        """Тест подключения"""
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        input_data = {"messages": [{"role": "user", "content": "Hi"}], "is_sync": True, "max_tokens": 10}
        
        try:
            response = requests.post(self.base_url, headers=headers, json=input_data, timeout=30)
            if response.status_code == 200:
                return {'success': True, 'error': None, 'details': {'status': 'OK', 'model': self.model}}
            elif response.status_code == 401:
                return {'success': False, 'error': "❌ Неверный токен", 'details': {'status_code': 401}}
            else:
                return {'success': False, 'error': f"❌ Ошибка {response.status_code}", 'details': response.text}
        except Exception as e:
            return {'success': False, 'error': f"❌ {type(e).__name__}", 'details': str(e)}

    def prepare_dataset_context(self, df: pd.DataFrame):
        """Подготовка контекста данных"""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        recommendations = []
        if numeric_cols:
            recommendations.append(f"📊 Построить гистограммы для: {', '.join(numeric_cols[:3])}")
            if len(numeric_cols) >= 2:
                recommendations.append(f"🔗 Найти корреляции между: {numeric_cols[0]} и {numeric_cols[1]}")
        if categorical_cols:
            recommendations.append(f"🥧 Показать распределение: {categorical_cols[0]}")
            if categorical_cols and numeric_cols:
                recommendations.append(f"📈 Сравнить {numeric_cols[0]} по группам {categorical_cols[0]}")
        if df.isnull().sum().sum() > 0:
            recommendations.append("⚠️ Проанализировать пропущенные значения")
        
        self.df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'sample': df.head(2).to_dict(orient='records'),
            'missing': {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
            'recommendations': recommendations[:5]
        }

    def _generate_auto_eda_prompt(self) -> str:
        """Генерация промпта для авто-анализа"""
        info = self.df_info
        prompt_parts = []
        
        prompt_parts.append("1. Покажи базовую статистику: df.describe() для числовых столбцов, df['col'].value_counts() для категориальных")
        
        if info['numeric_cols']:
            cols = info['numeric_cols'][:3]
            prompt_parts.append(f"2. Построй гистограммы для: {', '.join(cols)} через px.histogram с save_fig()")
            if len(info['numeric_cols']) >= 2:
                prompt_parts.append(f"3. Построй scatter plot: px.scatter(df, x='{info['numeric_cols'][0]}', y='{info['numeric_cols'][1]}') с save_fig()")
            if len(info['numeric_cols']) >= 3:
                prompt_parts.append("4. Если >=3 числовых столбца: покажи корреляционную матрицу через df.corr().style или px.imshow()")
        
        if info['categorical_cols']:
            col = info['categorical_cols'][0]
            prompt_parts.append(f"5. Построй бар-чарт распределения '{col}': px.bar(df['{col}'].value_counts()) с save_fig()")
            if info['categorical_cols'] and info['numeric_cols']:
                prompt_parts.append(f"6. Сравни '{info['numeric_cols'][0]}' по '{info['categorical_cols'][0]}': px.box или px.violin с save_fig()")
        
        if info['missing']:
            prompt_parts.append("7. Визуализируй пропуски: px.bar(pd.DataFrame({'missing': df.isnull().sum()})) с save_fig()")
        
        prompt_parts.append("8. Выведи print() с краткими инсайтами по данным")
        prompt_parts.append("9. Сохрани итоговую таблицу в result = df.describe() или аналогичную")
        
        return "\n".join(prompt_parts)

    def generate_code(self, user_query: Optional[str], auto_eda: bool = False) -> dict:
        """Генерация кода через gen-api.ru - ИСПРАВЛЕНО"""
        
        system_prompt = """Ты эксперт по анализу данных. Отвечай ТОЛЬКО валидным JSON:
{
"thought": "краткое рассуждение",
"code": "Python код. УЖЕ импортированы: pd, np, px, go, make_subplots. НЕ пиши import!",
"explanation": "что покажут результаты"
}

ПРАВИЛА:
1. Всегда save_fig(fig, 'unique_name', title='...') для графиков
2. Используй px для быстрых графиков, go для кастомных  
3. result = ... для табличных итогов
4. print() для текстовых выводов"""

        if auto_eda:
            context = f"АВТО-АНАЛИЗ ДАТАСЕТА.\nДанные: {json.dumps(self.df_info, ensure_ascii=False)}\n\nВыполни пошагово:\n{self._generate_auto_eda_prompt()}"
        else:
            context = f"Данные: {json.dumps(self.df_info, ensure_ascii=False)}\nЗадача: {user_query}"

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
                timeout=90 if auto_eda else 60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # === ИСПРАВЛЕНИЕ: правильная обработка ответа gen-api.ru ===
                content = None
                
                # Вариант 1: поле "response" (массив строк JSON)
                if "response" in result and isinstance(result["response"], list) and len(result["response"]) > 0:
                    response_str = result["response"][0]
                    try:
                        # Пробуем распарсить строку как JSON
                        parsed = json.loads(response_str)
                        content = json.dumps(parsed, ensure_ascii=False)
                    except:
                        content = response_str
                
                # Вариант 2: поле "output"
                elif "output" in result:
                    content = result["output"]
                
                # Вариант 3: OpenAI-style (choices)
                elif "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                
                # Вариант 4: поле "result"
                elif "result" in result:
                    content = result["result"]
                
                # Если ничего не нашли
                if content is None:
                    content = json.dumps(result, ensure_ascii=False)
                
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

    def run_analysis(self, user_query: Optional[str], df: pd.DataFrame, auto_eda: bool = False, theme: str = 'plotly_white') -> dict:
        """Запуск анализа"""
        self.prepare_dataset_context(df)
        gen_result = self.generate_code(user_query, auto_eda)
        
        if not gen_result['success']:
            return {'success': False, 'stage': 'api_call', 'error': gen_result.get('message'), 'error_type': gen_result.get('error_type'), 'details': gen_result.get('details')}
        
        try:
            plan = json.loads(gen_result['content'])
        except:
            match = re.search(r'\{[\s\S]*\}', gen_result['content'])
            plan = json.loads(match.group()) if match else {"error": "Parse error", "code": "print('Error')"}

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
            'recommendations': self.df_info.get('recommendations', []) if auto_eda else []
        }


def check_safety(query: str) -> tuple:
    """Проверка безопасности"""
    forbidden = [r'eval\s*\(', r'exec\s*\(', r'import\s+', r'os\.system']
    for pattern in forbidden:
        if re.search(pattern, query, re.IGNORECASE):
            return False, f"⚠️ Подозрительный паттерн"
    return True, "✅ Безопасно"


def download_plotly_fig(fig, filename: str, format: str = 'png'):
    """Экспорт графика"""
    try:
        if format == 'png':
            img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            return base64.b64encode(img_bytes).decode()
        elif format == 'html':
            html = fig.to_html(full_html=False, include_plotlyjs='cdn')
            return base64.b64encode(html.encode()).decode()
    except:
        pass
    return None


# ================= UI =================
st.set_page_config(page_title="AI Analytics Pro", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stPlotlyChart {border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .error-box {background: #ffebee; padding: 10px; border-radius: 5px; border-left: 4px solid #f44336;}
    .success-box {background: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 4px solid #4caf50;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Analytics Agent Pro")
st.markdown("*Авто-анализ данных на Qwen3.6 • gen-api.ru*")

# Session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'api_errors_log' not in st.session_state:
    st.session_state.api_errors_log = []
if 'auto_analysis_result' not in st.session_state:
    st.session_state.auto_analysis_result = None

with st.sidebar:
    st.header("⚙️ Настройки")
    
    api_key = st.text_input("🔑 Gen-API.ru Token", type="password")
    
    if api_key and st.button("🔌 Проверить подключение"):
        with st.spinner("Тест..."):
            test = QwenAnalyticsAgent(api_key=api_key).test_connection()
            if test['success']:
                st.success("✅ OK")
                st.json(test['details'])
            else:
                st.error(test['error'])
                if 'details' in test:
                    st.json(test['details'])
    
    model = st.selectbox("🧠 Модель", ["qwen-3-6-plus", "qwen-3-5-plus", "qwen-max"], index=0)
    
    st.divider()
    theme = st.selectbox("🎨 Тема графиков", ['plotly_white', 'plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_dark'])
    
    st.divider()
    st.info("📁 CSV, Excel")

# Загрузка файла
uploaded_file = st.file_uploader("📁 Загрузите файл", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Загружен: {uploaded_file.name}")
        
        # Статистика
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📊 Строк", f"{df.shape[0]:,}")
        c2.metric("📐 Столбцов", df.shape[1])
        c3.metric("🔢 Числовых", len(df.select_dtypes(include='number').columns))
        c4.metric("🏷️ Категориальных", len(df.select_dtypes(include=['object', 'category', 'bool']).columns))
        c5.metric("⚠️ Пропусков", df.isnull().sum().sum())
        
        with st.expander("📋 Превью"):
            st.dataframe(df.head(), use_container_width=True)
        
        # === КНОПКИ АНАЛИЗА ===
        col_btn1, col_btn2 = st.columns([1, 1])
        
        with col_btn1:
            query = st.text_area("💬 Ваш запрос", placeholder="Например: сравни зарплату по отделам", height=70)
            run_manual = st.button("🚀 Анализ по запросу", type="primary", disabled=not (api_key and query))
        
        with col_btn2:
            st.markdown("### 🎯 Авто-анализ")
            st.markdown("*Без ввода запроса — агент сам изучит данные*")
            run_auto = st.button("🔍 Запустить авто-анализ", type="secondary", disabled=not api_key, use_container_width=True)
        
        # === ОБРАБОТКА ===
        if run_manual or run_auto:
            if not api_key or len(api_key) < 10:
                st.error("❌ Введите валидный токен")
                st.stop()
            
            is_safe, msg = check_safety(query if not run_auto else "auto")
            if not is_safe:
                st.warning(msg)
                st.stop()
            
            with st.spinner("🤖 Генерация анализа..."):
                # Прогресс-бар для авто-анализа
                if run_auto:
                    progress = st.progress(0)
                    status = st.empty()
                    steps = ["Подготовка данных...", "Генерация кода...", "Выполнение...", "Визуализация..."]
                
                agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                
                if run_auto and 'progress' in locals():
                    progress.progress(25)
                    status.text(steps[0])
                
                # Тест подключения
                test = agent.test_connection()
                if not test['success']:
                    st.error(f"❌ {test['error']}")
                    if 'details' in test:
                        with st.expander("🔍 Детали"):
                            st.json(test['details'])
                    st.stop()
                
                if run_auto and 'progress' in locals():
                    progress.progress(50)
                    status.text(steps[1])
                
                result = agent.run_analysis(None if run_auto else query, df, auto_eda=run_auto, theme=theme)
                
                if run_auto and 'progress' in locals():
                    progress.progress(100)
                    status.text("✅ Готово!")
                    time.sleep(0.5)
                    progress.empty()
                    status.empty()
                
                # Сохранение авто-результата
                if run_auto:
                    st.session_state.auto_analysis_result = result
                
                st.markdown("---")
                
                # Обработка ошибок
                if not result['success']:
                    st.error(f"❌ {result.get('error', 'Ошибка')}")
                    if result.get('details'):
                        with st.expander("🔍 Детали"):
                            st.json(result['details'])
                    if result.get('traceback'):
                        with st.expander("📄 Traceback"):
                            st.code(result['traceback'])
                else:
                    # Успех
                    if result.get('thought'):
                        with st.expander("💭 Рассуждения агента", expanded=run_auto):
                            st.markdown(result['thought'])
                    
                    # === КОД С ПРОВЕРКОЙ ===
                    code = result.get('code', '')
                    if code and code.strip():
                        with st.expander("📝 Сгенерированный код", expanded=False):
                            st.code(code, language='python')
                            st.caption("💡 Совет: скопируйте и модифицируйте под свои нужды")
                    else:
                        st.warning("⚠️ Код не сгенерирован или пустой")
                        with st.expander("🔍 Отладка"):
                            st.write("**Полный ответ от агента:**")
                            st.json(result)
                    
                    if result.get('output'):
                        with st.expander("📤 Консольный вывод", expanded=True):
                            for line in result['output']:
                                st.text(line)
                    
                    if result.get('data_result') is not None:
                        st.markdown("### 📋 Результаты вычислений")
                        if isinstance(result['data_result'], pd.DataFrame):
                            st.dataframe(result['data_result'], use_container_width=True)
                            csv = result['data_result'].to_csv(index=False)
                            st.download_button("📥 Скачать CSV", csv, "result.csv", "text/csv")
                        else:
                            st.write(result['data_result'])
                    
                    if result.get('figures'):
                        st.markdown("### 📈 Визуализации")
                        if len(result['figures']) > 1:
                            tabs = st.tabs([f"📊 {i+1}" for i in range(len(result['figures']))])
                            for tab, fig in zip(tabs, result['figures']):
                                with tab:
                                    st.plotly_chart(fig, use_container_width=True)
                                    png = download_plotly_fig(fig, "plot", 'png')
                                    if png:
                                        st.download_button("📥 PNG", base64.b64decode(png), f"plot.png", "image/png")
                        else:
                            st.plotly_chart(result['figures'][0], use_container_width=True)
                    
                    if result.get('explanation'):
                        st.info(f"💡 **Интерпретация:** {result['explanation']}")
                    
                    # === РЕКОМЕНДАЦИИ для авто-анализа ===
                    if run_auto and result.get('recommendations'):
                        st.markdown("### 🔄 Что ещё можно изучить?")
                        for rec in result['recommendations']:
                            st.markdown(f"• {rec}")
                        st.caption("💡 Нажмите на рекомендацию, скопируйте и вставьте в поле запроса выше")
        
        # === Кэшированный авто-анализ ===
        if st.session_state.auto_analysis_result and not (run_manual or run_auto):
            if st.expander("📦 Показать последний авто-анализ", expanded=False):
                result = st.session_state.auto_analysis_result
                if result.get('figures'):
                    for fig in result['figures']:
                        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Ошибка: {type(e).__name__}: {e}")
        with st.expander("🔍 Traceback"):
            st.code(traceback.format_exc())
else:
    st.info("👆 Загрузите CSV/Excel файл для начала")
    st.markdown("### 💡 Примеры запросов:")
    for ex in ["Гистограмма возраста", "Зарплата по отделам", "Корреляции переменных", "Поиск выбросов"]:
        st.markdown(f"• `{ex}`")

st.markdown("---")
st.caption("🤖 AI Analytics Pro | Qwen3.6 • gen-api.ru")
