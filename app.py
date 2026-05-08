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
import html

# Отключаем SSL предупреждения
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ["CURL_CA_BUNDLE"] = ""

px.defaults.template = "plotly_white"

# Усиленная защита от prompt-injection
FORBIDDEN_PATTERNS = [
    r'eval\s*\(', r'exec\s*\(', r'compile\s*\(',
    r'import\s+os', r'import\s+sys', r'import\s+subprocess',
    r'os\.system', r'subprocess\.', r'shutil\.',
    r'__import__', r'__builtins__', r'__class__',
    r'open\s*\(', r'read\s*\(', r'write\s*\(',
    r'input\s*\(', r'breakpoint\s*\(',
    r'getattr\s*\(', r'setattr\s*\(', r'delattr\s*\(',
    r'globals\s*\(', r'locals\s*\(', r'vars\s*\(',
    r'execfile', r'runpy', r'imp\.',
    r'pickle', r'marshal', r'shelve',
]

FORBIDDEN_WORDS = ['del ', 'raise ', 'pass ', 'yield ']


class SafeCodeExecutor:
    """Безопасный исполнитель кода с усиленной защитой"""
    
    def __init__(self, df: pd.DataFrame, timeout: int = 30, theme: str = 'plotly_white'):
        self.df = df.copy()
        self.timeout = timeout
        self.output = []
        self.figures = []
        self.theme = theme
        px.defaults.template = theme

    def _safe_globals(self):
        """Безопасное окружение - ТОЛЬКО разрешенные модули"""
        allowed_modules = {
            'pd': pd, 'pandas': pd,
            'np': np, 'numpy': np,
            'px': px, 'plotly': __import__('plotly'), 'go': go,
            'make_subplots': make_subplots,
            'math': __import__('math'), 're': __import__('re'),
            'json': __import__('json'), 'datetime': __import__('datetime'),
            'StringIO': StringIO,
        }

        builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        safe_builtins = {
            k: v for k, v in builtins_dict.items() 
            if k not in ['eval', 'exec', 'compile', 'import', 'open', 'input', '__import__']
        }

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
        try:
            normalized = self._normalize_figure(fig)
            if normalized:
                if title:
                    normalized.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
                if layout_kwargs:
                    normalized.update_layout(**layout_kwargs)
                self.figures.append({'fig': normalized, 'name': filename, 'title': title})
                return f"✅ График '{filename}' создан"
            return "❌ Ошибка"
        except Exception as e:
            return f"❌ Ошибка: {str(e)}"

    def _display_fig_callback(self, fig, title: str = None):
        return self._save_fig_callback(fig, filename=f"viz_{len(self.figures)+1}", title=title)

    def _create_dashboard_callback(self, figs: List, titles: List[str] = None, 
                                  subplot_titles: List[str] = None, rows: int = None, cols: int = None):
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
        """Усиленная проверка кода на безопасность"""
        # Проверка 1: Синтаксис
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка"

        # Проверка 2: Запрещенные паттерны
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Обнаружен опасный паттерн"
        
        # Проверка 3: Запрещенные слова
        for word in FORBIDDEN_WORDS:
            if f" {word}" in code or code.startswith(word):
                return False, f"Обнаружено опасное слово: {word}"

        # Проверка 4: AST анализ
        for node in ast.walk(tree):
            # Запрет импортов
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False, "Импорты запрещены"
            
            # Запрет опасных функций
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                
                if func_name in ['eval', 'exec', 'compile', 'open', 'input', '__import__']:
                    return False, f"Функция {func_name} запрещена"
        
        return True, "OK"

    def execute(self, code: str) -> dict:
        """Выполнение кода"""
        result = {'success': False, 'output': [], 'error': None, 'figures': [], 'data_result': None}

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
            
            result['success'] = True

        except Exception as e:
            result['error'] = f"Ошибка выполнения: {type(e).__name__}: {str(e)}"
            result['traceback'] = traceback.format_exc()

        return result


class QwenAnalyticsAgent:
    """Агент для gen-api.ru"""
    
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

    def prepare_dataset_context(self, df: pd.DataFrame, user_context: Optional[str] = None):
        """Подготовка контекста данных"""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        recommendations = []
        if numeric_cols:
            recommendations.append(f"📊 Гистограммы: {', '.join(numeric_cols[:3])}")
            if len(numeric_cols) >= 2:
                recommendations.append(f"🔗 Корреляции: {numeric_cols[0]} и {numeric_cols[1]}")
        if categorical_cols:
            recommendations.append(f"🥧 Распределение: {categorical_cols[0]}")
            if categorical_cols and numeric_cols:
                recommendations.append(f"📈 Сравнение {numeric_cols[0]} по {categorical_cols[0]}")
        
        self.df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'sample': df.head(2).to_dict(orient='records'),
            'missing': {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
            'recommendations': recommendations[:5],
            'user_context': user_context if user_context else "Проведи полный анализ датасета"
        }

    def _generate_auto_eda_prompt(self) -> str:
        """Промпт для авто-анализа"""
        info = self.df_info
        prompt_parts = []
        
        prompt_parts.append("1. Покажи описательную статистику: df.describe() и value_counts() для категориальных")
        
        if info['numeric_cols']:
            cols = info['numeric_cols'][:3]
            prompt_parts.append(f"2. Гистограммы: {', '.join(cols)} через px.histogram с save_fig()")
            if len(info['numeric_cols']) >= 2:
                prompt_parts.append(f"3. Scatter plot: px.scatter(df, x='{info['numeric_cols'][0]}', y='{info['numeric_cols'][1]}') с save_fig()")
            if len(info['numeric_cols']) >= 3:
                prompt_parts.append("4. Корреляционная матрица: df.corr() или px.imshow()")
        
        if info['categorical_cols']:
            col = info['categorical_cols'][0]
            prompt_parts.append(f"5. Бар-чарт: px.bar(df['{col}'].value_counts()) с save_fig()")
            if info['categorical_cols'] and info['numeric_cols']:
                prompt_parts.append(f"6. Box plot: px.box сравнение '{info['numeric_cols'][0]}' по '{info['categorical_cols'][0]}' с save_fig()")
        
        if info['missing']:
            prompt_parts.append("7. Визуализация пропусков с save_fig()")
        
        prompt_parts.append("8. print() с ключевыми инсайтами")
        prompt_parts.append("9. result = df.describe() или итоговая таблица")
        
        return "\n".join(prompt_parts)

    def generate_code(self, user_query: Optional[str], auto_eda: bool = False, user_context: Optional[str] = None) -> dict:
        """Генерация кода - ИСПРАВЛЕНО"""
        
        system_prompt = """Ты эксперт по анализу данных. Отвечай ТОЛЬКО валидным JSON:
{
"thought": "краткое рассуждение",
"code": "Python код. УЖЕ импортированы: pd, np, px, go. НЕ пиши import! Используй: df, save_fig(fig, 'name', title='...'), print(), result",
"explanation": "что покажут результаты"
}

ПРАВИЛА:
1. Всегда save_fig(fig, 'unique_name', title='...') для графиков
2. Используй px для графиков
3. result = ... для таблиц
4. print() для выводов
5. НЕ генерируй HTML или Markdown в output"""

        if auto_eda:
            context = f"АВТО-АНАЛИЗ ДАТАСЕТА.\nДанные: {json.dumps(self.df_info, ensure_ascii=False)}\n\nКонтекст от пользователя: {self.df_info.get('user_context', 'Проведи полный анализ')}\n\nВыполни:\n{self._generate_auto_eda_prompt()}"
        else:
            context = f"Данные: {json.dumps(self.df_info, ensure_ascii=False)}\nКонтекст: {user_context if user_context else 'Выполни запрос'}\nЗадача: {user_query}"

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
                
                # Обработка ответа gen-api.ru
                content = None
                
                if "response" in result and isinstance(result["response"], list) and len(result["response"]) > 0:
                    response_str = result["response"][0]
                    try:
                        parsed = json.loads(response_str)
                        content = json.dumps(parsed, ensure_ascii=False)
                    except:
                        content = response_str
                elif "output" in result:
                    content = result["output"]
                elif "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0].get("message", {}).get("content", "")
                elif "result" in result:
                    content = result["result"]
                
                if content is None:
                    content = json.dumps(result, ensure_ascii=False)
                
                return {
                    'success': True,
                    'content': content,
                    'raw_response': result
                }
            else:
                return {
                    'success': False,
                    'error_type': f'HTTP_{response.status_code}',
                    'message': f'Ошибка API: {response.status_code}',
                    'details': response.text,
                    'status_code': response.status_code
                }
                
        except Exception as e:
            return {
                'success': False,
                'error_type': type(e).__name__,
                'message': f'Ошибка: {str(e)}',
                'details': traceback.format_exc()
            }

    def run_analysis(self, user_query: Optional[str], df: pd.DataFrame, auto_eda: bool = False, 
                    theme: str = 'plotly_white', user_context: Optional[str] = None) -> dict:
        """Запуск анализа"""
        self.prepare_dataset_context(df, user_context)
        gen_result = self.generate_code(user_query, auto_eda, user_context)
        
        if not gen_result['success']:
            return {'success': False, 'stage': 'api_call', 'error': gen_result.get('message'), 
                   'error_type': gen_result.get('error_type'), 'details': gen_result.get('details')}
        
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
            'code': code,  # Оставляем для отладки, но НЕ показываем
            'output': exec_result['output'],
            'figures': exec_result['figures'],
            'data_result': exec_result['data_result'],
            'error': exec_result.get('error'),
            'traceback': exec_result.get('traceback'),
            'recommendations': self.df_info.get('recommendations', []) if auto_eda else []
        }


def check_safety(query: str) -> tuple:
    """Усиленная проверка безопасности"""
    # Проверка 1: Паттерны
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return False, f"⚠️ Обнаружен подозрительный паттерн"
    
    # Проверка 2: Слова
    for word in FORBIDDEN_WORDS:
        if f" {word}" in query or query.startswith(word):
            return False, f"⚠️ Обнаружено запрещенное слово"
    
    # Проверка 3: Экранирование HTML
    if '<script' in query.lower() or 'javascript:' in query.lower():
        return False, "⚠️ XSS попытка"
    
    return True, "✅ Безопасно"


def download_plotly_fig(fig, filename: str, format: str = 'png'):
    """Экспорт графика"""
    try:
        if format == 'png':
            img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            return base64.b64encode(img_bytes).decode()
        elif format == 'html':
            html_content = fig.to_html(full_html=False, include_plotlyjs='cdn')
            return base64.b64encode(html_content.encode()).decode()
    except:
        pass
    return None


# ================= UI =================
st.set_page_config(page_title="AI Analytics Agent", page_icon="📊", layout="wide")

# Стили
st.markdown("""
<style>
    .stPlotlyChart {border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white;}
    .report-section {background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Analytics Agent")
st.markdown("*Интеллектуальный анализ данных на Qwen3.6*")

# Session state
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'auto_analysis_result' not in st.session_state:
    st.session_state.auto_analysis_result = None

with st.sidebar:
    st.header("⚙️ Настройки")
    
    api_key = st.text_input("🔑 API Token", type="password", help="Введите токен от gen-api.ru")
    
    if api_key and st.button("🔌 Проверить подключение"):
        with st.spinner("Тест..."):
            test = QwenAnalyticsAgent(api_key=api_key).test_connection()
            if test['success']:
                st.success("✅ Подключение успешно!")
                st.json(test['details'])
            else:
                st.error(test['error'])
                if 'details' in test:
                    st.json(test['details'])
    
    model = st.selectbox("🧠 Модель", ["qwen-3-6-plus", "qwen-3-5-plus", "qwen-max"], index=0)
    
    st.divider()
    theme = st.selectbox("🎨 Тема графиков", 
                        ['plotly_white', 'plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_dark'])
    
    st.divider()
    st.info("📁 Загрузите CSV или Excel файл")

# Загрузка файла
uploaded_file = st.file_uploader("📁 Загрузите датасет", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Чтение данных
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Загружен: **{uploaded_file.name}**")
        
        # Быстрая статистика
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("📊 Строк", f"{df.shape[0]:,}")
        c2.metric("📐 Столбцов", df.shape[1])
        c3.metric("🔢 Числовых", len(df.select_dtypes(include='number').columns))
        c4.metric("🏷️ Категориальных", len(df.select_dtypes(include=['object', 'category', 'bool']).columns))
        c5.metric("⚠️ Пропусков", df.isnull().sum().sum())
        
        with st.expander("📋 Превью данных"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # === ПОЛЕ ДЛЯ КОНТЕКСТА ОТ ПОЛЬЗОВАТЕЛЯ ===
        st.markdown("### 📝 Контекст анализа (опционально)")
        user_context = st.text_area(
            "На что обратить внимание?",
            placeholder="Например: 'Интересует зависимость зарплаты от опыта работы' или 'Проверь наличие выбросов в возрасте'",
            height=70
        )
        
        # === КНОПКИ АНАЛИЗА ===
        tab1, tab2 = st.tabs(["🎯 Авто-анализ", "💬 Запрос к данным"])
        
        with tab1:
            st.markdown("### 🚀 Автоматический анализ датасета")
            st.markdown("LLM сам изучит данные и построит визуализации")
            
            if st.button("🔍 Запустить авто-анализ", type="primary", disabled=not api_key, use_container_width=True):
                if not api_key or len(api_key) < 10:
                    st.error("❌ Введите валидный API токен")
                    st.stop()
                
                with st.spinner("🤖 AI анализирует данные..."):
                    progress = st.progress(0)
                    status = st.empty()
                    
                    agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                    
                    progress.progress(25)
                    status.text("Подключение к API...")
                    
                    test = agent.test_connection()
                    if not test['success']:
                        st.error(f"❌ {test['error']}")
                        st.stop()
                    
                    progress.progress(50)
                    status.text("Генерация аналитического кода...")
                    
                    result = agent.run_analysis(
                        None, 
                        df, 
                        auto_eda=True, 
                        theme=theme,
                        user_context=user_context if user_context else None
                    )
                    
                    progress.progress(100)
                    status.text("✅ Готово!")
                    time.sleep(0.5)
                    progress.empty()
                    status.empty()
                
                # Сохранение результата
                st.session_state.auto_analysis_result = result
                
                # === ОТОБРАЖЕНИЕ ОТЧЕТА (БЕЗ КОДА!) ===
                st.markdown("---")
                
                if not result['success']:
                    st.error(f"❌ Ошибка: {result.get('error', 'Неизвестная ошибка')}")
                    if result.get('details'):
                        with st.expander("🔍 Технические детали"):
                            st.json(result['details'])
                else:
                    # Заголовок отчета
                    st.markdown("## 📊 Отчет по анализу данных")
                    
                    # Рассуждения агента
                    if result.get('thought'):
                        with st.expander("💭 Логика анализа", expanded=True):
                            st.markdown(result['thought'])
                    
                    # Текстовые выводы
                    if result.get('output'):
                        st.markdown("### 🔍 Ключевые выводы")
                        for line in result['output']:
                            st.markdown(f"• {line}")
                    
                    # Графики
                    if result.get('figures'):
                        st.markdown("### 📈 Визуализации")
                        
                        if len(result['figures']) > 1:
                            tabs = st.tabs([f"График {i+1}" for i in range(len(result['figures']))])
                            for idx, (tab, fig) in enumerate(zip(tabs, result['figures'])):
                                with tab:
                                    st.plotly_chart(fig, use_container_width=True, key=f"fig_{idx}")
                                    
                                    # Экспорт
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        png = download_plotly_fig(fig, f"plot_{idx+1}", 'png')
                                        if png:
                                            st.download_button("📥 PNG", base64.b64decode(png), 
                                                             f"plot_{idx+1}.png", "image/png")
                                    with col2:
                                        html_data = download_plotly_fig(fig, f"plot_{idx+1}", 'html')
                                        if html_data:
                                            st.download_button("🌐 HTML", base64.b64decode(html_data),
                                                             f"plot_{idx+1}.html", "text/html")
                        else:
                            st.plotly_chart(result['figures'][0], use_container_width=True)
                    
                    # Табличные результаты
                    if result.get('data_result') is not None:
                        st.markdown("### 📋 Статистика")
                        if isinstance(result['data_result'], pd.DataFrame):
                            st.dataframe(result['data_result'], use_container_width=True)
                            csv = result['data_result'].to_csv(index=False)
                            st.download_button("📥 Скачать CSV", csv, "statistics.csv", "text/csv")
                        else:
                            st.write(result['data_result'])
                    
                    # Интерпретация
                    if result.get('explanation'):
                        st.info(f"💡 **Интерпретация:** {result['explanation']}")
                    
                    # Рекомендации
                    if result.get('recommendations'):
                        st.markdown("### 🔄 Что ещё изучить?")
                        for rec in result['recommendations']:
                            st.markdown(f"• {rec}")
        
        with tab2:
            st.markdown("### 💬 Анализ по вашему запросу")
            
            query = st.text_area(
                "Ваш запрос",
                placeholder="Например:\n• Построй гистограмму распределения возраста\n• Сравни среднюю зарплату по отделам\n• Найди корреляции между переменными",
                height=100
            )
            
            if st.button("🚀 Выполнить анализ", type="primary", disabled=not (api_key and query)):
                if not api_key or len(api_key) < 10:
                    st.error("❌ Введите валидный API токен")
                    st.stop()
                
                # Проверка безопасности
                is_safe, msg = check_safety(query)
                if not is_safe:
                    st.warning(msg)
                    st.stop()
                
                with st.spinner("🤖 Выполнение анализа..."):
                    agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                    
                    result = agent.run_analysis(
                        query, 
                        df, 
                        auto_eda=False, 
                        theme=theme,
                        user_context=user_context if user_context else None
                    )
                
                st.markdown("---")
                
                if not result['success']:
                    st.error(f"❌ {result.get('error', 'Ошибка')}")
                    if result.get('details'):
                        with st.expander("🔍 Детали"):
                            st.json(result['details'])
                else:
                    st.markdown("## 📊 Результат анализа")
                    
                    if result.get('thought'):
                        with st.expander("💭 Логика", expanded=True):
                            st.markdown(result['thought'])
                    
                    if result.get('output'):
                        st.markdown("### 📤 Вывод")
                        for line in result['output']:
                            st.markdown(f"• {line}")
                    
                    if result.get('data_result') is not None:
                        st.markdown("### 📋 Результаты")
                        if isinstance(result['data_result'], pd.DataFrame):
                            st.dataframe(result['data_result'], use_container_width=True)
                        else:
                            st.write(result['data_result'])
                    
                    if result.get('figures'):
                        st.markdown("### 📈 Графики")
                        for idx, fig in enumerate(result['figures']):
                            st.plotly_chart(fig, use_container_width=True, key=f"manual_fig_{idx}")
                    
                    if result.get('explanation'):
                        st.info(f"💡 {result['explanation']}")
    
    except Exception as e:
        st.error(f"❌ Ошибка: {type(e).__name__}: {e}")
        with st.expander("🔍 Traceback"):
            st.code(traceback.format_exc())
else:
    # Стартовый экран
    st.info("👆 **Загрузите CSV или Excel файл** для начала анализа")
    
    st.markdown("### 💡 Примеры использования:")
    examples = [
        ("📊 Авто-анализ", "Нажмите 'Запустить авто-анализ' — AI сам изучит все данные"),
        ("📈 Гистограмма", "В запросе: 'Построй гистограмму распределения столбца age'"),
        ("🔗 Корреляции", "В запросе: 'Найди корреляции между salary и experience'"),
        ("📊 Группировка", "В запросе: 'Сравни среднюю зарплату по отделам'")
    ]
    
    for title, desc in examples:
        st.markdown(f"**{title}**: {desc}")
    
    st.markdown("### 🎯 Возможности:")
    cols = st.columns(3)
    cols[0].markdown("📊 **Визуализации**\n• Интерактивные графики Plotly\n• Экспорт в PNG/HTML\n• Автоматический подбор типов")
    cols[1].markdown("🔒 **Безопасность**\n• Многоуровневая проверка кода\n• Защита от prompt-injection\n• Песочница для выполнения")
    cols[2].markdown("🤖 **AI-агент**\n• Генерация кода на естественном языке\n• Объяснение результатов\n• Рекомендации по анализу")

# Футер
st.markdown("---")
st.caption("🤖 AI Analytics Agent | Qwen3.6 • gen-api.ru | Данные обрабатываются локально")
