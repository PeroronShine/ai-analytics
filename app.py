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

# OpenAI-compatible client для Qwen
from openai import OpenAI

# Отключаем предупреждения SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ["CURL_CA_BUNDLE"] = ""

# Настройки Plotly для Streamlit
px.defaults.template = "plotly_white"
go.Figure.update_layout = lambda self, **kwargs: go.Figure.update_layout(self, **kwargs)

# Запрещённые модули
FORBIDDEN_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'httpx',
    'urllib', 'ftplib', 'smtplib', 'paramiko', 'pickle', 'marshal', 'eval',
    'exec', 'compile', 'open', 'input', 'breakpoint', 'import'
}
FORBIDDEN_BUILTINS = {'eval', 'exec', 'compile', 'import', 'open', 'input'}


class SafeCodeExecutor:
    """Безопасный исполнитель кода с улучшенной поддержкой визуализаций"""
    
    PLOTLY_TEMPLATES = ['plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_white', 'plotly_dark']
    
    def __init__(self, df: pd.DataFrame, timeout: int = 30, theme: str = 'plotly_white'):
        self.df = df.copy()
        self.timeout = timeout
        self.output = []
        self.figures = []
        self.theme = theme
        px.defaults.template = theme

    def _safe_globals(self):
        """Безопасное окружение с расширенными возможностями визуализации"""
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
            # Попытка создать фигуру из данных
            try:
                return go.Figure(data=fig)
            except:
                return None

    def _save_fig_callback(self, fig, filename: str = "plot", title: str = None, **layout_kwargs):
        """Сохранение графика с настройками оформления"""
        normalized = self._normalize_figure(fig)
        if normalized:
            if title:
                normalized.update_layout(title={'text': title, 'x': 0.5, 'xanchor': 'center'})
            if layout_kwargs:
                normalized.update_layout(**layout_kwargs)
            self.figures.append({'fig': normalized, 'name': filename, 'title': title})
            return f"✅ График '{filename}' создан"
        return "❌ Ошибка создания графика"

    def _display_fig_callback(self, fig, title: str = None):
        """Алиас для save_fig с акцентом на отображение"""
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
        """Проверка кода с улучшенной обработкой"""
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
        """Выполнение кода с улучшенной обработкой результатов"""
        result = {'success': False, 'output': [], 'error': None, 'figures': [], 'data_result': None}

        is_valid, message = self._validate_code(code)
        if not is_valid:
            result['error'] = f"Код отклонён: {message}"
            return result

        try:
            safe_globals = self._safe_globals()
            local_vars = {}

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                exec(code, safe_globals, local_vars)

            result['output'] = self.output
            # Извлекаем только фигуры, а не словари
            result['figures'] = [item['fig'] if isinstance(item, dict) and 'fig' in item else item 
                                for item in self.figures if isinstance(item, (go.Figure, dict))]
            
            if 'result' in local_vars:
                result['data_result'] = local_vars['result']
            elif 'df_result' in local_vars:
                result['data_result'] = local_vars['df_result']
            result['success'] = True

        except Exception as e:
            result['error'] = f"Ошибка: {type(e).__name__}: {str(e)}"
            result['traceback'] = traceback.format_exc()

        return result


class QwenAnalyticsAgent:
    """Агент на Qwen3.6-Plus с улучшенными промптами для визуализации"""
    
    def __init__(self, api_key: str, model: str = "qwen3.6-plus"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.df_info = None

    def prepare_dataset_context(self, df: pd.DataFrame):
        """Подготовка контекста данных с примерами визуализаций"""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'sample': df.head(2).to_dict(orient='records'),
            'missing': df.isnull().sum().to_dict(),
            'suggested_charts': self._suggest_charts(df, numeric_cols, categorical_cols)
        }

    def _suggest_charts(self, df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> List[dict]:
        """Генерация рекомендаций по визуализации"""
        suggestions = []
        if numeric and categorical:
            suggestions.append({'type': 'bar', 'x': categorical[0], 'y': numeric[0], 'desc': f'Столбчатая диаграмма: {numeric[0]} по {categorical[0]}'})
            suggestions.append({'type': 'box', 'x': categorical[0], 'y': numeric[0], 'desc': f'Box plot: распределение {numeric[0]}'})
        if len(numeric) >= 2:
            suggestions.append({'type': 'scatter', 'x': numeric[0], 'y': numeric[1], 'desc': f'Scatter plot: {numeric[0]} vs {numeric[1]}'})
        if categorical:
            suggestions.append({'type': 'pie' if len(df[categorical[0]].unique()) <= 10 else 'bar', 
                               'names': categorical[0], 'desc': f'Распределение: {categorical[0]}'})
        if numeric:
            suggestions.append({'type': 'histogram', 'x': numeric[0], 'desc': f'Гистограмма: {numeric[0]}'})
        return suggestions[:5]

    def generate_code(self, user_query: str, auto_eda: bool = False) -> str:
        """Генерация кода с акцентом на визуализацию"""
        system_prompt = """Ты эксперт по визуализации данных. Отвечай ТОЛЬКО JSON:
{
"thought": "краткое рассуждение о подходе",
"code": "Python код. УЖЕ импортированы: pd, np, px, go, make_subplots. НЕ пиши import! Используй: df, save_fig(fig, 'name', title='...'), display_fig(), print()",
"explanation": "что покажут графики и как интерпретировать"
}

ПРАВИЛА для графиков:
1. Всегда вызывай save_fig(fig, 'unique_name', title='Заголовок') для каждого графика
2. Используй px для быстрых графиков, go для кастомных
3. Добавляй подписи осей: labels={'x': '...', 'y': '...'}
4. Для нескольких графиков используй create_dashboard([fig1, fig2], titles=['A', 'B'])
5. result = df.groupby(...).agg(...) для табличных результатов"""

        context = "AUTO_EDA MODE: " if auto_eda else ""
        context += f"Данные: {json.dumps(self.df_info, ensure_ascii=False)}\nЗадача: {user_query}"
        
        if auto_eda:
            context += "\n\nСоздай комплексную визуализацию: 1) распределения числовых переменных, 2) корреляционную матрицу если >1 числового столбца, 3) бар-чарты для категориальных, 4) scatter plot для пар числовых"

        try:
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
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e), "code": "print('Ошибка API')", "explanation": "Попробуйте повторить запрос"})

    def run_analysis(self, user_query: str, df: pd.DataFrame, auto_eda: bool = False, theme: str = 'plotly_white') -> dict:
        """Запуск анализа с поддержкой тем оформления"""
        self.prepare_dataset_context(df)
        plan_response = self.generate_code(user_query, auto_eda)

        try:
            plan = json.loads(plan_response)
        except:
            match = re.search(r'\{[\s\S]*\}', plan_response)
            plan = json.loads(match.group()) if match else {"error": "Parse error", "code": "print('Error')"}

        code = plan.get('code', '')
        executor = SafeCodeExecutor(df, theme=theme)
        exec_result = executor.execute(code)

        return {
            'success': exec_result['success'],
            'thought': plan.get('thought', ''),
            'explanation': plan.get('explanation', ''),
            'code': code,
            'output': exec_result['output'],
            'figures': exec_result['figures'],
            'data_result': exec_result['data_result'],
            'error': exec_result.get('error'),
            'traceback': exec_result.get('traceback')
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
st.set_page_config(page_title="AI Analytics Pro", page_icon="📊", layout="wide")

# Кастомные стили для визуализаций
st.markdown("""
<style>
    .stPlotlyChart {border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {padding: 8px 16px; border-radius: 5px 5px 0 0;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Analytics Agent Pro")
st.markdown("*Интеллектуальный анализ данных с автоматической визуализацией на Qwen3.6-Plus*")

with st.sidebar:
    st.header("⚙️ Настройки")
    
    api_key = st.text_input("🔑 DashScope API Key", type="password", help="Получите ключ на dashscope.aliyun.com")
    if api_key and len(api_key) > 10:
        st.success("✅ Ключ сохранён")
    
    model = st.selectbox("🧠 Модель", ["qwen3.6-plus", "qwen3.5-plus", "qwen-max"], index=0)
    
    st.divider()
    
    # Настройки визуализации
    st.subheader("🎨 Визуализация")
    theme = st.selectbox("Тема графиков", 
                        ['plotly_white', 'plotly', 'ggplot2', 'seaborn', 'simple_white', 'plotly_dark'],
                        index=0, help="Влияет на стиль всех создаваемых графиков")
    
    auto_height = st.checkbox("📐 Авто-высота графиков", value=True)
    default_height = st.slider("Высота графика (px)", 300, 1000, 500) if not auto_height else 500
    
    st.divider()
    st.info("📁 Поддерживаемые форматы: CSV, Excel (.xlsx)")

# Загрузка данных
uploaded_file = st.file_uploader("📁 Загрузите файл с данными", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # Чтение файла
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
        
        # Превью данных
        with st.expander("📋 Превью данных", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.caption("Типы данных:")
            st.code(df.dtypes.astype(str).to_dict(), language='python')
        
        # Режим анализа
        tab1, tab2 = st.tabs(["🔍 Ручной запрос", "⚡ Авто-анализ (EDA)"])
        
        with tab1:
            query = st.text_area("💬 Ваш запрос к данным", 
                               placeholder="Примеры:\n• Построй гистограмму распределения столбца 'age'\n• Сравни средние значения 'salary' по группам 'department'\n• Найди корреляции между числовыми переменными", 
                               height=100)
        
        with tab2:
            st.markdown("### 🚀 Быстрый исследовательский анализ")
            st.markdown("Автоматически создаст набор визуализаций:")
            eda_features = st.multiselect(
                "Выберите типы графиков:",
                ["📊 Гистограммы числовых переменных", 
                 "📈 Корреляционная матрица", 
                 "🥧 Распределение категориальных", 
                 "🔗 Scatter plot пар переменных",
                 "📦 Box plot для выбросов"],
                default=["📊 Гистограммы числовых переменных", "📈 Корреляционная матрица"]
            )
            run_eda = st.button("🎯 Запустить авто-анализ", type="primary", use_container_width=True)
        
        # Кнопка выполнения
        run_analysis = st.button("🚀 Выполнить анализ", type="primary", use_container_width=True, 
                                disabled=not (api_key and (query or run_eda)))
        
        if run_analysis or run_eda:
            current_query = "Создай комплексную визуализацию: " + ", ".join(eda_features) if run_eda else query
            is_safe, msg = check_safety(current_query)
            
            if not is_safe:
                st.warning(msg)
                st.stop()
            
            with st.spinner("🤖 Генерация кода и выполнение анализа..."):
                agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                result = agent.run_analysis(current_query, df, auto_eda=run_eda, theme=theme)
                
                st.markdown("---")
                
                # Обработка ошибок
                if result.get('error'):
                    st.error(f"❌ {result['error']}")
                    if result.get('traceback'):
                        with st.expander("🔍 Детали ошибки"):
                            st.code(result['traceback'], language='python')
                else:
                    # Мысли агента
                    if result.get('thought'):
                        with st.expander("💭 Ход рассуждений агента", expanded=True):
                            st.markdown(result['thought'])
                    
                    # Сгенерированный код
                    with st.expander("📝 Сгенерированный код", expanded=st.session_state.get('show_code', False)):
                        st.code(result['code'], language='python')
                        st.caption("💡 Совет: Вы можете скопировать этот код и модифицировать его под свои нужды")
                    
                    # Текстовый вывод
                    if result.get('output'):
                        with st.expander("📤 Консольный вывод", expanded=True):
                            for line in result['output']:
                                st.text(line)
                    
                    # Табличные результаты
                    if result.get('data_result') is not None:
                        st.markdown("### 📋 Результаты вычислений")
                        if isinstance(result['data_result'], pd.DataFrame):
                            st.dataframe(result['data_result'], use_container_width=True)
                            # Кнопка скачивания
                            csv = result['data_result'].to_csv(index=False, encoding='utf-8-sig')
                            st.download_button("📥 Скачать CSV", data=csv, 
                                             file_name="result.csv", mime="text/csv")
                        else:
                            st.json(result['data_result'] if isinstance(result['data_result'], (dict, list)) else str(result['data_result']))
                    
                    # 🎨 ВИЗУАЛИЗАЦИИ - улучшенный блок
                    if result.get('figures'):
                        st.markdown("### 📈 Сгенерированные визуализации")
                        
                        # Вкладки для нескольких графиков
                        if len(result['figures']) > 1:
                            tabs = st.tabs([f"График {i+1}" for i in range(len(result['figures']))])
                            for idx, (tab, fig) in enumerate(zip(tabs, result['figures'])):
                                with tab:
                                    st.plotly_chart(fig, use_container_width=True, key=f"fig_{idx}")
                                    
                                    # Кнопки экспорта
                                    col_exp1, col_exp2 = st.columns(2)
                                    with col_exp1:
                                        png_data = download_plotly_fig(fig, f"plot_{idx+1}", 'png')
                                        if png_data:
                                            st.download_button("📥 PNG", data=base64.b64decode(png_data),
                                                             file_name=f"plot_{idx+1}.png", mime="image/png")
                                    with col_exp2:
                                        html_data = download_plotly_fig(fig, f"plot_{idx+1}", 'html')
                                        if html_data:
                                            st.download_button("🌐 HTML", data=base64.b64decode(html_data),
                                                             file_name=f"plot_{idx+1}.html", mime="text/html")
                        else:
                            fig = result['figures'][0]
                            st.plotly_chart(fig, use_container_width=True)
                            # Экспорт для одиночного графика
                            col_e1, col_e2, col_e3 = st.columns(3)
                            with col_e1:
                                png_data = download_plotly_fig(fig, "result", 'png')
                                if png_data:
                                    st.download_button("📥 PNG", data=base64.b64decode(png_data),
                                                     file_name="visualization.png", mime="image/png")
                            with col_e2:
                                html_data = download_plotly_fig(fig, "result", 'html')
                                if html_data:
                                    st.download_button("🌐 HTML", data=base64.b64decode(html_data),
                                                     file_name="visualization.html", mime="text/html")
                            with col_e3:
                                json_data = fig.to_plotly_json()
                                st.download_button("🔧 JSON", data=json.dumps(json_data),
                                                 file_name="visualization.json", mime="application/json")
                    
                    # Объяснение результатов
                    if result.get('explanation'):
                        st.info(f"💡 **Интерпретация:** {result['explanation']}")
                    
                    # Рекомендации для следующих шагов
                    if not run_eda and result.get('success'):
                        with st.expander("🔄 Что можно сделать дальше?"):
                            suggestions = [
                                "🔍 Детализировать конкретный аспект анализа",
                                "📊 Добавить новые типы визуализаций",
                                "🧹 Очистить данные от выбросов или пропусков",
                                "📈 Построить прогнозную модель",
                                "💾 Экспортировать результаты в отчёт"
                            ]
                            for s in suggestions:
                                st.markdown(f"• {s}")
    
    except Exception as e:
        st.error(f"❌ Ошибка обработки: {type(e).__name__}: {e}")
        with st.expander("🔍 Stack trace"):
            st.code(traceback.format_exc(), language='python')
else:
    # Стартовый экран
    st.info("👆 **Загрузите CSV или Excel файл** для начала анализа")
    
    st.markdown("### 💡 Примеры запросов:")
    examples = [
        "Построй гистограмму распределения возраста",
        "Сравни среднюю зарплату по отделам",
        "Найди корреляции между числовыми переменными",
        "Визуализируй динамику продаж по месяцам",
        "Покажи box plot для выявления выбросов"
    ]
    for ex in examples:
        st.markdown(f"• `{ex}`")
    
    st.markdown("### 🎯 Возможности:")
    cols = st.columns(3)
    cols[0].markdown("📊 **Визуализации**\n• Plotly Express & Graph Objects\n• Интерактивные дашборды\n• Экспорт в PNG/HTML")
    cols[1].markdown("🔒 **Безопасность**\n• Песочница для кода\n• Фильтрация опасных операций\n• Валидация запросов")
    cols[2].markdown("🤖 **AI-помощник**\n• Генерация кода на естественном языке\n• Объяснение результатов\n• Рекомендации по анализу")

# Футер
st.markdown("---")
st.caption("🤖 AI Analytics Agent Pro | Powered by Qwen3.6-Plus & Plotly | Данные обрабатываются локально в сессии")
