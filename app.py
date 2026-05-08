# app.py - AI Analytics Agent (LLM анализирует → Streamlit визуализирует)
import ssl
import os
import re
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import traceback
from datetime import datetime
from typing import Optional, List

# Отключаем SSL предупреждения
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
os.environ["CURL_CA_BUNDLE"] = ""

px.defaults.template = "plotly_white"

# Защита от prompt-injection (проверка входящих запросов пользователя)
FORBIDDEN_PATTERNS = [
    r'eval\s*\(', r'exec\s*\(', r'compile\s*\(',
    r'import\s+os', r'import\s+sys', r'import\s+subprocess',
    r'os\.system', r'subprocess\.', r'shutil\.',
    r'__import__', r'__builtins__', r'__class__',
    r'open\s*\(', r'read\s*\(', r'write\s*\(',
    r'input\s*\(', r'breakpoint\s*\(',
    r'pickle', r'marshal', r'shelve'
]
FORBIDDEN_WORDS = ['del ', 'raise ', 'pass ', 'yield ']


class QwenAnalyticsAgent:
    """Агент для gen-api.ru — LLM анализирует данные в своём интерпретаторе"""
    
    def __init__(self, api_key: str, model: str = "qwen-3-6-plus"):
        self.api_key = api_key
        self.model = model
        self.df_info = None
        self.base_url = "https://api.gen-api.ru/api/v1/networks/qwen-3-6-plus"
        
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
                return {'success': True, 'error': None, 'details': {'status': 'OK', 'model': self.model}}
            elif response.status_code == 401:
                return {'success': False, 'error': "❌ Неверный токен", 'details': {'status_code': 401}}
            else:
                return {'success': False, 'error': f"❌ Ошибка {response.status_code}", 'details': response.text}
        except Exception as e:
            return {'success': False, 'error': f"❌ {type(e).__name__}", 'details': str(e)}

    def prepare_dataset_context(self, df: pd.DataFrame):
        """Подготовка метаданных датасета для отправки в LLM"""
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'sample': df.head(3).to_dict(orient='records'),
            'missing': {k: int(v) for k, v in df.isnull().sum().items() if v > 0},
            'numeric_stats': df[numeric_cols].describe().to_dict() if numeric_cols else {},
            'categorical_stats': {col: df[col].value_counts().head(5).to_dict() for col in categorical_cols[:3]}
        }

    def generate_analysis(self, user_query: str, auto_mode: bool) -> dict:
        """
        LLM запускает код в СВОЁМ интерпретаторе и возвращает ТОЛЬКО текстовый отчёт.
        Мы не выполняем код локально — только принимаем результат.
        """
        
        system_prompt = """Ты эксперт по анализу данных. У тебя есть доступ к датасету и Python-интерпретатору.
Проведи анализ и верни ТОЛЬКО валидный JSON:
{
"thought": "краткое рассуждение о подходе (2-3 предложения)",
"insights": ["ключевой вывод 1", "ключевой вывод 2", "ключевой вывод 3"],
"metrics_text": "Статистика:\\n• Среднее X: ...\\n• Медиана Y: ...\\n• Корреляция: ...",
"explanation": "бизнес-интерпретация результатов (2-4 предложения)"
}

ПРАВИЛА:
1. Ты выполняешь код в своём интерпретаторе (describe, groupby, corr, value_counts и т.д.)
2. В ответе — ТОЛЬКО текст и цифры. Никакого кода, графиков, plotly, matplotlib.
3. metrics_text — человекочитаемый текст с метриками, не JSON.
4. insights — список коротких утверждений фактов."""

        if auto_mode:
            context = f"АВТО-АНАЛИЗ ДАТАСЕТА.\nМетаданные: {json.dumps(self.df_info, ensure_ascii=False)}\n\nЗадача: Проведи полный статистический анализ. Найди: 1) распределения, 2) выбросы, 3) корреляции, 4) групповые сравнения. Верни текстовый отчёт."
        else:
            context = f"Метаданные датасета: {json.dumps(self.df_info, ensure_ascii=False)}\nЗапрос пользователя: {user_query}\n\nПроведи анализ согласно запросу и верни текстовый отчёт."

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
            "temperature": 0.2 if auto_mode else 0.3,
            "top_p": 0.9,
            "response_format": {"type": "json_object"}
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=input_data,
                timeout=120 if auto_mode else 90
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Обработка ответа gen-api.ru (поле "response" — массив строк)
                content = None
                if "response" in result and isinstance(result["response"], list) and result["response"]:
                    response_str = result["response"][0]
                    try:
                        content = json.dumps(json.loads(response_str), ensure_ascii=False)
                    except:
                        content = response_str
                elif "output" in result:
                    content = result["output"]
                elif "choices" in result and result["choices"]:
                    content = result["choices"][0].get("message", {}).get("content", "")
                
                if content is None:
                    content = json.dumps(result, ensure_ascii=False)
                
                return {'success': True, 'content': content}
            else:
                return {
                    'success': False,
                    'error_type': f'HTTP_{response.status_code}',
                    'message': f'Ошибка API: {response.status_code}',
                    'details': response.text[:500] if isinstance(response.text, str) else str(response.text)
                }
                
        except requests.exceptions.Timeout:
            return {'success': False, 'error_type': 'Timeout', 'message': 'Превышено время ожидания', 'details': 'Попробуйте упростить запрос'}
        except requests.exceptions.ConnectionError:
            return {'success': False, 'error_type': 'ConnectionError', 'message': 'Нет подключения к API', 'details': 'Проверьте интернет'}
        except Exception as e:
            return {'success': False, 'error_type': type(e).__name__, 'message': str(e), 'details': traceback.format_exc()[:500]}

    def run_analysis(self, user_query: str, df: pd.DataFrame, auto_mode: bool) -> dict:
        """Запуск анализа: LLM работает на своей стороне, мы получаем только текст"""
        self.prepare_dataset_context(df)
        gen_result = self.generate_analysis(user_query, auto_mode)
        
        if not gen_result['success']:
            return {'success': False, 'stage': 'api', 'error': gen_result.get('message'), 'details': gen_result.get('details')}
        
        # Парсинг ответа LLM
        try:
            report = json.loads(gen_result['content'])
        except:
            match = re.search(r'\{[\s\S]*\}', gen_result['content'])
            report = json.loads(match.group()) if match else {
                "error": "Не удалось распарсить ответ LLM",
                "insights": ["Проверьте запрос и повторите"],
                "explanation": gen_result['content'][:200]
            }

        return {
            'success': True,
            'thought': report.get('thought', ''),
            'insights': report.get('insights', []),
            'metrics_text': report.get('metrics_text', ''),
            'explanation': report.get('explanation', ''),
            'raw_response': gen_result['content']  # для отладки
        }


def check_safety(query: str) -> tuple:
    """Проверка пользовательского запроса на безопасность"""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "⚠️ Обнаружен подозрительный паттерн"
    for word in FORBIDDEN_WORDS:
        if f" {word}" in query or query.startswith(word):
            return False, "⚠️ Обнаружено запрещенное слово"
    if '<script' in query.lower() or 'javascript:' in query.lower():
        return False, "⚠️ XSS попытка"
    return True, "✅ Безопасно"


# ================= UI =================
st.set_page_config(page_title="AI Analytics Agent", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stPlotlyChart {border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .report-box {background: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #0068c9; margin: 15px 0;}
    .insight-tag {display: inline-block; background: #e3f2fd; padding: 5px 12px; border-radius: 20px; margin: 4px 4px 4px 0; font-size: 0.9em;}
    .metrics-block {background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; font-family: monospace; white-space: pre-wrap;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Analytics Agent")
st.markdown("*LLM анализирует данные в своём интерпретаторе → Streamlit строит графики*")

# Session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:
    st.header("⚙️ Настройки")
    api_key = st.text_input("🔑 API Token", type="password", value=st.session_state.api_key, help="Токен от gen-api.ru")
    st.session_state.api_key = api_key
    
    if api_key and st.button("🔌 Проверить подключение"):
        with st.spinner("Тест..."):
            test = QwenAnalyticsAgent(api_key=api_key).test_connection()
            if test['success']:
                st.success("✅ Подключение успешно!")
                st.json(test['details'])
            else:
                st.error(test['error'])
                if 'details' in test:
                    with st.expander("🔍 Детали"):
                        st.json(test['details'])
    
    model = st.selectbox("🧠 Модель", ["qwen-3-6-plus", "qwen-3-5-plus", "qwen-max"], index=0)
    st.info("📁 Поддерживаются: CSV, Excel (.xlsx)")

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
        
        st.markdown("---")
        
        # === БЛОК ПРОМПТА ПОЛЬЗОВАТЕЛЯ ===
        st.subheader("💬 Запрос к аналитику")
        user_query = st.text_area(
            "Опишите задачу анализа или оставьте пустым для автоматического анализа",
            placeholder="Примеры:\n• 'Сравни среднюю зарплату по отделам'\n• 'Найди корреляции между переменными'\n• 'Есть ли выбросы в возрасте?'",
            height=90
        )
        
        # Кнопка запуска анализа
        run_btn = st.button("🚀 Запустить анализ", type="primary", disabled=not api_key, use_container_width=True)
        
        if run_btn:
            if not api_key or len(api_key) < 10:
                st.error("❌ Введите валидный API токен")
                st.stop()
            
            # Авто-режим если поле пустое
            is_auto = not user_query.strip()
            
            # Проверка безопасности запроса
            is_safe, msg = check_safety(user_query if not is_auto else "auto")
            if not is_safe:
                st.warning(msg)
                st.stop()
            
            with st.spinner("🤖 LLM анализирует данные в своём интерпретаторе..."):
                agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                result = agent.run_analysis(user_query, df, auto_mode=is_auto)
            
            st.markdown("---")
            
            # Обработка ошибок API
            if not result['success']:
                st.error(f"❌ {result.get('error', 'Ошибка анализа')}")
                if result.get('details'):
                    with st.expander("🔍 Технические детали"):
                        st.json(result['details'])
            else:
                # === 📝 ОТЧЁТ ОТ LLM (ТОЛЬКО ТЕКСТ) ===
                st.subheader("📋 Аналитический отчёт")
                
                # Логика анализа
                if result.get('thought'):
                    with st.expander("💭 Логика анализа", expanded=True):
                        st.markdown(result['thought'])
                
                # Ключевые выводы (теги)
                if result.get('insights'):
                    st.markdown("**🔍 Ключевые выводы:**")
                    for insight in result['insights']:
                        st.markdown(f"<span class='insight-tag'>✅ {insight}</span>", unsafe_allow_html=True)
                    st.markdown("")  # отступ
                
                # Статистические метрики (текст)
                if result.get('metrics_text'):
                    st.markdown("**📊 Статистические метрики:**")
                    st.markdown(f"<div class='metrics-block'>{result['metrics_text']}</div>", unsafe_allow_html=True)
                
                # Бизнес-интерпретация
                if result.get('explanation'):
                    st.info(f"💡 **Интерпретация:** {result['explanation']}")
                
                # === 📈 АВТОМАТИЧЕСКАЯ ВИЗУАЛИЗАЦИЯ (БЕЗ LLM, ТОЛЬКО STREAMLIT) ===
                st.markdown("---")
                st.subheader("📈 Автоматическая визуализация данных")
                st.caption("Графики построены автоматически через Plotly — независимо от LLM")
                
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
                
                # Визуализации для числовых переменных
                if numeric_cols:
                    st.markdown("#### 🔢 Числовые переменные")
                    
                    if len(numeric_cols) >= 2:
                        col1, col2 = st.columns(2)
                        with col1:
                            # Гистограмма с выбором колонки
                            chart_col = st.selectbox("📊 Колонка для гистограммы", numeric_cols, key="hist_select")
                            fig_hist = px.histogram(
                                df, x=chart_col, 
                                title=f"Распределение: {chart_col}",
                                labels={'x': chart_col, 'y': 'Количество'},
                                nbins=30
                            )
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        with col2:
                            # Корреляционная матрица
                            corr_cols = numeric_cols[:10]  # ограничиваем для читаемости
                            corr_matrix = df[corr_cols].corr(numeric_only=True)
                            fig_corr = px.imshow(
                                corr_matrix, 
                                text_auto=True, 
                                title="Корреляционная матрица",
                                color_continuous_scale='RdBu_r',
                                aspect='auto'
                            )
                            st.plotly_chart(fig_corr, use_container_width=True)
                    elif len(numeric_cols) == 1:
                        fig_hist = px.histogram(df, x=numeric_cols[0], title=f"Распределение: {numeric_cols[0]}")
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Box plot для сравнения по категориям
                    if cat_cols and len(numeric_cols) >= 1:
                        st.markdown("#### 📦 Box plot: сравнение по категориям")
                        box_cat = st.selectbox("Категория для группировки", cat_cols, key="box_cat_select")
                        box_num = st.selectbox("Числовая переменная", numeric_cols, key="box_num_select")
                        
                        if df[box_cat].nunique() <= 15:  # чтобы не перегружать
                            fig_box = px.box(
                                df, x=box_cat, y=box_num,
                                title=f"Распределение '{box_num}' по '{box_cat}'",
                                points='outliers'
                            )
                            st.plotly_chart(fig_box, use_container_width=True)
                
                # Визуализации для категориальных переменных
                if cat_cols:
                    st.markdown("#### 🏷️ Категориальные переменные")
                    for col in cat_cols[:3]:  # максимум 3, чтобы не перегрузить
                        if df[col].nunique() <= 20 and df[col].nunique() >= 2:
                            counts = df[col].value_counts()
                            fig_bar = px.bar(
                                x=counts.index, y=counts.values,
                                title=f"Распределение: {col}",
                                labels={'x': col, 'y': 'Количество'},
                                color=counts.index,
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            fig_bar.update_layout(showlegend=False)
                            st.plotly_chart(fig_bar, use_container_width=True)
                
                # Если нет подходящих данных
                if not numeric_cols and not cat_cols:
                    st.info("ℹ️ В датасете нет столбцов, подходящих для автоматической визуализации")
                
                # Кнопки экспорта (опционально)
                if numeric_cols or cat_cols:
                    st.markdown("*💡 Совет: нажмите на график для интерактивного изучения. Используйте лупу для приближения.*")
    
    except Exception as e:
        st.error(f"❌ Ошибка обработки: {type(e).__name__}: {e}")
        with st.expander("🔍 Traceback для разработчика"):
            st.code(traceback.format_exc(), language='python')
else:
    # Стартовый экран
    st.info("👆 **Загрузите CSV или Excel файл** для начала анализа")
    
    st.markdown("### 💡 Как это работает:")
    steps = [
        ("1️⃣", "Загрузите датасет (CSV/Excel)"),
        ("2️⃣", "Введите запрос к аналитику или оставьте пустым для авто-анализа"),
        ("3️⃣", "LLM на gen-api.ru запускает код в своём интерпретаторе и анализирует данные"),
        ("4️⃣", "Вы получаете текстовый отчёт с выводами и метриками"),
        ("5️⃣", "Streamlit автоматически строит интерактивные графики через Plotly")
    ]
    for num, desc in steps:
        st.markdown(f"{num} {desc}")
    
    st.markdown("### 🎯 Примеры запросов:")
    examples = [
        "Сравни среднюю зарплату по отделам",
        "Найди корреляции между числовыми переменными",
        "Есть ли выбросы в столбце age?",
        "Какие категории встречаются чаще всего?",
        "Построй описательную статистику"
    ]
    for ex in examples:
        st.markdown(f"• `{ex}`")

# Футер
st.markdown("---")
st.caption("🤖 AI Analytics Agent | Qwen3.6 • gen-api.ru | Данные не покидают ваш браузер после загрузки")
