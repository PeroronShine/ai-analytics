import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from llm_client import GigaChatAgent, get_gigachat_token
from code_interpreter import CodeInterpreter
from prompt_security import PromptSecurity
from analytics_core import AnalyticsEngine
import os
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings('ignore')

# Конфигурация страницы
st.set_page_config(
    page_title="AI Analytics Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS для темной темы
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stAlert {
        background-color: #1a1f2e;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Инициализация session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_authorized' not in st.session_state:
    st.session_state.api_authorized = False
if 'dataset_context' not in st.session_state:
    st.session_state.dataset_context = ""

# Sidebar
with st.sidebar:
    st.title("⚙️ Настройки")

    # Авторизация GigaChat
    st.subheader("🔐 GigaChat API")
    auth_method = st.radio("Метод авторизации", ["OAuth SBER", "API Key (Access Token)"])

    if auth_method == "OAuth SBER":
        client_id = st.text_input("Client ID", type="password")
        client_secret = st.text_input("Client Secret", type="password")
        scope = st.selectbox("Scope", ["GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP"])

        if st.button("Получить токен"):
            if not client_id or not client_secret:
                st.error("Введите Client ID и Client Secret")
            else:
                try:
                    token = get_gigachat_token(client_id, client_secret, scope)
                    st.session_state.agent = GigaChatAgent(token=token)
                    st.session_state.api_authorized = True
                    st.success("✅ Авторизация успешна!")
                except Exception as e:
                    st.error(f"❌ Ошибка авторизации: {str(e)}")
                    st.code(traceback.format_exc())

    else:
        api_key = st.text_input("Access Token / API Key", type="password")
        if api_key:
            try:
                st.session_state.agent = GigaChatAgent(api_key=api_key)
                st.session_state.api_authorized = True
                st.success("✅ API ключ принят")
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")

    st.divider()

    # Загрузка файла
    st.subheader("📁 Данные")
    uploaded_file = st.file_uploader(
        "Загрузите CSV или Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Поддерживаемые форматы: CSV, XLSX, XLS"
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)

            st.success(f"✅ Файл загружен: {uploaded_file.name}")
            st.info(f"📊 Строк: {len(st.session_state.df)}")
            st.info(f"📊 Столбцов: {len(st.session_state.df.columns)}")

        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {str(e)}")

    # Контекст датасета
    st.subheader("📝 Контекст датасета")
    st.session_state.dataset_context = st.text_area(
        "Опишите данные (что означают колонки, бизнес-контекст):",
        value=st.session_state.dataset_context,
        height=100,
        placeholder="Например: Колонка 'sales' — выручка в рублях, 'date' — дата продажи..."
    )

# Основной контент
st.title("🤖 AI Analytics Agent")
st.markdown("**Агент на GigaChat для анализа данных с Code Interpreter**")

if not st.session_state.api_authorized:
    st.warning("⚠️ Пожалуйста, авторизуйтесь в GigaChat API в боковой панели")
    st.stop()

if st.session_state.df is None:
    st.info("📂 Загрузите файл с данными в боковой панели")
    st.stop()

# Показ превью данных
with st.expander("👁️ Превью данных", expanded=False):
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Строк", f"{len(st.session_state.df):,}")
    col2.metric("Столбцов", len(st.session_state.df.columns))
    col3.metric("Числовых", len(st.session_state.df.select_dtypes(include='number').columns))
    col4.metric("Пропусков", int(st.session_state.df.isnull().sum().sum()))

# Зона чата
st.subheader("💬 Запрос к агенту")
st.markdown("Опишите, что нужно проанализировать:")

with st.expander("📋 Примеры запросов"):
    st.code("'Построй гистограмму распределения возраста'")
    st.code("'Найди корреляции между числовыми колонками'")
    st.code("'Рассчитай статистику по группам'")
    st.code("'Построй временной ряд продаж'")

query = st.text_area(
    "Ваш запрос:",
    height=100,
    placeholder="Например: 'Проанализируй данные и построй график продаж по месяцам'"
)

if st.button("🚀 Запустить анализ агента", type="primary"):
    if not query:
        st.warning("Введите запрос")
    else:
        security = PromptSecurity()
        interpreter = CodeInterpreter()
        analytics = AnalyticsEngine(st.session_state.df)

        if not security.is_safe(query):
            st.error("⚠️ Обнаружена потенциальная угроза безопасности!")
            st.stop()

        sanitized_query = security.sanitize(query)

        with st.spinner("🤖 Агент анализирует данные..."):
            try:
                code_response = st.session_state.agent.generate_analysis_code(
                    query=sanitized_query,
                    dataframe=st.session_state.df,
                    context=st.session_state.dataset_context or None,
                    chat_history=st.session_state.chat_history
                )

                generated_code = code_response.get('code', '')

                st.info("📝 Сгенерированный код:")
                st.code(generated_code, language='python')

                result = interpreter.execute_with_dataframe(
                    generated_code,
                    st.session_state.df,
                    timeout=30
                )

                if result.get('success'):
                    st.success("✅ Анализ завершен успешно!")

                    if result.get('plot'):
                        try:
                            fig = go.Figure(json.loads(result['plot']))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Не удалось отобразить график: {e}")

                    if result.get('output'):
                        st.markdown("### 📝 Результаты:")
                        st.markdown(result['output'])

                    if result.get('data'):
                        st.markdown("### 📋 Таблица результатов:")
                        st.dataframe(pd.DataFrame(result['data']), use_container_width=True)

                    st.session_state.chat_history.append({
                        'query': query,
                        'code': generated_code,
                        'timestamp': datetime.now()
                    })

                else:
                    st.error(f"❌ Ошибка выполнения: {result.get('error', 'Неизвестная ошибка')}")
                    if result.get('output'):
                        st.text(result['output'])

            except Exception as e:
                st.error(f"❌ Критическая ошибка: {str(e)}")
                st.code(traceback.format_exc(), language='python')

if st.session_state.chat_history:
    with st.expander(f"📜 История запросов ({len(st.session_state.chat_history)})"):
        for i, item in enumerate(reversed(st.session_state.chat_history)):
            st.markdown(f"**{len(st.session_state.chat_history)-i}.** {item['query']}")
            st.caption(f"Время: {item['timestamp'].strftime('%H:%M:%S')}")
            with st.expander("Показать код"):
                st.code(item['code'], language='python')
            st.divider()
