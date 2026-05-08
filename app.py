import streamlit as st
import pandas as pd
from llm_client import GigaChatAgent
from code_interpreter import CodeInterpreter
from prompt_security import PromptSecurity
from analytics_core import AnalyticsEngine
import os
from datetime import datetime

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

# Sidebar
with st.sidebar:
    st.title("⚙️ Настройки")
    
    # Авторизация GigaChat
    st.subheader("🔐 GigaChat API")
    auth_method = st.radio("Метод авторизации", ["OAuth SBER", "API Key"])
    
    if auth_method == "OAuth SBER":
        client_id = st.text_input("Client ID", type="password")
        client_secret = st.text_input("Client Secret", type="password")
        
        if st.button("Получить токен"):
            try:
                from llm_client import get_gigachat_token
                token = get_gigachat_token(client_id, client_secret)
                st.session_state.agent = GigaChatAgent(token=token)
                st.session_state.api_authorized = True
                st.success("✅ Авторизация успешна!")
            except Exception as e:
                st.error(f"❌ Ошибка: {e}")
    
    else:
        api_key = st.text_input("API Key", type="password")
        if api_key:
            st.session_state.agent = GigaChatAgent(api_key=api_key)
            st.session_state.api_authorized = True
            st.success("✅ API ключ принят")
    
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
            # Определение типа файла
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Файл загружен: {uploaded_file.name}")
            
            # Показ статистики
            st.info(f"📊 Строк: {len(st.session_state.df)}")
            st.info(f"📊 Столбцов: {len(st.session_state.df.columns)}")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки: {e}")

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
    st.dataframe(st.session_state.df.head(10))
    
    # Статистика
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Строк", f"{len(st.session_state.df):,}")
    col2.metric("Столбцов", len(st.session_state.df.columns))
    col3.metric("Числовых", len(st.session_state.df.select_dtypes(include='number').columns))
    col4.metric("Пропусков", st.session_state.df.isnull().sum().sum())

# Зона чата
st.subheader("💬 Запрос к агенту")
st.markdown("Опишите, что нужно проанализировать:")

# Примеры запросов
with st.expander("📋 Примеры запросов"):
    st.code("'Построй гистограмму распределения возраста'")
    st.code("'Найди корреляции между числовыми колонками'")
    st.code("'Рассчитай статистику по группам'")
    st.code("'Построй временной ряд продаж'")

# Поле ввода запроса
query = st.text_area(
    "Ваш запрос:",
    height=100,
    placeholder="Например: 'Проанализируй данные и построй график продаж по месяцам'"
)

# Кнопка анализа
if st.button("🚀 Запустить анализ агента", type="primary"):
    if not query:
        st.warning("Введите запрос")
    else:
        # Инициализация компонентов
        security = PromptSecurity()
        interpreter = CodeInterpreter()
        analytics = AnalyticsEngine(st.session_state.df)
        
        # Проверка на prompt injection
        if not security.is_safe(query):
            st.error("⚠️ Обнаружена потенциальная угроза безопасности!")
            st.stop()
        
        with st.spinner("🤖 Агент анализирует данные..."):
            try:
                # Получение кода от LLM
                code_response = st.session_state.agent.generate_analysis_code(
                    query=query,
                    dataframe=st.session_state.df,
                    chat_history=st.session_state.chat_history
                )
                
                # Выполнение кода
                result = interpreter.execute(code_response['code'], timeout=30)
                
                # Обработка результатов
                if result['success']:
                    st.success("✅ Анализ завершен успешно!")
                    
                    # Показ результатов
                    if 'plot' in result and result['plot']:
                        st.plotly_chart(result['plot'], use_container_width=True)
                    
                    if 'output' in result and result['output']:
                        st.markdown("### 📝 Результаты:")
                        st.markdown(result['output'])
                    
                    if 'data' in result and result['data'] is not None:
                        st.dataframe(result['data'])
                    
                    # Добавление в историю
                    st.session_state.chat_history.append({
                        'query': query,
                        'code': code_response['code'],
                        'timestamp': datetime.now()
                    })
                    
                else:
                    st.error(f"❌ Ошибка выполнения: {result['error']}")
                    st.code(code_response['code'], language='python')
                    
            except Exception as e:
                st.error(f"❌ Критическая ошибка: {str(e)}")

# История запросов
if st.session_state.chat_history:
    with st.expander(f"📜 История запросов ({len(st.session_state.chat_history)})"):
        for i, item in enumerate(st.session_state.chat_history):
            st.markdown(f"**{i+1}.** {item['query']}")
            st.caption(f"Время: {item['timestamp'].strftime('%H:%M:%S')}")
            st.divider()
