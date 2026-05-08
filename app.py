"""
AI Analytics Agent — Streamlit Application
Тёмная тема, загрузка файлов, чат с агентом, защита от injection.
"""

import streamlit as st
import pandas as pd
import json
import time
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any

# Импорт модулей проекта
from prompt_guard import PromptGuard
from llm_client import GigaChatAgent, AgentConfig
from code_interpreter import CodeInterpreter
from chat_cache import ChatCache
from analytics_core import AnalyticsCore

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="AI Analytics Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS (Тёмная тема) ====================
st.markdown("""
<style>
    /* Основной фон */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Сайдбар */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Заголовки */
    h1, h2, h3 {
        color: #FAFAFA !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Тайтл платформы */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #8B949E;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Карточки */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #58A6FF;
    }
    
    .metric-label {
        color: #8B949E;
        font-size: 0.875rem;
    }
    
    /* Чат */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    .chat-user {
        background-color: #238636;
        margin-left: auto;
        color: white;
    }
    
    .chat-assistant {
        background-color: #161B22;
        border: 1px solid #30363D;
        color: #FAFAFA;
    }
    
    /* Кнопки */
    .stButton > button {
        background-color: #FF4B4B !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    
    .stButton > button:hover {
        background-color: #FF6B6B !important;
        transform: translateY(-1px);
    }
    
    /* Инпуты */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #21262D !important;
        color: #FAFAFA !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }
    
    /* Файл-аплоадер */
    .stFileUploader > div > button {
        background-color: #21262D !important;
        color: #58A6FF !important;
        border: 2px dashed #30363D !important;
        border-radius: 12px !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
        color: #FAFAFA !important;
    }
    
    /* Код */
    .stCodeBlock {
        background-color: #161B22 !important;
        border: 1px solid #30363D !important;
        border-radius: 8px !important;
    }
    
    /* Success/Error сообщения */
    .stSuccess {
        background-color: #23863620 !important;
        border: 1px solid #238636 !important;
        color: #3FB950 !important;
    }
    
    .stError {
        background-color: #F8514920 !important;
        border: 1px solid #F85149 !important;
    }
    
    /* Spinner */
    .stSpinner > div > div {
        border-top-color: #FF4B4B !important;
    }
    
    /* Таблицы */
    .dataframe {
        background-color: #161B22 !important;
        color: #FAFAFA !important;
    }
    
    .dataframe th {
        background-color: #21262D !important;
        color: #58A6FF !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0E1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363D;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484F58;
    }
    
    /* Alert для injection */
    .injection-alert {
        background-color: #F8514920;
        border: 1px solid #F85149;
        color: #F85149;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if 'chat_cache' not in st.session_state:
    st.session_state.chat_cache = ChatCache(ttl_seconds=3600)
if 'current_session' not in st.session_state:
    st.session_state.current_session = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'df_info' not in st.session_state:
    st.session_state.df_info = None
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'prompt_guard' not in st.session_state:
    st.session_state.prompt_guard = PromptGuard(strict_mode=True)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'analysis_running' not in st.session_state:
    st.session_state.analysis_running = False

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("### ⚙️ Настройки")
    
    # API Key Input
    st.markdown("**🔑 GigaChat API Key**")
    api_key = st.text_input(
        "Authorization Key",
        type="password",
        placeholder="Введите ваш API ключ...",
        help="Получите ключ в личном кабинете Sber ID",
        label_visibility="collapsed"
    )
    
    # Проверка и сохранение ключа
    if api_key:
        # Хешируем для проверки (не храним сам ключ в session_state)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:8]
        if st.session_state.get('key_hash') != key_hash:
            try:
                # Тестовое подключение
                test_agent = GigaChatAgent(
                    auth_key=api_key,
                    verify_ssl=False  # Для разработки
                )
                st.session_state.agent = test_agent
                st.session_state.key_hash = key_hash
                st.success(f"✅ API ключ сохранён")
            except Exception as e:
                st.error(f"❌ Ошибка подключения: {str(e)}")
                st.session_state.agent = None
    else:
        st.info("Введите API ключ для начала работы")
    
    st.divider()
    
    # Модель
    st.markdown("**🧠 Модель**")
    model_choice = st.selectbox(
        "Выберите модель",
        ["GigaChat-Max", "GigaChat-Pro", "GigaChat"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Поддерживаемые форматы
    st.markdown("**📁 Поддерживаемые форматы:**")
    st.markdown("""
    - CSV (.csv)
    - Excel (.xlsx, .xls)
    """)
    
    st.divider()
    
    # Примеры запросов
    st.markdown("**💡 Примеры запросов:**")
    examples = [
        "Построй гистограмму распределения возраста",
        "Найди корреляции между числовыми колонками",
        "Рассчитай статистику по группам",
        "Выяви аномалии в данных",
        "Сделай прогноз на основе трендов"
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex[:20]}", use_container_width=True):
            st.session_state.example_query = ex
    
    st.divider()
    
    # Очистка
    if st.button("🗑️ Очистить историю", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_session = None
        st.session_state.uploaded_file_path = None
        st.session_state.df_info = None
        st.rerun()

# ==================== MAIN CONTENT ====================
st.markdown('<div class="main-title">🎯 AI Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Агент на GigaChat с Code Interpreter для анализа данных</div>', unsafe_allow_html=True)

# ==================== FILE UPLOAD ====================
st.markdown("### 📤 Загрузите файл с данными")

uploaded_file = st.file_uploader(
    "Выберите CSV или Excel файл",
    type=['csv', 'xlsx', 'xls'],
    label_visibility="collapsed"
)

if uploaded_file:
    # Сохранение файла
    file_extension = uploaded_file.name.split('.')[-1]
    temp_path = f"/tmp/uploaded_data_{int(time.time())}.{file_extension}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    st.session_state.uploaded_file_path = temp_path
    
    # Чтение и анализ файла
    try:
        if file_extension == 'csv':
            df = pd.read_csv(temp_path)
        else:
            df = pd.read_excel(temp_path)
        
        st.session_state.df_info = {
            "rows": len(df),
            "columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "missing_values": int(df.isnull().sum().sum()),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
        
        # Метрики файла
        cols = st.columns(4)
        metrics = [
            ("📊 Строк", len(df)),
            ("📋 Столбцов", len(df.columns)),
            ("🔢 Числовых", len(df.select_dtypes(include=['number']).columns)),
            ("⚠️ Пропусков", int(df.isnull().sum().sum()))
        ]
        
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{value:,}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Превью данных
        with st.expander("📋 Превью данных", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Ошибка чтения файла: {str(e)}")
        st.session_state.uploaded_file_path = None

# ==================== CHAT INTERFACE ====================
st.markdown("### 💬 Запрос к агенту")

# Поле ввода запроса
default_query = st.session_state.get('example_query', '')
query = st.text_area(
    "Опишите, что нужно проанализировать:",
    value=default_query,
    placeholder="Например: Рассчитай статистику по группам, построй графики распределения...",
    height=100,
    label_visibility="collapsed"
)

# Кнопка запуска
run_disabled = not (st.session_state.agent and st.session_state.uploaded_file_path)
run_button = st.button(
    "🚀 Запустить анализ агента",
    disabled=run_disabled,
    use_container_width=True
)

if run_button and query:
    # === PROMPT INJECTION GUARD ===
    guard = st.session_state.prompt_guard
    guard_result = guard.scan(query)
    
    if not guard_result.is_safe:
        st.markdown(f"""
        <div class="injection-alert">
            <strong>🛡️ Обнаружена подозрительная активность!</strong><br>
            Риск: {guard_result.risk_score:.2f}<br>
            Причина: {guard_result.reason}<br>
            Запрос отклонён для безопасности.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Безопасный запрос — запускаем анализ
        st.session_state.analysis_running = True
        
        # Добавляем в историю
        st.session_state.messages.append({
            "role": "user",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        
        # Подготовка контекста данных
        df_info = st.session_state.df_info
        data_context = f"""
        Датасет: {st.session_state.uploaded_file_path.split('/')[-1]}
        Размер: {df_info['rows']} строк, {df_info['columns']} колонок
        Числовые колонки: {df_info['numeric_columns']}
        Пропусков: {df_info['missing_values']}
        Колонки: {', '.join(df_info['column_names'])}
        Типы данных: {json.dumps(df_info['dtypes'], ensure_ascii=False)}
        """
        
        # Регистрация инструментов для агента
        agent = st.session_state.agent
        
        def execute_python(code: str) -> str:
            """Инструмент выполнения Python-кода."""
            with CodeInterpreter(timeout=60) as interpreter:
                result = interpreter.execute(
                    code,
                    context={
                        'file_path': st.session_state.uploaded_file_path,
                        'df_info': data_context
                    }
                )
                
                output = []
                if result.stdout:
                    output.append(f"STDOUT:\n{result.stdout}")
                if result.stderr:
                    output.append(f"STDERR:\n{result.stderr}")
                if result.error:
                    output.append(f"ERROR:\n{result.error}")
                if result.results:
                    output.append(f"RESULTS:\n{result.results}")
                
                return "\n\n".join(output)
        
        # Регистрируем инструмент
        agent.register_tool(
            name="execute_python",
            description="Execute Python code for data analysis with pandas, numpy, matplotlib, seaborn, plotly. The dataframe is pre-loaded as 'df' variable.",
            parameters={
                "code": {
                    "type": "string",
                    "description": "Python code to execute for data analysis"
                }
            },
            func=execute_python
        )
        
        # Запуск агента
        with st.spinner("🤖 Агент проводит анализ..."):
            try:
                result = agent.run_agent(
                    user_prompt=query,
                    data_context=data_context,
                    file_path=st.session_state.uploaded_file_path
                )
                
                # Обработка результата
                analytics = AnalyticsCore()
                report = analytics.process_agent_result(result)
                
                # Добавляем ответ в историю
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['final_answer'],
                    "report": report,
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                st.error(f"❌ Ошибка агента: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Ошибка при выполнении анализа: {str(e)}",
                    "error": True,
                    "timestamp": datetime.now().isoformat()
                })
        
        st.session_state.analysis_running = False
        st.rerun()

# ==================== DISPLAY MESSAGES ====================
st.markdown("---")

for msg in st.session_state.messages:
    if msg['role'] == 'user':
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-end;">
            <div class="chat-message chat-user">
                <strong>Вы:</strong><br>{msg['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Ассистент
        st.markdown(f"""
        <div style="display: flex; justify-content: flex-start;">
            <div class="chat-message chat-assistant">
                <strong>🤖 Агент:</strong><br>{msg['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Если есть отчёт — показываем детали
        if 'report' in msg:
            report = msg['report']
            
            with st.expander("📊 Детали анализа", expanded=False):
                # Статистика
                stats = report.get('statistics', {})
                c1, c2, c3 = st.columns(3)
                c1.metric("Шагов анализа", stats.get('steps_count', 0))
                c2.metric("Выполнений кода", stats.get('code_executions', 0))
                c3.metric("Графиков", stats.get('charts_generated', 0))
                
                # Ключевые находки
                findings = report.get('key_findings', [])
                if findings:
                    st.markdown("**🔍 Ключевые находки:**")
                    for finding in findings:
                        st.markdown(f"- {finding}")
                
                # Выполненный код
                code_snippets = report.get('code_executed', [])
                if code_snippets:
                    with st.expander("💻 Сгенерированный код"):
                        for i, code in enumerate(code_snippets, 1):
                            st.markdown(f"**Шаг {i}:**")
                            st.code(code, language='python')
                
                # Наблюдения
                observations = report.get('observations', [])
                if observations:
                    with st.expander("📋 Результаты выполнения"):
                        for obs in observations:
                            st.text(obs)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #484F58; font-size: 0.75rem;">
    AI Analytics Platform • Powered by GigaChat Max • E2B Sandbox
</div>
""", unsafe_allow_html=True)
