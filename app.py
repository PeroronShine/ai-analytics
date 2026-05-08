import re
import base64
import pandas as pd
import streamlit as st
from agent import DataAnalysisAgent

# ============== ЗАЩИТА ==============
FORBIDDEN_PATTERNS = [
    r'(?i)(?:ignore|disregard).*?(?:previous|above|earlier).*?(?:instruction|prompt)',
    r'(?i)(?:new|ignore).*?(?:instruction|prompt)',
    r'(?i)\bDAN\b',
    r'(?i)jailbreak',
    r'<script',
    r'javascript:',
    r'data:text/html',
    r'(?i)\bprompt\b.*?\binjection\b',
]

def check_safety(text: str) -> tuple:
    if not text:
        return True, ""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, text):
            return False, "⚠️ Обнаружена попытка манипуляции (prompt injection)"
    return True, ""

# ============== UI ==============
st.set_page_config(page_title="🤖 AI Data Agent", page_icon="📊", layout="wide")

st.markdown("""
<style>
.insight-card {background: #e8f5e9; padding: 12px; border-radius: 8px; margin: 6px 0; border-left: 4px solid #4caf50;}
.metric-card {background: #e3f2fd; padding: 12px; border-radius: 8px; margin: 6px 0; border-left: 4px solid #2196f3;}
</style>
""", unsafe_allow_html=True)

st.title("AI Data Agent")
st.caption("Агентная аналитика: LLM пишет и выполняет Python-код над вашими данными")

for key in ['agent', 'chat_history', 'df', 'file_name']:
    if key not in st.session_state:
        st.session_state[key] = None if key != 'chat_history' else []

with st.sidebar:
    st.header("⚙️ Настройки API")
    api_key = st.text_input("🔑 API Token", type="password")
    base_url = st.text_input(
        "🔗 API Base URL",
        value="https://api.gen-api.ru/api/v1/networks/qwen-3-6-plus",
        help="Для gen-api.ru укажите полный URL с моделью. Для OpenAI: https://api.openai.com/v1"
    )
    model = st.text_input("🧠 Model ID (если OpenAI-compatible)", value="qwen-3-6-plus")
    
    st.divider()
    st.header("📁 Данные")
    uploaded_file = st.file_uploader("Загрузите CSV или Excel", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if st.session_state.file_name != uploaded_file.name:
            st.session_state.df = None
            st.session_state.agent = None
            st.session_state.chat_history = []
            st.session_state.file_name = uploaded_file.name
        
        if st.session_state.df is None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                st.session_state.df = df
                st.success(f"✅ {uploaded_file.name}: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
            except Exception as e:
                st.error(f"Ошибка чтения файла: {e}")
    
    if st.session_state.df is not None and st.button("🗑️ Сбросить чат и данные"):
        for key in ['agent', 'chat_history', 'df', 'file_name']:
            st.session_state[key] = None if key != 'chat_history' else []
        st.rerun()
    
    st.divider()
    st.info("""
    **Как работает:**
    1. LLM получает схему данных
    2. Планирует анализ и пишет Python-код
    3. Код выполняется в изолированной среде
    4. LLM интерпретирует результаты
    5. Вы получаете отчёт с цифрами и графиками
    """)

if st.session_state.df is not None:
    df = st.session_state.df
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Строк", f"{df.shape[0]:,}")
    c2.metric("Столбцов", df.shape[1])
    c3.metric("Числовых", len(df.select_dtypes(include='number').columns))
    c4.metric("Пропусков", df.isnull().sum().sum())
    
    with st.expander("📋 Предпросмотр данных"):
        st.dataframe(df.head(10), use_container_width=True)
    
    st.markdown("---")
    st.subheader("💬 Аналитический чат")
    
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg['role']):
            if msg['role'] == 'user':
                st.write(msg['content'])
            else:
                data = msg['content']
                st.write(data.get('final_answer', ''))
                
                if data.get('metrics'):
                    st.markdown(
                        f"<div class='metric-card'><b>📊 Метрики:</b><br><pre>{data['metrics']}</pre></div>",
                        unsafe_allow_html=True
                    )
                
                if data.get('insights'):
                    for ins in data['insights']:
                        st.markdown(f"<div class='insight-card'>💡 {ins}</div>", unsafe_allow_html=True)
                
                if data.get('images'):
                    cols = st.columns(min(len(data['images']), 2))
                    for i, img_b64 in enumerate(data['images']):
                        img_bytes = base64.b64decode(img_b64)
                        cols[i % 2].image(img_bytes, use_container_width=True)
                
                if msg.get('steps') and st.toggle("🔍 Показать шаги агента", key=f"steps_{idx}"):
                    for step in msg['steps']:
                        st.markdown(f"**Шаг {step['iteration']}** — *{step.get('thought', '')}*")
                        if 'code' in step:
                            st.code(step['code'], language='python')
                            if 'execution' in step:
                                ex = step['execution']
                                if ex.get('stdout'):
                                    st.text(ex['stdout'][:800])
                                if ex.get('error'):
                                    st.error(ex['error'][:400])
    
    if st.session_state.agent is None and api_key and base_url:
        try:
            st.session_state.agent = DataAnalysisAgent(
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            st.session_state.agent.set_dataset(st.session_state.df)
        except Exception as e:
            st.error(f"Ошибка инициализации агента: {e}")
    
    user_query = st.chat_input("Задайте вопрос по данным...")
    
    if user_query:
        safe, warn = check_safety(user_query)
        if not safe:
            st.warning(warn)
        elif not api_key:
            st.error("Введите API ключ")
        elif st.session_state.agent is None:
            st.error("Ошибка инициализации агента")
        else:
            st.session_state.chat_history.append({'role': 'user', 'content': user_query})
            
            with st.chat_message("assistant"):
                with st.spinner("🤖 Агент думает и пишет код..."):
                    result = st.session_state.agent.run(user_query)
                
                if result.get('error'):
                    st.error(f"Ошибка: {result['error']}")
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': {'final_answer': f"Ошибка: {result['error']}"},
                        'steps': []
                    })
                else:
                    st.write(result.get('final_answer', ''))
                    
                    if result.get('metrics'):
                        st.markdown(
                            f"<div class='metric-card'><b>📊 Метрики:</b><br><pre>{result['metrics']}</pre></div>",
                            unsafe_allow_html=True
                        )
                    
                    if result.get('insights'):
                        for ins in result['insights']:
                            st.markdown(f"<div class='insight-card'>💡 {ins}</div>", unsafe_allow_html=True)
                    
                    if result.get('images'):
                        cols = st.columns(min(len(result['images']), 2))
                        for i, img_b64 in enumerate(result['images']):
                            img_bytes = base64.b64decode(img_b64)
                            cols[i % 2].image(img_bytes, use_container_width=True)
                    
                    with st.expander("🔍 Шаги агента"):
                        for step in result.get('steps', []):
                            st.markdown(f"**Шаг {step['iteration']}** — *{step.get('thought', '')}*")
                            if 'code' in step:
                                st.code(step['code'], language='python')
                                if 'execution' in step:
                                    ex = step['execution']
                                    if ex.get('stdout'):
                                        st.text(ex['stdout'][:1000])
                                    if ex.get('error'):
                                        st.error(ex['error'][:500])
                    
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': result,
                        'steps': result.get('steps', [])
                    })
            
            st.rerun()
else:
    st.info("👆 Загрузите датасет в боковой панели, чтобы начать анализ")
    st.markdown("""
    ### 💡 Примеры вопросов:
    - Проведи полный EDA и найди ключевые инсайты
    - Есть ли выбросы в числовых колонках?
    - Построй корреляционную матрицу и объясни связи
    - Сравни средние значения по категориям
    """)
