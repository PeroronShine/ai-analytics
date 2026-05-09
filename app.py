import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
import os
from pathlib import Path
from llm_client import GigaChatAnalyticsAgent
from chat_cache import ChatCache
from datetime import datetime

st.set_page_config(
    page_title="AI Analytics Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #fafafa; }
    [data-testid="stSidebar"] { background-color: #1a1d24; }
    h1, h2, h3 { color: #fafafa !important; }
    [data-testid="stMetricValue"] { color: #ffffff !important; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 8px; border: none; padding: 10px 24px; font-weight: 600; }
    .stButton>button:hover { background-color: #ff2b2b; color: white; }
    .success-box { background-color: #2e7d32; padding: 10px; border-radius: 8px; margin: 10px 0; color: white; }
    .info-box { background-color: #1e3a5f; padding: 15px; border-radius: 8px; margin: 10px 0; }
    [data-testid="stFileUploader"] { background-color: #262730; padding: 20px; border-radius: 10px; border: 2px dashed #4a4a4a; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "api_key_saved" not in st.session_state:
    st.session_state.api_key_saved = False
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

cache = ChatCache(ttl_hours=24)


@st.cache_resource
def get_agent(api_key: str):
    return GigaChatAnalyticsAgent(api_key=api_key)


def display_metric(label: str, value: str, icon: str = ""):
    return f"""
    <div style="text-align: center; padding: 15px; background-color: #262730; border-radius: 10px; margin: 5px;">
        <div style="font-size: 28px; font-weight: bold; color: #ffffff;">{value}</div>
        <div style="font-size: 12px; color: #878a8c; margin-top: 5px;">{icon} {label}</div>
    </div>
    """


def main():
    with st.sidebar:
        st.markdown("### 🔐 API Ключ")
        
        api_key = os.getenv('LLM_API_KEY', '')
        if api_key and api_key != 'your_gigachat_auth_token_here':
            st.markdown('<div class="success-box">✅ API ключ сохранён</div>', unsafe_allow_html=True)
            st.session_state.api_key_saved = True
        else:
            api_key_input = st.text_input("Введите API ключ GigaChat", type="password", help="Получите ключ на developers.sber.ru")
            if api_key_input:
                os.environ['LLM_API_KEY'] = api_key_input
                st.markdown('<div class="success-box">✅ API ключ сохранён</div>', unsafe_allow_html=True)
                st.session_state.api_key_saved = True
                api_key = api_key_input
        
        st.divider()
        
        st.markdown("### 🤖 Модель")
        model_options = {
            "GigaChat-Pro": "GigaChat-Pro (рекомендуется)",
            "GigaChat-Max": "GigaChat-Max (максимальное качество)",
            "GigaChat-Lite": "GigaChat-Lite (быстрее)"
        }
        selected_model = st.selectbox("Выберите модель", options=list(model_options.keys()), format_func=lambda x: model_options[x], index=0)
        os.environ['LLM_MODEL'] = selected_model
        
        st.divider()
        
        st.markdown("### 📁 Поддерживаемые форматы:")
        st.info("CSV (.csv), Excel (.xlsx, .xls)")
        
        st.divider()
        
        st.markdown("### 💡 Примеры запросов:")
        examples = [
            "Построй гистограмму распределения возраста",
            "Найди корреляции между числовыми колонками",
            "Рассчитай статистику по группам",
            "Найди аномалии в данных",
            "Покажи тренд продаж по месяцам",
            "Сравни средние значения по категориям"
        ]
        for example in examples:
            st.markdown(f"• *{example}*")
        
        st.divider()
        
        if st.button("🗑️ Очистить всё", use_container_width=True):
            st.session_state.messages = []
            st.session_state.df = None
            st.session_state.analysis_result = None
            st.session_state.generated_code = ""
            st.rerun()
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #ffffff; margin: 0;">🤖 AI Analytics Agent</h1>
        <p style="color: #878a8c; margin-top: 10px;">Агент на GigaChat с Code Interpreter для анализа данных</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.api_key_saved:
        st.warning("⚠️ Пожалуйста, введите API ключ в боковой панели")
        return
    
    st.markdown("### 📂 Загрузите файл с данными")
    
    uploaded_file = st.file_uploader("", type=['csv', 'xlsx', 'xls'], help="Загрузите CSV или Excel файл", label_visibility="collapsed")
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df
            
            st.markdown(f'<div class="success-box">✅ Файл загружен: <b>{uploaded_file.name}</b></div>', unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            missing_values = df.isnull().sum().sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(display_metric("Строк", f"{len(df):,}", "📊"), unsafe_allow_html=True)
            with col2:
                st.markdown(display_metric("Столбцов", len(df.columns), "📐"), unsafe_allow_html=True)
            with col3:
                st.markdown(display_metric("Числовых", len(numeric_cols), "🔢"), unsafe_allow_html=True)
            with col4:
                st.markdown(display_metric("Пропусков", missing_values, "⚠️"), unsafe_allow_html=True)
            
            with st.expander("📋 Превью данных", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("##### Типы данных:")
                st.write(dict(df.dtypes.astype(str)))
            
            st.markdown("### 🔍 Запрос к агенту")
            st.markdown('<p style="color: #878a8c;">Опишите, что нужно проанализировать:</p>', unsafe_allow_html=True)
            
            user_query = st.text_area("", placeholder="Например: 'Рассчитай статистику по группам' или 'Построй график продаж'", height=100, label_visibility="collapsed")
            
            col1, col2 = st.columns([1, 4])
            with col1:
                analyze_btn = st.button("🚀 Запустить анализ агента", use_container_width=True)
            
            if analyze_btn and user_query:
                with st.spinner("Агент анализирует данные..."):
                    try:
                        agent = get_agent(api_key)
                        
                        result = agent.run_agent_cycle(
                            user_query=user_query,
                            df=df,
                            context="",
                            max_charts=3
                        )
                        
                        st.session_state.analysis_result = result
                        st.session_state.generated_code = result.get('code', '')
                        
                        st.markdown("---")
                        st.markdown("### 📊 Результат анализа")
                        
                        # Текстовый ответ агента
                        if result.get('text'):
                            st.markdown(result['text'])
                        
                        if result.get('code'):
                            with st.expander("💻 Сгенерированный код", expanded=False):
                                st.code(result['code'], language='python')
                            
                            # === РЕЗУЛЬТАТ ВЫПОЛНЕНИЯ ===
                            if result.get('execution_result'):
                                exec_res = result['execution_result']
                                st.markdown("---")
                                st.markdown("#### 🎯 Результат выполнения кода:")
                                
                                if exec_res.get('success'):
                                    # Выводим результат текстом
                                    if exec_res.get('result') and exec_res['result'] != "None":
                                        st.markdown("##### 📊 Таблица результатов:")
                                        st.text(exec_res['result'])
                                    
                                    # График если есть
                                    if exec_res.get('chart'):
                                        try:
                                            chart_data = json.loads(exec_res['chart']) if isinstance(exec_res['chart'], str) else exec_res['chart']
                                            fig = go.Figure(chart_data)
                                            fig.update_layout(template="plotly_dark", height=400)
                                            st.plotly_chart(fig, use_container_width=True)
                                        except Exception as e:
                                            st.warning(f"⚠️ Ошибка отображения графика: {e}")
                                else:
                                    st.error(f"⚠️ Ошибка: {exec_res.get('error')}")
                            
                            # Графики из charts (backup)
                            elif result.get('charts'):
                                st.markdown("---")
                                st.markdown("#### 📈 Визуализации:")
                                for i, chart_json in enumerate(result['charts']):
                                    try:
                                        chart_data = json.loads(chart_json) if isinstance(chart_json, str) else chart_json
                                        fig = go.Figure(chart_data)
                                        fig.update_layout(template="plotly_dark", height=400)
                                        st.plotly_chart(fig, use_container_width=True)
                                    except Exception as e:
                                        st.warning(f"⚠️ Ошибка отображения графика: {e}")
                        
                        # Ошибки агента
                        if result.get('error'):
                            st.error(f"⚠️ {result['error']}")
                        
                        # Информация о шагах
                        if result.get('steps_used'):
                            st.info(f"💡 Анализ выполнен за {result['steps_used']} шаг(а)")
                        
                        # История
                        st.session_state.messages.append({
                            "role": "user",
                            "content": user_query,
                            "timestamp": datetime.now()
                        })
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result.get('text', ''),
                            "timestamp": datetime.now()
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Ошибка при анализе: {str(e)}")
                        st.exception(e)
            
            # История запросов
            if st.session_state.messages:
                st.divider()
                st.markdown("### 📜 История запросов")
                for msg in st.session_state.messages[-5:]:
                    role_emoji = "👤" if msg["role"] == "user" else "🤖"
                    st.markdown(f"**{role_emoji} {msg['role'].capitalize()}:** {msg['content']}")
        
        except Exception as e:
            st.error(f"❌ Ошибка загрузки файла: {e}")
            st.exception(e)
    
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 40px;">
            <h3 style="color: #ffffff;">📁 Загрузите файл для начала работы</h3>
            <p style="color: #878a8c;">Поддерживаются форматы CSV и Excel<br>Максимальный размер файла: 200 MB</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💡 Что может агент?")
        cols = st.columns(3)
        with cols[0]:
            st.info("""**📊 Описательная статистика**\n- Средние значения\n- Распределения\n- Корреляции""")
        with cols[1]:
            st.info("""**📈 Визуализации**\n- Графики и диаграммы\n- Гистограммы\n- Scatter plots""")
        with cols[2]:
            st.info("""**🔍 Углубленный анализ**\n- Поиск аномалий\n- Группировки\n- Тренды""")


if __name__ == "__main__":
    main()