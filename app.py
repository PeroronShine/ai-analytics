import ssl
import os
import re
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import traceback
import ast
from contextlib import redirect_stdout, redirect_stderr
import time

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

# Запрещённые модули
FORBIDDEN_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'httpx',
    'urllib', 'ftplib', 'smtplib', 'paramiko', 'pickle', 'marshal', 'eval',
    'exec', 'compile', 'open', 'input', 'breakpoint', 'import'
}
FORBIDDEN_BUILTINS = {'eval', 'exec', 'compile', 'import', 'open', 'input'}


class SafeCodeExecutor:
    """Безопасный исполнитель кода"""
    def __init__(self, df: pd.DataFrame, timeout: int = 30):
        self.df = df.copy()
        self.timeout = timeout
        self.output = []
        self.figures = []

    def _safe_globals(self):
        """Безопасное окружение"""
        allowed_modules = {
            'pd': pd, 'pandas': pd,
            'np': __import__('numpy'), 'numpy': __import__('numpy'),
            'px': px, 'plotly': __import__('plotly'), 'go': go,
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
            **allowed_modules
        }

    def _save_fig_callback(self, fig, filename: str = "plot"):
        """Сохранение графика"""
        self.figures.append(fig)
        return f"График '{filename}' создан"

    def _validate_code(self, code: str) -> tuple:
        """Проверка кода"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in FORBIDDEN_MODULES:
                        return False, f"Запрещённый модуль: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in FORBIDDEN_MODULES:
                    return False, f"Запрещённый модуль: {node.module}"
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                    return False, f"Запрещённая функция: {node.func.id}"

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

            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                exec(code, safe_globals, local_vars)

            result['output'] = self.output
            result['figures'] = self.figures
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
    """Агент на Qwen3.6-Plus"""
    def __init__(self, api_key: str, model: str = "qwen3.6-plus"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.df_info = None

    def prepare_dataset_context(self, df: pd.DataFrame):
        """Подготовка контекста данных"""
        self.df_info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'numeric_cols': df.select_dtypes(include='number').columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'sample': df.head(2).to_dict(orient='records')
        }

    def generate_code(self, user_query: str) -> str:
        """Генерация кода через LLM"""
        system_prompt = """Ты аналитик данных. Отвечай ТОЛЬКО JSON:
{
"thought": "рассуждение",
"code": "Python код (pd, np, px, go уже импортированы! Не пиши import!)",
"explanation": "объяснение"
}
Используй: df, print(), save_fig(fig, 'name'), result"""

        user_prompt = f"Данные: {json.dumps(self.df_info)}\nЗадача: {user_query}"

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2048,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({"error": str(e), "code": "print('Ошибка')", "explanation": "Ошибка API"})

    def run_analysis(self, user_query: str, df: pd.DataFrame) -> dict:
        """Запуск анализа"""
        self.prepare_dataset_context(df)
        plan_response = self.generate_code(user_query)

        try:
            plan = json.loads(plan_response)
        except:
            match = re.search(r'\{[\s\S]*\}', plan_response)
            plan = json.loads(match.group()) if match else {"error": "Parse error", "code": "print('Error')"}

        code = plan.get('code', '')
        executor = SafeCodeExecutor(df)
        exec_result = executor.execute(code)

        return {
            'success': exec_result['success'],
            'thought': plan.get('thought', ''),
            'explanation': plan.get('explanation', ''),
            'code': code,
            'output': exec_result['output'],
            'figures': exec_result['figures'],
            'data_result': exec_result['data_result'],
            'error': exec_result.get('error')
        }


def check_safety(query: str) -> tuple:
    """Проверка безопасности"""
    forbidden = [r'eval\s*\(', r'exec\s*\(', r'import', r'os\.system']
    for pattern in forbidden:
        if re.search(pattern, query, re.IGNORECASE):
            return False, f"Подозрительный паттерн: {pattern}"
    return True, "OK"


# ================= UI =================
st.set_page_config(page_title="AI Analytics", page_icon="🤖", layout="wide")

st.title("AI Analytics Agent")
st.markdown("Анализ данных на Qwen3.6-Plus")

with st.sidebar:
    st.header("Настройки")
    api_key = st.text_input("🔑 DashScope API Key", type="password")
    if api_key:
        st.success("✅ Ключ сохранён")
    
    model = st.selectbox("Модель", ["qwen3.6-plus", "qwen3.5-plus"], index=0)
    st.info("📁 CSV, Excel")

uploaded_file = st.file_uploader("📁 Загрузите файл", type=["csv", "xlsx"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
        st.success(f"✅ Загружен: {uploaded_file.name}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Строк", f"{df.shape[0]:,}")
        col2.metric("Столбцов", df.shape[1])
        col3.metric("Числовых", len(df.select_dtypes(include='number').columns))
        col4.metric("Пропусков", df.isnull().sum().sum())
        
        with st.expander("📋 Превью"):
            st.dataframe(df.head())
        
        query = st.text_area("🔍 Запрос", placeholder="Рассчитай статистику по группам", height=80)
        
        if st.button("🚀 Анализ", type="primary", disabled=not (api_key and query)):
            is_safe, msg = check_safety(query)
            if not is_safe:
                st.warning(msg)
                st.stop()
            
            with st.spinner("🤖 Генерация..."):
                agent = QwenAnalyticsAgent(api_key=api_key, model=model)
                result = agent.run_analysis(query, df)
                
                st.markdown("---")
                
                if result.get('error'):
                    st.error(f"❌ {result['error']}")
                else:
                    if result.get('thought'):
                        st.expander("💭 Мысли", expanded=True).write(result['thought'])
                    
                    st.expander("📝 Код", expanded=True).code(result['code'], language='python')
                    
                    if result.get('output'):
                        st.markdown("📤 **Вывод:**")
                        for line in result['output']:
                            st.text(line)
                    
                    if result.get('data_result') is not None:
                        st.markdown("📊 **Результат:**")
                        if isinstance(result['data_result'], pd.DataFrame):
                            st.dataframe(result['data_result'])
                        else:
                            st.write(result['data_result'])
                    
                    if result.get('figures'):
                        st.markdown("📈 **Графики:**")
                        for fig in result['figures']:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if result.get('explanation'):
                        st.info(f"💡 {result['explanation']}")
    
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")
        st.code(traceback.format_exc())
else:
    st.info("👆 Загрузите файл для анализа")
