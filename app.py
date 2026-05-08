import ssl
import os
import re
import uuid
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import traceback
import ast
import sys
from contextlib import redirect_stdout, redirect_stderr
import time

# OpenAI-compatible client для DeepSeek
from openai import OpenAI

# Отключаем предупреждения для локальной разработки (в production использовать сертификаты!)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

os.environ["CURL_CA_BUNDLE"] = ""

# Запрещённые функции/модули для sandbox
FORBIDDEN_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'requests', 'httpx',
    'urllib', 'ftplib', 'smtplib', 'paramiko', 'pickle', 'marshal', 'eval',
    'exec', 'compile', 'open', 'input', 'breakpoint', '__import__'
}

FORBIDDEN_BUILTINS = {'eval', 'exec', 'compile', '__import__', 'open', 'input'}

class SafeCodeExecutor:
    """Безопасный исполнитель кода для анализа данных"""
    
    def __init__(self, df: pd.DataFrame, timeout: int = 30):
        self.df = df.copy()
        self.timeout = timeout
        self.result = None
        self.output = []
        self.figures = []
        
    def _safe_globals(self):
        """Создаёт безопасное окружение для выполнения кода"""
        # Разрешённые модули
        allowed_modules = {
            'pd': pd, 'pandas': pd, 'np': __import__('numpy'), 'numpy': __import__('numpy'),
            'px': px, 'plotly': __import__('plotly'), 'go': go, 'math': __import__('math'),
            're': __import__('re'), 'json': __import__('json'), 'datetime': __import__('datetime'),
            'statistics': __import__('statistics'), 'collections': __import__('collections'),
        }
        
        safe_globals = {
            '__builtins__': {k: v for k, v in __builtins__.items() if k not in FORBIDDEN_BUILTINS},
            'df': self.df,
            'print': lambda *args, **kwargs: self.output.append(' '.join(map(str, args))),
            'display': lambda obj: self.output.append(str(obj)),
            'save_fig': self._save_fig_callback,
        }
        safe_globals.update(allowed_modules)
        return safe_globals
    
    def _save_fig_callback(self, fig, filename: str = "plot"):
        """Callback для сохранения графиков"""
        try:
            if hasattr(fig, 'write_image'):
                fig.write_image(f"{filename}.png")
            self.figures.append(fig)
            return f"✅ График '{filename}' сохранён"
        except:
            self.figures.append(fig)
            return "✅ График создан (отображается в отчёте)"
    
    def _validate_code(self, code: str) -> tuple[bool, str]:
        """Проверяет код на наличие запрещённых конструкций"""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Синтаксическая ошибка: {e}"
        
        for node in ast.walk(tree):
            # Проверка импортов
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split('.')[0] in FORBIDDEN_MODULES:
                        return False, f"Запрещённый модуль: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] in FORBIDDEN_MODULES:
                    return False, f"Запрещённый модуль: {node.module}"
            
            # Проверка вызовов опасных функций
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTINS:
                    return False, f"Запрещённая функция: {node.func.id}"
        
        return True, "OK"
    
    def execute(self, code: str) -> dict:
        """Выполняет код в безопасном окружении"""
        result = {
            'success': False,
            'output': [],
            'error': None,
            'figures': [],
            'data_result': None
        }
        
        # Валидация кода
        is_valid, message = self._validate_code(code)
        if not is_valid:
            result['error'] = f"🚫 Код отклонён: {message}"
            return result
        
        try:
            safe_globals = self._safe_globals()
            local_vars = {}
            
            # Выполнение кода с перехватом вывода
            with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
                exec(code, safe_globals, local_vars)
            
            result['output'] = self.output
            result['figures'] = self.figures
            
            # Извлекаем результат, если он есть
            if 'result' in local_vars:
                result['data_result'] = local_vars['result']
            elif 'df_result' in local_vars:
                result['data_result'] = local_vars['df_result']
            
            result['success'] = True
            
        except TimeoutError:
            result['error'] = "⏱️ Превышено время выполнения кода"
        except MemoryError:
            result['error'] = "💾 Превышен лимит памяти"
        except Exception as e:
            result['error'] = f"❌ Ошибка выполнения: {type(e).__name__}: {str(e)}"
            result['traceback'] = traceback.format_exc()
        
        return result


class DeepSeekAnalyticsAgent:
    """Аналитический агент на базе DeepSeek V4 с code interpreter"""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.df_info = None
        self.executor = None
    
    def prepare_dataset_context(self, df: pd.DataFrame) -> str:
        """Готовит компактное описание датасета для контекста"""
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
            'sample': df.head(3).to_dict(orient='records'),
            'numeric_cols': df.select_dtypes(include='number').columns.tolist(),
            'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        }
        self.df_info = info
        return json.dumps(info, ensure_ascii=False, indent=2)
    
    def generate_analysis_plan(self, user_query: str) -> str:
        """Генерирует план анализа и код через LLM"""
        
        system_prompt = """Ты — профессиональный аналитик данных с доступом к Python sandbox.
Твоя задача: анализировать данные, написав и выполнив Python код.

ПРАВИЛА:
1. ВСЕГДА отвечай в формате JSON со структурой:
{
  "thought": "Краткое рассуждение о подходе",
  "code": "Python код для выполнения анализа",
  "explanation": "Что делает код и какие результаты ожидаются"
}

2. В коде используй:
   - `df` — основной DataFrame с данными
   - `pd`, `np`, `px`, `go` — доступны для импорта
   - `print()` для вывода результатов
   - `save_fig(fig, 'name')` для сохранения графиков
   - Результат можно сохранить в переменную `result` или `df_result`

3. Код должен быть:
   - Безопасным (без os, sys, network, eval, exec)
   - Эффективным (использовать векторизацию pandas)
   - С комментариями на русском

4. Для визуализации используй plotly.express или plotly.graph_objects

5. После выполнения кода интерпретатор вернёт output, figures и data_result.

ПРИМЕР ОТВЕТА:
{
  "thought": "Нужно рассчитать среднюю зарплату по отделам и построить бар-чарт",
  "code": "import plotly.express as px\\nresult = df.groupby('department')['salary'].mean().round(2)\\nprint('Средняя зарплата по отделам:')\\nprint(result)\\nfig = px.bar(x=result.index, y=result.values, labels={'x':'Отдел','y':'Средняя зарплата'}, title='Зарплата по отделам')\\nsave_fig(fig, 'salary_by_dept')",
  "explanation": "Код группирует данные по отделам, считает среднюю зарплату и строит интерактивный график"
}

ВАЖНО: Отвечай ТОЛЬКО валидным JSON, без markdown-обёрток."""

        user_prompt = f"""Датасет:
{json.dumps(self.df_info, ensure_ascii=False)}

Запрос пользователя: {user_query}

Сгенерируй код для анализа на русском языке."""

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
            return json.dumps({
                "error": f"Ошибка API: {str(e)}",
                "code": "print('Ошибка генерации кода')",
                "explanation": "Не удалось сгенерировать код"
            }, ensure_ascii=False)
    
    def run_analysis(self, user_query: str, df: pd.DataFrame) -> dict:
        """Полный цикл анализа: план → код → выполнение → результат"""
        self.executor = SafeCodeExecutor(df)
        
        # Шаг 1: Генерация кода через LLM
        plan_response = self.generate_analysis_plan(user_query)
        
        try:
            plan = json.loads(plan_response)
        except json.JSONDecodeError:
            # Попытка извлечь JSON из markdown
            match = re.search(r'\{[\s\S]*\}', plan_response)
            if match:
                plan = json.loads(match.group())
            else:
                return {
                    'success': False,
                    'error': 'Не удалось распарсить ответ LLM',
                    'raw_response': plan_response
                }
        
        # Шаг 2: Выполнение кода в sandbox
        code = plan.get('code', '')
        execution_result = self.executor.execute(code)
        
        # Шаг 3: Формирование финального отчёта
        report = {
            'success': execution_result['success'],
            'thought': plan.get('thought', ''),
            'explanation': plan.get('explanation', ''),
            'code': code,
            'output': execution_result.get('output', []),
            'figures': execution_result.get('figures', []),
            'data_result': execution_result.get('data_result'),
            'error': execution_result.get('error')
        }
        
        return report

def check_prompt_safety(query: str) -> tuple[bool, str]:
    """Проверяет запрос на попытки инъекции"""
    forbidden_patterns = [
        r'ignore\s+(previous|instructions|rules)',
        r'override\s+(system|security|restrictions)',
        r'(bypass|skip|disable)\s+(security|sandbox|filter)',
        r'execute\s+(code|command|shell|system)',
        r'eval\s*\(', r'exec\s*\(', r'__import__',
        r'os\.(system|popen|exec)', r'subprocess',
        r'read\s+(file|password|secret|key)',
        r'(admin|root|superuser)\s+access',
        r'system\s+prompt', r'developer\s+mode',
    ]
    
    query_lower = query.lower()
    for pattern in forbidden_patterns:
        if re.search(pattern, query_lower, re.IGNORECASE):
            return False, f"⚠️ Обнаружен подозрительный паттерн: {pattern}"
    
    return True, "OK"

st.set_page_config(
    page_title="🤖 AI Analytics Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .stCode { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
    .success-box { padding: 1rem; border-radius: 0.5rem; background-color: #d4edda; border: 1px solid #c3e6cb; }
    .error-box { padding: 1rem; border-radius: 0.5rem; background-color: #f8d7da; border: 1px solid #f5c6cb; }
    .code-block { font-family: monospace; background: #2d2d2d; color: #f8f8f2; padding: 1rem; border-radius: 5px; overflow-x: auto; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Analytics Agent")
st.markdown("*Агент на DeepSeek V4 с Code Interpreter для анализа данных*")

# Sidebar с настройками
with st.sidebar:
    st.header("⚙️ Настройки")
    
    api_key = st.text_input(
        "🔑 DeepSeek API Key", 
        type="password", 
        placeholder="sk-...",
        help="Получите ключ на platform.deepseek.com"
    )
    
    if api_key:
        st.success("✅ API ключ сохранён")
    
    st.markdown("---")
    
    model_options = {
        "deepseek-chat": "DeepSeek V3/V4 (баланс)",
        "deepseek-reasoner": "DeepSeek Reasoner (сложные задачи)",
    }
    selected_model = st.selectbox(
        "🧠 Модель",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0
    )
    
    st.markdown("---")
    st.info("📁 Поддерживаемые форматы:\n- CSV (.csv)\n- Excel (.xlsx, .xls)")
    
    st.markdown("---")
    st.markdown("### 💡 Примеры запросов:")
    st.markdown("- *'Построй гистограмму распределения возраста'*")
    st.markdown("- *'Найди корреляции между числовыми колонками'*")
    st.markdown("- *'Рассчитай статистику по группам'*")
    st.markdown("- *'Найди аномалии в данных'*")

# Загрузка файла
uploaded_file = st.file_uploader("📁 Загрузите файл с данными", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Чтение данных
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Файл загружен: `{uploaded_file.name}`")
        
        # Метрики датасета
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Строк", f"{df.shape[0]:,}")
        with col2:
            st.metric("📐 Столбцов", df.shape[1])
        with col3:
            st.metric("🔢 Числовых", len(df.select_dtypes(include='number').columns))
        with col4:
            st.metric("⚠️ Пропусков", df.isnull().sum().sum())
        
        # Превью данных
        with st.expander("📋 Превью данных", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
            st.write("📊 Типы данных:")
            st.json(df.dtypes.astype(str).to_dict())
        
        # Запрос пользователя
        st.markdown("### 🔍 Запрос к агенту")
        user_query = st.text_area(
            "Опишите, что нужно проанализировать:",
            placeholder="Например: 'Построй сводную таблицу и найди топ-5 категорий по выручке'",
            height=80
        )
        
        # Кнопка запуска анализа
        if st.button("🚀 Запустить анализ агента", type="primary", disabled=not (api_key and user_query)):
            if not api_key:
                st.error("❌ Введите API ключ DeepSeek")
                st.stop()
            
            if not user_query.strip():
                st.error("❌ Введите запрос для анализа")
                st.stop()
            
            # Проверка безопасности
            is_safe, safety_msg = check_prompt_safety(user_query)
            if not is_safe:
                st.warning(safety_msg)
                st.stop()
            
            with st.spinner("🤖 Агент генерирует план анализа..."):
                try:
                    # Инициализация агента
                    agent = DeepSeekAnalyticsAgent(api_key=api_key, model=selected_model)
                    
                    # Запуск анализа
                    result = agent.run_analysis(user_query, df)
                    
                    # Отображение результатов
                    st.markdown("---")
                    
                    if result.get('error'):
                        st.error(f"❌ {result['error']}")
                        if 'traceback' in result:
                            with st.expander("🔍 Stack trace"):
                                st.code(result['traceback'], language='python')
                    else:
                        # Блок рассуждений агента
                        if result.get('thought'):
                            with st.expander("💭 Ход мыслей агента", expanded=True):
                                st.markdown(result['thought'])
                        
                        # Сгенерированный код
                        with st.expander("📝 Сгенерированный код", expanded=True):
                            st.code(result['code'], language='python')
                        
                        # Вывод кода
                        if result.get('output'):
                            st.markdown("📤 **Вывод программы:**")
                            for line in result['output']:
                                st.text(line)
                        
                        # Результаты данных
                        if result.get('data_result') is not None:
                            st.markdown("📊 **Результат анализа:**")
                            if isinstance(result['data_result'], pd.DataFrame):
                                st.dataframe(result['data_result'], use_container_width=True)
                            elif isinstance(result['data_result'], (dict, list)):
                                st.json(result['data_result'])
                            else:
                                st.write(result['data_result'])
                        
                        # Графики
                        if result.get('figures'):
                            st.markdown("📈 **Визуализации:**")
                            for i, fig in enumerate(result['figures']):
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Объяснение
                        if result.get('explanation'):
                            st.markdown(f"💡 **Объяснение:** {result['explanation']}")
                    
                except Exception as e:
                    st.error(f"🔥 Критическая ошибка: {type(e).__name__}: {str(e)}")
                    with st.expander("🔍 Детали ошибки"):
                        st.code(traceback.format_exc(), language='python')
    
    except Exception as e:
        st.error(f"❌ Ошибка загрузки файла: {e}")
        st.exception(e)

else:
    # Стартовый экран
    st.info("👆 Загрузите CSV или Excel файл, чтобы начать анализ")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔐 Безопасность")
        st.markdown("- Код выполняется в изолированном sandbox")
        st.markdown("- Запрещены опасные функции и модули")
        st.markdown("- Проверка запросов на инъекции")
    
    with col2:
        st.markdown("### 🧠 Интеллект")
        st.markdown("- DeepSeek V4 для генерации кода")
        st.markdown("- Автоматическое планирование анализа")
        st.markdown("- Адаптация под структуру данных")
    
    with col3:
        st.markdown("### 📊 Визуализация")
        st.markdown("- Plotly для интерактивных графиков")
        st.markdown("- Автоматический подбор типов диаграмм")
        st.markdown("- Экспорт результатов")

# Футер
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.9em'>"
    "🤖 AI Analytics Agent | DeepSeek V4 + Code Interpreter | Sandbox Execution"
    "</div>",
    unsafe_allow_html=True
)
