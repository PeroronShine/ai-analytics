"""
llm_client.py
Клиент GigaChat API с function calling и агентным циклом
"""
import os
import json
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters
from analytics_core import ChartSpec, SecurityGuard
from code_interpreter import CodeInterpreter
import pandas as pd

load_dotenv()


class GigaChatAnalyticsAgent:
    """AI-агент для аналитики данных через GigaChat"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('LLM_API_KEY')
        self.model = os.getenv('LLM_MODEL', 'GigaChat-Pro')
        self.temperature = float(os.getenv('LLM_TEMPERATURE', 0.2))
        self.max_tokens = int(os.getenv('LLM_MAX_TOKENS', 1200))
        self.max_steps = int(os.getenv('LLM_AGENT_MAX_STEPS', 5))
        self.code_timeout = int(os.getenv('LLM_CODE_TIMEOUT_SECONDS', 12))
        
        self.client = GigaChat(credentials=self.api_key, verify_ssl_certs=False)
        self.interpreter = CodeInterpreter(timeout=self.code_timeout)
        
        # Определение tool для function calling
        self.analyze_tool = Function(
            name="analyze_data",
            description="Выполнить Python-код для анализа DataFrame. "
                       "Используй pandas для вычислений и plotly для графиков. "
                       "Результат: числовые значения или JSON-спецификация графика.",
            parameters=FunctionParameters(
                type="object",
                properties={
                    "python_code": {
                        "type": "string",
                        "description": "Python код для выполнения. Используй переменную 'df' как DataFrame."
                    },
                    "description": {
                        "type": "string", 
                        "description": "Краткое описание того, что делает код"
                    }
                },
                required=["python_code", "description"]
            )
        )
    
    def _build_system_prompt(self, context: str = "") -> str:
        """Формирование системного промпта для агента"""
        return f"""Ты — аналитический AI-агент для работы с данными.

ПРАВИЛА:
1. Всегда анализируй данные программно через вызов функции analyze_data
2. НЕ генерируй ответы на основе предположений — только на основе реальных вычислений
3. Для визуализации создавай plotly фигуры и сохраняй в переменную 'fig'
4. Код должен быть безопасным: только pandas/plotly, без системных вызовов
5. Отвечай кратко, по делу, с выводами на русском языке

{context if context else ''}

Когда пользователь задаёт вопрос о данных:
1. Проанализируй вопрос
2. Сгенерируй безопасный Python-код для ответа
3. Вызови analyze_data с этим кодом
4. Интерпретируй результат и дай ответ пользователю"""

    def _create_tool_response(self, tool_result: Dict) -> Messages:
        """Создание сообщения с результатом выполнения tool"""
        content = json.dumps({
            'executed': True,
            'success': tool_result.get('success'),
            'result': tool_result.get('result'),
            'chart_available': tool_result.get('chart') is not None
        }, ensure_ascii=False)
        return Messages(role=MessagesRole.FUNCTION, name="analyze_data", content=content)

    def run_agent_cycle(
        self, 
        user_query: str, 
        df: pd.DataFrame, 
        context: str = "",
        max_charts: int = 3
    ) -> Dict[str, Any]:
        """
        Основной цикл агента с function calling
        
        Returns:
            dict с ответом: {'text': str, 'charts': List[str], 'error': Optional[str]}
        """
        # Санитизация ввода
        user_query = SecurityGuard.sanitize_input(user_query)
        
        # Подготовка контекста о данных (только метаданные!)
        data_context = f"""
Датасет: {len(df)} строк, {len(df.columns)} колонок
Колонки: {list(df.columns)}
Типы: {dict(df.dtypes.astype(str))}
Пример первых 3 строк:
{df.head(3).to_markdown(index=False)}
"""
        
        messages = [
            Messages(role=MessagesRole.SYSTEM, content=self._build_system_prompt(context)),
            Messages(role=MessagesRole.USER, content=f"Вопрос: {user_query}\n\n{data_context}")
        ]
        
        charts = []
        last_error = None
        
        for step in range(self.max_steps):
            try:
                # Запрос к LLM с function calling
                chat_request = Chat(
                    model=self.model,
                    messages=messages,
                    functions=[self.analyze_tool],
                    function_call={"name": "analyze_data"},  # Принудительный вызов tool
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                response = self.client.chat(chat_request)
                assistant_msg = response.choices[0].message
                
                # Проверка: модель вызвала функцию?
                if response.choices[0].finish_reason == "function_call":
                    # Парсинг аргументов функции
                    func_args = assistant_msg.function_call.arguments
                    args = func_args if isinstance(func_args, dict) else json.loads(func_args) if isinstance(func_args, str) else {}
                    code = args.get('python_code', '')
                    
                    # Выполнение кода
                    result = self.interpreter.execute_python(code, df)
                    
                    if result['success']:
                        messages.append(assistant_msg)
                        messages.append(self._create_tool_response(result))
                        if result.get('chart') and len(charts) < max_charts:
                            charts.append(result['chart'])
                        last_execution_result = result  
                    else:
                        # Обработка ошибки выполнения
                        error_msg = f"Ошибка выполнения кода: {result['error']}"
                        messages.append(Messages(
                            role=MessagesRole.USER,
                            content=f"⚠️ {error_msg}. Попробуй исправить код или задать вопрос иначе."
                        ))
                        last_error = error_msg
                        break
                else:
                    # Модель дала текстовый ответ без tool call
                    final_answer = assistant_msg.content
                    return {
                        'text': final_answer,
                        'charts': charts,
                        'error': last_error,
                        'steps_used': step + 1
                    }
                    
            except Exception as e:
                return {
                    'text': "Произошла ошибка при обработке запроса.",
                    'charts': charts,
                    'error': str(e),
                    'steps_used': step
                }
        
        # Если достигнут лимит шагов — финальный синтез ответа
        messages.append(Messages(
            role=MessagesRole.USER,
            content="На основе выполненных вычислений дай итоговый ответ на вопрос пользователя."
        ))
        
        final_request = Chat(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        final_response = self.client.chat(final_request)
    
        return {
            'text': final_response.choices[0].message.content,
            'charts': charts,
            'error': last_error,
            'steps_used': self.max_steps,
            'code': code,  # Добавляем код
            'execution_result': result if 'result' in locals() else None  # Добавляем результат выполнения
        }