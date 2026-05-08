import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field

try:
    from gigachat import GigaChat
    from gigachat.models import Chat, Messages, MessagesRole, Function, FunctionParameters
    GIGA_AVAILABLE = True
except ImportError:
    GIGA_AVAILABLE = False
    GigaChat = object
    Chat = object
    Messages = object
    MessagesRole = object
    Function = object
    FunctionParameters = object

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: str

@dataclass
class AgentStep:
    thought: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    observation: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None

@dataclass
class AgentConfig:
    max_iterations: int = 10
    temperature: float = 0.1
    top_p: float = 0.1
    model: str = "GigaChat-Max"
    timeout: int = 120

class GigaChatAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        auth_key: Optional[str] = None,
        config: Optional[AgentConfig] = None,
        verify_ssl: bool = True
    ):
        self.config = config or AgentConfig()
        self.conversation_history: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.tool_registry: Dict[str, Callable] = {}
        self._use_native_functions = True  # Пробуем native function calling

        if not GIGA_AVAILABLE:
            raise ImportError("gigachat package not installed. Run: pip install gigachat")

        if auth_key:
            self.client = GigaChat(
                credentials=auth_key,
                model=self.config.model,
                verify_ssl_certs=verify_ssl,
                timeout=self.config.timeout
            )
        elif api_key:
            self.client = GigaChat(
                access_token=api_key,
                model=self.config.model,
                verify_ssl_certs=verify_ssl,
                timeout=self.config.timeout
            )
        else:
            raise ValueError("Either api_key or auth_key must be provided")

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        func: Callable
    ) -> None:
        tool_schema = {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": list(parameters.keys())
            }
        }
        self.tools.append(tool_schema)
        self.tool_registry[name] = func

    def _build_system_prompt(self, data_context: str = "") -> str:
        base_prompt = """Ты — AI Analytics Agent, эксперт по анализу данных. 
Твоя задача — проводить глубокий анализ предоставленных данных и генерировать Python-код для вычислений и визуализации.

У тебя есть инструмент execute_python для выполнения Python-кода.
Когда тебе нужно выполнить код — используй этот инструмент.

ПРАВИЛА:
1. ВСЕГДА используй инструмент execute_python для выполнения кода анализа
2. НЕ делай предположения — проверяй факты через код
3. Если нужно построить график — генерируй код с matplotlib/plotly/seaborn
4. Возвращай структурированный отчёт с выводами на основе РЕАЛЬНЫХ результатов выполнения кода
5. Если данные неоднозначны — укажи это явно
6. Используй pandas для обработки данных, scipy/statsmodels для статистики

ФОРМАТ ВЫЗОВА ИНСТРУМЕНТА:
Для вызова execute_python используй формат:
<function=execute_python>{{"code": "твой python код здесь"}}</function>

ФОРМАТ ОТВЕТА:
- Начни с краткого плана анализа
- Выполняй шаги последовательно через execute_python
- После каждого шага анализируй результаты
- В конце дай итоговый отчёт с ключевыми находками и рекомендациями
- Если строишь графики — сохраняй их и возвращай путь к файлу

Текущая дата: {date}
"""
        if data_context:
            base_prompt += f"\n\nКОНТЕКСТ ДАННЫХ:\n{data_context}"

        from datetime import datetime
        return base_prompt.format(date=datetime.now().strftime("%Y-%m-%d"))

    def _parse_tool_calls(self, content: str) -> List[ToolCall]:
        """Парсит теги <function=...> из текста модели."""
        tool_calls = []
        # Паттерн: <function=name>{"key": "value"}</function>
        pattern = r'<function=(\w+)>(.*?)</function>'
        matches = re.findall(pattern, content, re.DOTALL)

        for name, args_str in matches:
            try:
                args = json.loads(args_str)
                tool_calls.append(ToolCall(name=name, arguments=args, call_id=f"tc_{len(tool_calls)}"))
            except json.JSONDecodeError:
                # Пробуем найти JSON внутри строки
                try:
                    json_match = re.search(r'\{.*\}', args_str, re.DOTALL)
                    if json_match:
                        args = json.loads(json_match.group())
                        tool_calls.append(ToolCall(name=name, arguments=args, call_id=f"tc_{len(tool_calls)}"))
                except:
                    pass

        return tool_calls

    def _remove_tool_calls_from_content(self, content: str) -> str:
        """Удаляет теги <function=...> из текста."""
        return re.sub(r'<function=\w+>.*?</function>', '', content, flags=re.DOTALL).strip()

    def _build_gigachat_functions(self) -> List[Function]:
        """Конвертирует наши tool schemas в GigaChat Function объекты."""
        functions = []
        for tool in self.tools:
            func = Function(
                name=tool["name"],
                description=tool["description"],
                parameters=FunctionParameters(
                    type=tool["parameters"]["type"],
                    properties=tool["parameters"]["properties"],
                    required=tool["parameters"].get("required", [])
                )
            )
            functions.append(func)
        return functions

    def _build_messages(self, messages: List[Dict[str, Any]]) -> List[Messages]:
        """Конвертирует dict messages в GigaChat Messages объекты."""
        gigachat_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                gigachat_messages.append(Messages(role=MessagesRole.SYSTEM, content=content))
            elif role == "user":
                gigachat_messages.append(Messages(role=MessagesRole.USER, content=content))
            elif role == "assistant":
                gigachat_messages.append(Messages(role=MessagesRole.ASSISTANT, content=content))
            elif role == "function":
                # Для function results используем ASSISTANT с контекстом
                gigachat_messages.append(Messages(
                    role=MessagesRole.ASSISTANT, 
                    content=f"Результат выполнения {msg.get('name', 'tool')}: {content}"
                ))

        return gigachat_messages

    def run_agent(
        self,
        user_prompt: str,
        data_context: str = "",
        file_path: Optional[str] = None,
        stream: bool = False
    ):
        self.conversation_history = []

        system_prompt = self._build_system_prompt(data_context)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        if file_path:
            messages[1]["content"] += f"\n\nФайл с данными: {file_path}"

        if stream:
            return self._run_agent_stream(messages)
        else:
            return self._run_agent_sync(messages)

    def _run_agent_sync(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        steps = []
        final_answer = None

        for iteration in range(self.config.max_iterations):
            try:
                if self._use_native_functions and self.tools:
                    # Пробуем native function calling
                    response = self._call_with_functions(messages)
                else:
                    # Fallback: ручной парсинг тегов
                    response = self._call_simple(messages)

                assistant_message = response.choices[0].message
                content = assistant_message.content or ""

                # Проверяем native function call
                native_tool_calls = []
                if hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                    func_call = assistant_message.function_call
                    try:
                        args = json.loads(func_call.arguments) if isinstance(func_call.arguments, str) else func_call.arguments
                        native_tool_calls.append(ToolCall(name=func_call.name, arguments=args, call_id="native_1"))
                    except:
                        pass

                # Проверяем теги в контенте (manual parsing)
                manual_tool_calls = self._parse_tool_calls(content)

                all_tool_calls = native_tool_calls + manual_tool_calls

                if all_tool_calls:
                    # Убираем tool calls из контента для thought
                    clean_content = self._remove_tool_calls_from_content(content)

                    # Выполняем все tool calls
                    observations = []
                    for tc in all_tool_calls:
                        obs = self._execute_tool(tc.name, tc.arguments)
                        observations.append(str(obs)[:1500])

                        # Добавляем в историю
                        messages.append({
                            "role": "assistant",
                            "content": clean_content or f"Вызов инструмента {tc.name}"
                        })
                        messages.append({
                            "role": "function",
                            "name": tc.name,
                            "content": str(obs)[:1500]
                        })

                    step = AgentStep(
                        thought=clean_content or f"Вызов инструмента {all_tool_calls[0].name}",
                        tool_calls=all_tool_calls,
                        observation="\n---\n".join(observations),
                        is_final=False
                    )
                    steps.append(step)

                else:
                    # Финальный ответ
                    final_answer = content
                    steps.append(AgentStep(
                        thought=content,
                        is_final=True,
                        final_answer=content
                    ))
                    break

            except Exception as e:
                # Если native function calling падает — переключаемся на manual
                if self._use_native_functions and "function" in str(e).lower():
                    self._use_native_functions = False
                    continue

                # Иначе — ошибка
                final_answer = f"Ошибка при выполнении анализа: {str(e)}"
                steps.append(AgentStep(
                    thought=final_answer,
                    is_final=True,
                    final_answer=final_answer
                ))
                break

        return {
            "steps": steps,
            "final_answer": final_answer,
            "iterations": len(steps),
            "conversation": messages
        }

    def _call_with_functions(self, messages: List[Dict[str, Any]]):
        """Вызов с native function calling."""
        gigachat_messages = self._build_messages(messages)
        functions = self._build_gigachat_functions()

        chat = Chat(
            messages=gigachat_messages,
            functions=functions,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )

        return self.client.chat(chat)

    def _call_simple(self, messages: List[Dict[str, Any]]):
        """Вызов без function calling (manual parsing)."""
        gigachat_messages = self._build_messages(messages)

        chat = Chat(
            messages=gigachat_messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p
        )

        return self.client.chat(chat)

    def _run_agent_stream(self, messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
        for iteration in range(self.config.max_iterations):
            try:
                if self._use_native_functions and self.tools:
                    response = self._call_with_functions(messages)
                else:
                    response = self._call_simple(messages)

                assistant_message = response.choices[0].message
                content = assistant_message.content or ""

                yield json.dumps({
                    "type": "thought",
                    "content": content,
                    "iteration": iteration + 1
                }) + "\n"

                native_tool_calls = []
                if hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                    func_call = assistant_message.function_call
                    try:
                        args = json.loads(func_call.arguments) if isinstance(func_call.arguments, str) else func_call.arguments
                        native_tool_calls.append(ToolCall(name=func_call.name, arguments=args, call_id="native_1"))
                    except:
                        pass

                manual_tool_calls = self._parse_tool_calls(content)
                all_tool_calls = native_tool_calls + manual_tool_calls

                if all_tool_calls:
                    clean_content = self._remove_tool_calls_from_content(content)

                    for tc in all_tool_calls:
                        yield json.dumps({
                            "type": "tool_call",
                            "name": tc.name,
                            "arguments": tc.arguments
                        }) + "\n"

                        observation = self._execute_tool(tc.name, tc.arguments)

                        yield json.dumps({
                            "type": "observation",
                            "content": str(observation)[:1000]
                        }) + "\n"

                        messages.append({
                            "role": "assistant",
                            "content": clean_content or f"Вызов инструмента {tc.name}"
                        })
                        messages.append({
                            "role": "function",
                            "name": tc.name,
                            "content": str(observation)[:1500]
                        })
                else:
                    yield json.dumps({
                        "type": "final",
                        "content": content
                    }) + "\n"
                    break

            except Exception as e:
                if self._use_native_functions and "function" in str(e).lower():
                    self._use_native_functions = False
                    continue

                yield json.dumps({
                    "type": "error",
                    "content": str(e)
                }) + "\n"
                break

            time.sleep(0.1)

    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name not in self.tool_registry:
            return f"Error: Tool '{name}' not found"

        try:
            return self.tool_registry[name](**arguments)
        except Exception as e:
            return f"Error executing tool '{name}': {str(e)}"

    def simple_chat(self, prompt: str) -> str:
        chat = Chat(
            messages=[Messages(role=MessagesRole.USER, content=prompt)],
            temperature=self.config.temperature
        )
        response = self.client.chat(chat)
        return response.choices[0].message.content
