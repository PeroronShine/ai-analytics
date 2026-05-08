import os
import json
import time
from typing import List, Dict, Any, Optional, Callable, Generator
from dataclasses import dataclass, field

try:
    from gigachat import GigaChat
    GIGA_AVAILABLE = True
except ImportError:
    GIGA_AVAILABLE = False
    GigaChat = Any  # fallback для type hints

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

ПРАВИЛА:
1. ВСЕГДА используй инструмент execute_python для выполнения кода анализа
2. НЕ делай предположения — проверяй факты через код
3. Если нужно построить график — генерируй код с matplotlib/plotly/seaborn
4. Возвращай структурированный отчёт с выводами на основе РЕАЛЬНЫХ результатов выполнения кода
5. Если данные неоднозначны — укажи это явно
6. Используй pandas для обработки данных, scipy/statsmodels для статистики

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
            response = self.client.chat(
                messages=messages,
                functions=self.tools if self.tools else None,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )

            assistant_message = response.choices[0].message
            content = assistant_message.content or ""

            if hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                func_call = assistant_message.function_call
                tool_name = func_call.name
                tool_args = json.loads(func_call.arguments) if isinstance(func_call.arguments, str) else func_call.arguments

                observation = self._execute_tool(tool_name, tool_args)

                step = AgentStep(
                    thought=content,
                    tool_calls=[ToolCall(name=tool_name, arguments=tool_args, call_id="1")],
                    observation=str(observation)[:2000],
                    is_final=False
                )
                steps.append(step)

                messages.append({
                    "role": "assistant",
                    "content": content,
                    "function_call": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args)
                    }
                })
                messages.append({
                    "role": "function",
                    "name": tool_name,
                    "content": str(observation)[:2000]
                })

            else:
                final_answer = content
                steps.append(AgentStep(
                    thought=content,
                    is_final=True,
                    final_answer=content
                ))
                break

        return {
            "steps": steps,
            "final_answer": final_answer,
            "iterations": len(steps),
            "conversation": messages
        }

    def _run_agent_stream(self, messages: List[Dict[str, Any]]) -> Generator[str, None, None]:
        for iteration in range(self.config.max_iterations):
            response = self.client.chat(
                messages=messages,
                functions=self.tools if self.tools else None,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )

            assistant_message = response.choices[0].message
            content = assistant_message.content or ""

            yield json.dumps({
                "type": "thought",
                "content": content,
                "iteration": iteration + 1
            }) + "\n"

            if hasattr(assistant_message, 'function_call') and assistant_message.function_call:
                func_call = assistant_message.function_call
                tool_name = func_call.name
                tool_args = json.loads(func_call.arguments) if isinstance(func_call.arguments, str) else func_call.arguments

                yield json.dumps({
                    "type": "tool_call",
                    "name": tool_name,
                    "arguments": tool_args
                }) + "\n"

                observation = self._execute_tool(tool_name, tool_args)

                yield json.dumps({
                    "type": "observation",
                    "content": str(observation)[:1000]
                }) + "\n"

                messages.extend([
                    {
                        "role": "assistant",
                        "content": content,
                        "function_call": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args)
                        }
                    },
                    {
                        "role": "function",
                        "name": tool_name,
                        "content": str(observation)[:2000]
                    }
                ])
            else:
                yield json.dumps({
                    "type": "final",
                    "content": content
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
        response = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature
        )
        return response.choices[0].message.content
