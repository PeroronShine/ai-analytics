import json
import re
import requests
from typing import List, Dict, Any, Optional
from sandbox import SandboxExecutor


class DataAnalysisAgent:
    """ReAct-агент: думает → пишет код → выполняет → интерпретирует."""
    
    def __init__(self, api_key: str, base_url: str, model: str = "qwen-3-6-plus"):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.sandbox: Optional[SandboxExecutor] = None
        self.is_genapi = 'gen-api.ru' in self.base_url
    
    def set_dataset(self, df):
        self.sandbox = SandboxExecutor(df)
    
    def _call_llm(self, messages: List[Dict], temperature: float = 0.1) -> Dict[str, Any]:
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        
        if self.is_genapi:
            url = self.base_url
            payload = {
                "messages": messages,
                "is_sync": True,
                "temperature": temperature,
                "top_p": 0.9,
                "max_tokens": 2500,
                "response_format": {"type": "json_object"}
            }
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 2500,
                "response_format": {"type": "json_object"}
            }
        
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            
            content = self._extract_content(data)
            
            if content is None:
                content = ""
            
            return {"content": content}
        except requests.exceptions.RequestException as e:
            return {"error": f"HTTP Error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}
    
    def _extract_content(self, data: Dict) -> Optional[str]:
        """Извлекает текст ответа из различных форматов API."""
        if not isinstance(data, dict):
            return str(data)
        
        # Формат gen-api.ru: response[0] (старая версия)
        if "response" in data and isinstance(data["response"], list) and data["response"]:
            return str(data["response"][0])
        
        # Формат gen-api.ru: output
        if "output" in data:
            if isinstance(data["output"], str):
                return data["output"]
            return json.dumps(data["output"], ensure_ascii=False)
        
        # OpenAI-compatible: choices[0].message.content
        if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    return msg.get("content", "")
                return str(msg)
            return str(choice)
        
        # Если ничего не подошло — вернуть весь ответ как строку
        return json.dumps(data, ensure_ascii=False)
    
    def _get_system_prompt(self, dataset_schema: str) -> str:
        return f"""Ты — агент-аналитик данных. У тебя есть доступ к pandas DataFrame `df`.
Твоя задача — анализировать данные, запуская Python-код через инструмент execute_python.
НЕ делай предположений без проверки кодом. Всегда используй код для подсчёта метрик.

СХЕМА ДАТАСЕТА:
{dataset_schema}

ДОСТУПНЫЕ ПЕРЕМЕННЫЕ И МОДУЛИ:
- df: pandas DataFrame с данными
- pd: pandas
- np: numpy
- plt: matplotlib.pyplot
- json, base64, io

ИНСТРУМЕНТ execute_python(code):
Выполняет Python код в безопасной среде. Для графиков используй plt.show() или plt.savefig() — они будут автоматически перехвачены.

ФОРМАТ ОТВЕТА (строго JSON):
Если нужен анализ кодом:
{{"thought": "что я хочу проверить", "action": "execute_python", "action_input": {{"code": "код"}}}}
Если готов ответить:
{{"thought": "логика анализа", "final_answer": "развёрнутый ответ пользователю", "metrics": "ключевые цифры", "insights": ["инсайт1", "инсайт2"]}}

ПРАВИЛА:
1. Код должен быть самодостаточным и валидным Python
2. Не используй опасные конструкции (os, sys, subprocess, eval, exec, open)
3. Всегда проверяй факты кодом, не придумывай цифры
4. Для графиков: строй и вызывай plt.show()
5. Если ошибка — исправь код и попробуй снова"""
    
    def _get_dataset_schema(self) -> str:
        df = self.sandbox.df
        lines = [
            f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов",
            f"Столбцы: {list(df.columns)}",
            "Типы данных:"
        ]
        for col in df.columns:
            lines.append(f"  - {col}: {df[col].dtype}, пропусков: {df[col].isnull().sum()}")
        lines.append("Первые 3 строки:")
        lines.append(df.head(3).to_csv(index=False))
        return "\n".join(lines)
    
    def _parse_response(self, text: Any) -> Dict[str, Any]:
        if text is None:
            return {"final_answer": "Пустой ответ от модели", "thought": "Нет ответа"}
        
        text = str(text).strip()
        
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            text = text.strip()
        
        if not text:
            return {"final_answer": "Пустой ответ от модели", "thought": "Нет ответа"}
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            return {"final_answer": text, "thought": "Неструктурированный ответ"}
    
    def run(self, user_query: str, max_iterations: int = 5) -> Dict[str, Any]:
        if self.sandbox is None:
            return {"error": "Датасет не загружен"}
        
        schema = self._get_dataset_schema()
        messages = [
            {"role": "system", "content": self._get_system_prompt(schema)},
            {"role": "user", "content": user_query}
        ]
        
        steps = []
        final_result = None
        
        for i in range(max_iterations):
            llm_resp = self._call_llm(messages, temperature=0.1)
            
            if "error" in llm_resp:
                return {"error": llm_resp["error"], "steps": steps}
            
            raw_content = llm_resp.get("content", "")
            parsed = self._parse_response(raw_content)
            
            step = {
                "iteration": i + 1,
                "thought": parsed.get("thought", ""),
                "raw": raw_content
            }
            
            if "action" in parsed and parsed["action"] == "execute_python":
                code = parsed.get("action_input", {}).get("code", "")
                step["code"] = code
                
                exec_result = self.sandbox.execute(code)
                step["execution"] = exec_result
                
                obs = self._format_observation(exec_result)
                steps.append(step)
                
                messages.append({"role": "assistant", "content": raw_content})
                messages.append({"role": "user", "content": f"Результат выполнения кода:\n{obs}"})
            elif "final_answer" in parsed:
                step["final_answer"] = parsed["final_answer"]
                steps.append(step)
                final_result = parsed
                break
            else:
                steps.append(step)
                final_result = {"final_answer": raw_content, "thought": parsed.get("thought", "")}
                break
        
        if final_result is None:
            final_result = {"final_answer": "Агент не смог завершить анализ за отведённое число шагов.", "thought": ""}
        
        return {
            "success": True,
            "final_answer": final_result.get("final_answer", ""),
            "thought": final_result.get("thought", ""),
            "metrics": final_result.get("metrics", ""),
            "insights": final_result.get("insights", []),
            "steps": steps,
            "images": self._collect_images(steps)
        }
    
    def _format_observation(self, exec_result: Dict) -> str:
        parts = []
        if exec_result.get("stdout"):
            parts.append(f"[STDOUT]\n{exec_result['stdout']}")
        if exec_result.get("stderr"):
            parts.append(f"[STDERR]\n{exec_result['stderr']}")
        if exec_result.get("error"):
            parts.append(f"[ERROR]\n{exec_result['error']}")
            if exec_result.get("traceback"):
                parts.append(f"[TRACEBACK]\n{exec_result['traceback'][:800]}")
        if exec_result.get("images"):
            parts.append(f"[IMAGES] Сгенерировано графиков: {len(exec_result['images'])}")
        return "\n".join(parts) if parts else "Код выполнен успешно, но не дал вывода."
    
    def _collect_images(self, steps: List[Dict]) -> List[str]:
        imgs = []
        for step in steps:
            if "execution" in step and step["execution"].get("images"):
                imgs.extend(step["execution"]["images"])
        return imgs
