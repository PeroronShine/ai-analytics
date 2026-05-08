import requests
import uuid
import base64
import json
import urllib3
from typing import Dict, Any, Optional
import pandas as pd

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
BASE_URL = "https://gigachat.devices.sberbank.ru/api/v1"


def get_gigachat_token(client_id: str, client_secret: str, scope: str = "GIGACHAT_API_PERS") -> str:
    """Получение OAuth токена для GigaChat"""
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"scope": scope}

    response = requests.post(AUTH_URL, headers=headers, data=data, verify=False)
    response.raise_for_status()
    return response.json()["access_token"]


class GigaChatAgent:
    """Агент для генерации аналитического кода через GigaChat API"""

    def __init__(self, token: Optional[str] = None, api_key: Optional[str] = None, model: str = "GigaChat"):
        if token:
            self.token = token
        elif api_key:
            self.token = api_key
        else:
            raise ValueError("Необходимо указать token или api_key")

        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/json"
        }

    def _build_system_prompt(self) -> str:
        return """Ты — агент данных (Data Agent). Твоя задача — писать Python код для анализа данных.
Правила:
1. Пиши только исполняемый Python код. Никаких пояснений вне кода.
2. Данные доступны в переменной `df` (pandas DataFrame).
3. Для визуализации используй plotly (px или go). Если строишь график, присвой фигуру переменной `agent_output_fig`.
4. Если нужно вернуть таблицу результатов, присвой её переменной `agent_output_df`.
5. Используй только безопасные операции: pandas, numpy, plotly. Не используй `os`, `sys`, `subprocess`, `eval`, `exec`, `open`.
6. Выводи текстовые результаты через `print()`.
7. Код должен быть самодостаточным и обрабатывать возможные ошибки (например, отсутствие колонок).
8. Верни код внутри блока ```python ... ```."""

    def _build_user_prompt(
        self,
        query: str,
        df: pd.DataFrame,
        context: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> str:
        metadata = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "head": df.head(5).to_dict(orient="records"),
            "numeric_columns": list(df.select_dtypes(include="number").columns),
            "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
        }

        prompt_parts = []
        if context:
            prompt_parts.append(f"Контекст/инструкция к датасету: {context}\n")

        prompt_parts.append(f"Запрос пользователя: {query}\n")
        prompt_parts.append(f"Метаданные датафрейма:\n{json.dumps(metadata, ensure_ascii=False, indent=2)}\n")
        prompt_parts.append("Напиши Python код для выполнения запроса. Следуй системной инструкции.")

        return "\n".join(prompt_parts)

    def generate_analysis_code(
        self,
        query: str,
        dataframe: pd.DataFrame,
        context: Optional[str] = None,
        chat_history: Optional[list] = None
    ) -> Dict[str, Any]:
        """Генерация кода анализа через GigaChat API"""

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": self._build_user_prompt(query, dataframe, context, chat_history)}
        ]

        if chat_history:
            for item in chat_history[-3:]:
                messages.append({"role": "user", "content": item["query"]})
                messages.append({"role": "assistant", "content": f"```python\n{item['code']}\n```"})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 4000
        }

        response = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=self.headers,
            json=payload,
            verify=False
        )
        response.raise_for_status()
        result = response.json()

        content = result["choices"][0]["message"]["content"]
        code = self._extract_code(content)

        return {"code": code, "raw_response": content}

    def _extract_code(self, text: str) -> str:
        """Извлечение Python кода из ответа LLM"""
        import re

        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        return text.strip()
