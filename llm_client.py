import requests
import base64
import json
from typing import List, Dict, Optional
import re

def get_gigachat_token(client_id: str, client_secret: str) -> str:
    """Получение OAuth токена для GigaChat API"""
    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': '24b19549-7981-44b5-b331-b856fad131c6',
        'Authorization': f'Basic {base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()}'
    }
    
    data = {
        'scope': 'GIGACHAT_API_PERS'
    }
    
    response = requests.post(url, headers=headers, data=data, verify=False)
    response.raise_for_status()
    
    return response.json()['access_token']

class GigaChatAgent:
    """Агент для аналитики данных с использованием GigaChat"""
    
    def __init__(self, token: str = None, api_key: str = None):
        self.token = token or api_key
        self.base_url = "https://gigachat.devices.sberbank.ru/api/v1"
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }
    
    def generate_analysis_code(
        self, 
        query: str, 
        dataframe,
        chat_history: List[Dict] = None
    ) -> Dict:
        """Генерация кода для анализа данных"""
        
        # Получение информации о DataFrame
        df_info = {
            'columns': list(dataframe.columns),
            'dtypes': {col: str(dtype) for col, dtype in dataframe.dtypes.items()},
            'shape': dataframe.shape,
            'numeric_columns': list(dataframe.select_dtypes(include='number').columns),
            'sample': dataframe.head(2).to_dict()
        }
        
        # Системный промпт
        system_prompt = """Ты — AI аналитик данных. Твоя задача:
1. Анализировать данные pandas DataFrame
2. Писать безопасный и эффективный Python код
3. Создавать визуализации с помощью plotly
4. Предоставлять статистические выводы

Правила:
- Используй только pandas, numpy, plotly
- DataFrame доступен как переменная 'df'
- Всегда возвращай код в переменной 'code'
- Добавляй пояснения в переменную 'explanation'
- Для графиков используй plotly.express
- Обрабатывай возможные ошибки (пропуски, типы данных)
- Результат сохраняй в переменную 'result' или выводи через print()"""

        # Формирование пользовательского запроса
        user_prompt = f"""
Запрос пользователя: {query}

Информация о датасете:
- Столбцы: {df_info['columns']}
- Типы данных: {df_info['dtypes']}
- Числовые колонки: {df_info['numeric_columns']}
- Размер: {df_info['shape']}

Напиши Python код для выполнения этого запроса.
Код должен:
1. Быть безопасным и эффективным
2. Использовать pandas для анализа
3. При необходимости создавать визуализации plotly
4. Выводить результаты через print()

Верни ответ в формате JSON:
{{
    "code": "python код здесь",
    "explanation": "описание что делает код"
}}"""

        # Формирование сообщений
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": "GigaChat",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json=payload,
                verify=False
            )
            response.raise_for_status()
            
            content = response.json()['choices'][0]['message']['content']
            
            # Извлечение кода из ответа
            code = self._extract_code(content)
            
            return {
                'code': code,
                'explanation': 'Код сгенерирован агентом'
            }
            
        except Exception as e:
            raise Exception(f"Ошибка при обращении к GigaChat: {str(e)}")
    
    def _extract_code(self, content: str) -> str:
        """Извлечение Python кода из ответа"""
        # Попытка найти JSON
        json_match = re.search(r'\{.*"code".*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get('code', content)
            except:
                pass
        
        # Поиск кода в markdown блоках
        if '```python' in content:
            code = content.split('```python')[1].split('```')[0].strip()
        elif '```' in content:
            code = content.split('```')[1].split('```')[0].strip()
        else:
            code = content
        
        return code
