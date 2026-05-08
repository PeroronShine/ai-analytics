from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
import requests
import base64
from typing import List, Dict, Optional
import json

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
    
    response = requests.post(url, headers=headers, data=data)
    response.raise_for_status()
    
    return response.json()['access_token']

class GigaChatAgent:
    """Агент для аналитики данных с использованием GigaChat"""
    
    def __init__(self, token: str = None, api_key: str = None):
        self.client = GigaChat(
            credentials=token or api_key,
            verify_ssl_certs=False
        )
        self.tools = self._define_tools()
    
    def _define_tools(self) -> List[Dict]:
        """Определение доступных инструментов (tools)"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python_code",
                    "description": "Выполняет Python код для анализа данных pandas DataFrame",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python код для выполнения. Должен использовать pandas DataFrame 'df'"
                            },
                            "description": {
                                "type": "string",
                                "description": "Описание того, что делает код"
                            }
                        },
                        "required": ["code", "description"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_visualization",
                    "description": "Создает визуализацию с помощью plotly",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chart_type": {
                                "type": "string",
                                "enum": ["line", "bar", "scatter", "histogram", "box", "heatmap"],
                                "description": "Тип графика"
                            },
                            "x_column": {
                                "type": "string",
                                "description": "Колонка для оси X"
                            },
                            "y_column": {
                                "type": "string",
                                "description": "Колонка для оси Y"
                            },
                            "title": {
                                "type": "string",
                                "description": "Заголовок графика"
                            }
                        },
                        "required": ["chart_type", "title"]
                    }
                }
            }
        ]
    
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
            'sample': dataframe.head(3).to_dict()
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
- Обрабатывай возможные ошибки (пропуски, типы данных)"""

        # Формирование пользовательского запроса
        user_prompt = f"""
Запрос пользователя: {query}

Информация о датасете:
- Столбцы: {df_info['columns']}
- Типы данных: {df_info['dtypes']}
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
            Messages(role=MessagesRole.SYSTEM, content=system_prompt),
            Messages(role=MessagesRole.USER, content=user_prompt)
        ]

        # Добавление истории чата
        if chat_history:
            for item in chat_history[-3:]:  # Последние 3 запроса
                messages.append(Messages(role=MessagesRole.USER, content=item['query']))
        
        try:
            # Запрос к GigaChat с tool calling
            response = self.client.chat(
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.1,  # Низкая температура для точности кода
                max_tokens=2000
            )
            
            # Обработка ответа
            assistant_message = response.choices[0].message
            
            # Если модель вызвала tool
            if assistant_message.tool_calls:
                tool_call = assistant_message.tool_calls[0]
                if tool_call.function.name == "execute_python_code":
                    args = json.loads(tool_call.function.arguments)
                    return {
                        'code': args['code'],
                        'explanation': args.get('description', '')
                    }
            
            # Если модель вернула обычный текст с кодом
            content = assistant_message.content
            if '```python' in content:
                code = content.split('```python')[1].split('```')[0].strip()
            elif '```' in content:
                code = content.split('```')[1].split('```')[0].strip()
            else:
                code = content
            
            return {
                'code': code,
                'explanation': 'Код сгенерирован агентом'
            }
            
        except Exception as e:
            raise Exception(f"Ошибка при обращении к GigaChat: {str(e)}")
