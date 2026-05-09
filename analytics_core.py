"""
analytics_core.py
Нормализация, валидация и защита аналитических запросов
"""
import re
import pandas as pd
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
import bleach

class ChartSpec(BaseModel):
    """Спецификация графика для Plotly"""
    type: str = Field(..., description="Тип графика: line, bar, scatter, hist, box")
    x_column: Optional[str] = Field(None, description="Колонка для оси X")
    y_column: Optional[str] = Field(None, description="Колонка для оси Y")
    title: str = Field(..., description="Заголовок графика")
    description: str = Field("", description="Пояснение к графику")
    
    @validator('type')
    def validate_chart_type(cls, v):
        allowed = {'line', 'bar', 'scatter', 'hist', 'box', 'pie'}
        if v.lower() not in allowed:
            raise ValueError(f"Тип графика должен быть одним из: {allowed}")
        return v.lower()


class SecurityGuard:
    """Защита от prompt-injection и опасных операций"""
    
    FORBIDDEN_PATTERNS = [
        r'os\.(system|popen|exec)',
        r'subprocess\.(call|run|Popen)',
        r'__import__\s*\(',
        r'eval\s*\(',
        r'exec\s*\(',
        r'open\s*\([^)]*["\'](?:/etc|/proc|\\windows)',
        r'shutil\.(rmtree|remove)',
        r'pickle\.(load|dump)',
        r'getattr\s*\([^,]+,\s*["\']__\w+__["\']',
    ]
    
    @staticmethod
    def sanitize_input(text: str) -> str:
        """Очистка пользовательского ввода"""
        # Удаление потенциально опасных тегов
        cleaned = bleach.clean(text, tags=[], strip=True)
        # Нормализация пробелов
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    @staticmethod
    def validate_code(code: str) -> tuple[bool, str]:
        """Проверка кода на безопасность"""
        for pattern in SecurityGuard.FORBIDDEN_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Обнаружен опасный паттерн: {pattern}"
        
        # Проверка на попытки доступа к системным модулям
        dangerous_imports = ['os', 'sys', 'subprocess', 'socket', 'requests']
        for imp in dangerous_imports:
            if re.search(rf'^\s*import\s+{imp}\b|^\s*from\s+{imp}\b', code, re.MULTILINE):
                return False, f"Запрещён импорт модуля: {imp}"
        
        return True, "OK"
    
    @staticmethod
    def validate_dataframe_operations(allowed_methods: List[str], code: str) -> tuple[bool, str]:
        """Проверка, что используются только разрешённые методы pandas"""
        # Разрешаем только безопасные цепочки вызовов
        pattern = r'df\.(\w+)'
        matches = re.findall(pattern, code)
        for method in matches:
            if method not in allowed_methods and not method.startswith('_'):
                return False, f"Метод df.{method} не в списке разрешённых"
        return True, "OK"