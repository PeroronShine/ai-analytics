import re
from typing import List

class PromptSecurity:
    """Защита от prompt injection атак"""
    
    # Опасные паттерны
    DANGEROUS_PATTERNS = [
        r'ignore\s+previous',
        r'forget\s+all',
        r'bypass\s+security',
        r'system\s+prompt',
        r'instruction\s+override',
        r'execute\s+arbitrary',
        r'import\s+os',
        r'import\s+subprocess',
        r'__import__',
        r'eval\s*\(',
        r'exec\s*\(',
        r'open\s*\([\'"]\/',
        r'shutil\.',
        r'os\.system',
        r'os\.popen',
    ]
    
    # Запрещенные функции
    FORBIDDEN_FUNCTIONS = [
        'open', 'eval', 'exec', 'compile', '__import__',
        'getattr', 'setattr', 'delattr', 'globals', 'locals'
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]
    
    def is_safe(self, prompt: str) -> bool:
        """
        Проверка промпта на безопасность
        
        Args:
            prompt: Текст промпта для проверки
        
        Returns:
            True если безопасен, False если обнаружена угроза
        """
        # Проверка на опасные паттерны
        for pattern in self.patterns:
            if pattern.search(prompt):
                return False
        
        # Проверка на запрещенные функции
        for func in self.FORBIDDEN_FUNCTIONS:
            if f'{func}(' in prompt:
                return False
        
        # Проверка на попытки выхода за пределы
        if '..' in prompt or '/' in prompt:
            if any(x in prompt for x in ['../', '/etc', '/root', '/home']):
                return False
        
        return True
    
    def sanitize(self, prompt: str) -> str:
        """
        Очистка промпта от потенциально опасных конструкций
        
        Args:
            prompt: Исходный промпт
        
        Returns:
            Очищенный промпт
        """
        # Удаление многострочных комментариев
        prompt = re.sub(r'#.*', '', prompt)
        
        # Удаление escape-последовательностей
        prompt = prompt.replace('\\n', '').replace('\\t', '')
        
        return prompt.strip()
    
    def validate_code(self, code: str) -> tuple[bool, str]:
        """
        Валидация сгенерированного кода
        
        Args:
            code: Python код для проверки
        
        Returns:
            (is_safe, message)
        """
        # Проверка на импорты
        dangerous_imports = ['os', 'sys', 'subprocess', 'shutil', 'socket']
        for imp in dangerous_imports:
            if f'import {imp}' in code or f'from {imp}' in code:
                return False, f"Обнаружен опасный импорт: {imp}"
        
        # Проверка на вызовы системных команд
        if any(x in code for x in ['os.system', 'os.popen', 'subprocess']):
            return False, "Обнаружен вызов системных команд"
        
        # Проверка на работу с файловой системой
        if re.search(r'open\s*\([\'"]/', code):
            return False, "Обнаружена попытка доступа к файловой системе"
        
        return True, "Код безопасен"
