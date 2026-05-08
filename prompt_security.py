import re
from typing import List

class PromptSecurity:
    """Защита от prompt injection атак"""
    
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
    
    FORBIDDEN_FUNCTIONS = [
        'open', 'eval', 'exec', 'compile', '__import__',
        'getattr', 'setattr', 'delattr', 'globals', 'locals'
    ]
    
    def __init__(self):
        self.patterns = [re.compile(p, re.IGNORECASE) for p in self.DANGEROUS_PATTERNS]
    
    def is_safe(self, prompt: str) -> bool:
        """Проверка промпта на безопасность"""
        for pattern in self.patterns:
            if pattern.search(prompt):
                return False
        
        for func in self.FORBIDDEN_FUNCTIONS:
            if f'{func}(' in prompt:
                return False
        
        if '..' in prompt or '/' in prompt:
            if any(x in prompt for x in ['../', '/etc', '/root', '/home']):
                return False
        
        return True
    
    def sanitize(self, prompt: str) -> str:
        """Очистка промпта от потенциально опасных конструкций"""
