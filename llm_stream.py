"""
llm_stream.py
SSE streaming для постепенной отправки ответов
"""
import json
from typing import AsyncGenerator, Dict


class StreamResponse:
    """Формат ответа для Server-Sent Events"""
    
    @staticmethod
    def format_chunk(text: str, chart_json: str = None, done: bool = False) -> str:
        """Форматирование чанка для SSE"""
        data = {
            'type': 'chunk' if not done else 'done',
            'text': text,
            'chart': chart_json
        }
        return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
    
    @staticmethod  
    def format_error(error: str) -> str:
        return f"data: {json.dumps({'type': 'error', 'message': error})}\n\n"