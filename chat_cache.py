"""
chat_cache.py
Хранение истории диалогов с автоматической очисткой
"""
import json
import time
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime, timedelta


class ChatCache:
    """Простой файловый кэш для истории чатов"""
    
    def __init__(self, cache_dir: str = ".cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
    
    def _get_cache_path(self, session_id: str) -> Path:
        return self.cache_dir / f"{session_id}.json"
    
    def save(self, session_id: str, messages: List[Dict], metadata: Dict = None):
        """Сохранение истории"""
        data = {
            'timestamp': time.time(),
            'messages': messages,
            'metadata': metadata or {}
        }
        with open(self._get_cache_path(session_id), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, session_id: str) -> Optional[Dict]:
        """Загрузка истории с проверкой TTL"""
        path = self._get_cache_path(session_id)
        if not path.exists():
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Проверка срока жизни
        if time.time() - data['timestamp'] > self.ttl_seconds:
            path.unlink()  # Удаление просроченного кэша
            return None
        
        return data
    
    def cleanup_expired(self):
        """Очистка просроченных кэшей"""
        now = time.time()
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                if now - data.get('timestamp', 0) > self.ttl_seconds:
                    cache_file.unlink()
            except:
                continue  # Пропуск повреждённых файлов