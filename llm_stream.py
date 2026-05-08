"""
SSE (Server-Sent Events) Stream Handler
Потоковая передача результатов агента в реальном времени.
"""

import json
import queue
import threading
from typing import Generator, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class StreamEvent:
    event_type: str  # thought, tool_call, observation, chart, final, error
    data: Dict[str, Any]
    timestamp: float

class SSEStreamHandler:
    """
    Обработчик потоковых событий от агента.
    """
    
    def __init__(self):
        self._queue: queue.Queue[StreamEvent] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
    
    def emit(self, event_type: str, data: Dict[str, Any]):
        """Отправляет событие в поток."""
        if not self._closed:
            event = StreamEvent(
                event_type=event_type,
                data=data,
                timestamp=time.time()
            )
            self._queue.put(event)
    
    def stream(self) -> Generator[str, None, None]:
        """
        Генератор SSE-событий.
        Формат: data: {...}\n\n
        """
        while not self._closed:
            try:
                event = self._queue.get(timeout=0.5)
                
                # Форматирование SSE
                sse_data = {
                    "type": event.event_type,
                    "timestamp": event.timestamp,
                    **event.data
                }
                
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                
                if event.event_type in ('final', 'error'):
                    break
                    
            except queue.Empty:
                # Heartbeat для поддержания соединения
                yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
    
    def close(self):
        """Закрывает поток."""
        with self._lock:
            self._closed = True
            self._queue.put(StreamEvent(
                event_type='close',
                data={},
                timestamp=time.time()
            ))
    
    def agent_callback(self, chunk: Dict[str, Any]):
        """
        Callback для интеграции с агентом.
        """
        event_type = chunk.get('type', 'unknown')
        self.emit(event_type, chunk)
