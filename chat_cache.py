import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class ChatSession:
    session_id: str
    created_at: float
    last_accessed: float
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "metadata": metadata or {}
        })
        self.last_accessed = time.time()

    def get_context(self, max_messages: int = 10) -> str:
        recent = self.messages[-max_messages:]
        return "\n".join([f"{m['role']}: {m['content'][:200]}" for m in recent])

class ChatCache:
    def __init__(self, ttl_seconds: int = 3600, max_sessions: int = 100):
        self.ttl = ttl_seconds
        self.max_sessions = max_sessions
        self._cache: Dict[str, ChatSession] = {}
        self._lock = Lock()

    def _generate_id(self, user_id: str, file_hash: Optional[str] = None) -> str:
        data = f"{user_id}:{file_hash}:{time.time()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def create_session(self, user_id: str, file_hash: Optional[str] = None) -> str:
        self._cleanup_expired()

        session_id = self._generate_id(user_id, file_hash)

        with self._lock:
            if len(self._cache) >= self.max_sessions:
                oldest = min(self._cache.items(), key=lambda x: x[1].last_accessed)
                del self._cache[oldest[0]]

            self._cache[session_id] = ChatSession(
                session_id=session_id,
                created_at=time.time(),
                last_accessed=time.time()
            )

        return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        with self._lock:
            session = self._cache.get(session_id)
            if session:
                if time.time() - session.last_accessed > self.ttl:
                    del self._cache[session_id]
                    return None
                session.last_accessed = time.time()
                return session
            return None

    def add_to_session(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        session = self.get_session(session_id)
        if session:
            session.add_message(role, content, metadata)

    def get_history(self, session_id: str, max_messages: int = 20) -> List[Dict[str, Any]]:
        session = self.get_session(session_id)
        if session:
            return session.messages[-max_messages:]
        return []

    def delete_session(self, session_id: str):
        with self._lock:
            self._cache.pop(session_id, None)

    def _cleanup_expired(self):
        with self._lock:
            now = time.time()
            expired = [
                sid for sid, session in self._cache.items()
                if now - session.last_accessed > self.ttl
            ]
            for sid in expired:
                del self._cache[sid]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_sessions": len(self._cache),
                "max_sessions": self.max_sessions,
                "ttl_seconds": self.ttl,
                "active_sessions": sum(
                    1 for s in self._cache.values()
                    if time.time() - s.last_accessed < self.ttl
                )
            }
