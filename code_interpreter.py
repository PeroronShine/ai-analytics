"""
Code Interpreter using E2B Sandbox
Выполнение Python-кода в изолированной среде с поддержкой файлов.
"""

import os
import base64
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from io import BytesIO

try:
    from e2b_code_interpreter import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False
    print("⚠️ e2b-code-interpreter not installed. Using local fallback.")

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    results: List[Any]
    error: Optional[str] = None
    artifacts: Dict[str, bytes] = None  # PNG, CSV и т.д.
    execution_time: float = 0.0

class CodeInterpreter:
    """
    Интерпретатор кода с поддержкой E2B Sandbox.
    Fallback: локальное выполнение в subprocess с ограничениями.
    """
    
    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.timeout = timeout
        self.sandbox: Optional[Sandbox] = None
        self._local_fallback = not (E2B_AVAILABLE and self.api_key)
        
        if not self._local_fallback:
            try:
                self.sandbox = Sandbox.create(
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            except Exception as e:
                print(f"⚠️ E2B init failed: {e}. Using local fallback.")
                self._local_fallback = True
    
    def upload_file(self, file_path: str, remote_name: Optional[str] = None) -> str:
        """Загружает файл в sandbox."""
        if self._local_fallback:
            # Локально — просто возвращаем путь
            return file_path
        
        remote_name = remote_name or os.path.basename(file_path)
        with open(file_path, "rb") as f:
            remote_path = self.sandbox.files.write(f"/home/user/{remote_name}", f.read())
        return f"/home/user/{remote_name}"
    
    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """
        Выполняет Python-код в sandbox.
        """
        if self._local_fallback:
            return self._execute_local(code, context)
        
        return self._execute_e2b(code, context)
    
    def _execute_e2b(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Выполнение в E2B Sandbox."""
        import time as time_module
        
        start_time = time_module.time()
        
        try:
            # Добавляем контекст в начало кода
            if context:
                context_code = self._build_context_code(context)
                code = context_code + "\n\n" + code
            
            execution = self.sandbox.run_code(code, timeout=self.timeout)
            
            # Извлечение артефактов (графики, файлы)
            artifacts = {}
            for result in execution.results:
                if hasattr(result, 'png') and result.png:
                    artifacts['plot.png'] = base64.b64decode(result.png)
                if hasattr(result, 'jpg') and result.jpg:
                    artifacts['plot.jpg'] = base64.b64decode(result.jpg)
            
            exec_time = time_module.time() - start_time
            
            return ExecutionResult(
                success=True,
                stdout=execution.logs.stdout,
                stderr=execution.logs.stderr,
                results=[r.text for r in execution.results if hasattr(r, 'text')],
                artifacts=artifacts if artifacts else None,
                execution_time=exec_time
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                results=[],
                error=str(e),
                execution_time=time_module.time() - start_time
            )
    
    def _execute_local(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        """Локальное выполнение с ограничениями (fallback)."""
        import subprocess
        import tempfile
        import time as time_module
        
        start_time = time_module.time()
        
        # Создаём временный файл
        if context:
            context_code = self._build_context_code(context)
            full_code = context_code + "\n\n" + code
        else:
            full_code = code
        
        # Ограничиваем опасные операции
        restricted_code = self._sanitize_code(full_code)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(restricted_code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            exec_time = time_module.time() - start_time
            
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout,
                stderr=result.stderr,
                results=[],
                error=result.stderr if result.returncode != 0 else None,
                execution_time=exec_time
            )
            
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                results=[],
                error=f"Execution timeout after {self.timeout}s",
                execution_time=self.timeout
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                stdout="",
                stderr="",
                results=[],
                error=str(e),
                execution_time=time_module.time() - start_time
            )
        finally:
            # Очистка
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def _build_context_code(self, context: Dict[str, Any]) -> str:
        """Строит код инициализации контекста."""
        lines = ["# Auto-generated context"]
        
        if 'file_path' in context:
            lines.append(f"import pandas as pd")
            lines.append(f"df = pd.read_csv('{context['file_path']}')")
            lines.append(f"print(f'Dataset loaded: {len(df)} rows, {len(df.columns)} columns')")
            lines.append(f"print(f'Columns: {list(df.columns)}')")
        
        if 'df_info' in context:
            lines.append(f"# Dataset info: {context['df_info']}")
        
        return "\n".join(lines)
    
    def _sanitize_code(self, code: str) -> str:
        """Санитизация кода для локального выполнения."""
        # Блокируем опасные импорты
        dangerous = ['os.system', 'subprocess', 'eval(', 'exec(', '__import__', 'open('/']
        for d in dangerous:
            if d in code:
                code = code.replace(d, f"# BLOCKED: {d}")
        
        return code
    
    def close(self):
        """Закрывает sandbox."""
        if self.sandbox:
            self.sandbox.kill()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
