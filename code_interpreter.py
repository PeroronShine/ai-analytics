import os
import base64
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from e2b_code_interpreter import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False
    Sandbox = Any  # fallback для type hints

@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    results: List[Any]
    error: Optional[str] = None
    artifacts: Optional[Dict[str, bytes]] = None
    execution_time: float = 0.0

class CodeInterpreter:
    def __init__(self, api_key: Optional[str] = None, timeout: int = 60):
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        self.timeout = timeout
        self.sandbox = None
        self._local_fallback = not (E2B_AVAILABLE and self.api_key)

        if not self._local_fallback:
            try:
                self.sandbox = Sandbox.create(
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            except Exception as e:
                print(f"E2B init failed: {e}. Using local fallback.")
                self._local_fallback = True

    def upload_file(self, file_path: str, remote_name: Optional[str] = None) -> str:
        if self._local_fallback:
            return file_path

        remote_name = remote_name or os.path.basename(file_path)
        with open(file_path, "rb") as f:
            self.sandbox.files.write(f"/home/user/{remote_name}", f.read())
        return f"/home/user/{remote_name}"

    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        if self._local_fallback:
            return self._execute_local(code, context)
        return self._execute_e2b(code, context)

    def _execute_e2b(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        start_time = time.time()

        try:
            if context:
                context_code = self._build_context_code(context)
                code = context_code + "\n\n" + code

            execution = self.sandbox.run_code(code, timeout=self.timeout)

            artifacts = {}
            for result in execution.results:
                if hasattr(result, 'png') and result.png:
                    artifacts['plot.png'] = base64.b64decode(result.png)
                if hasattr(result, 'jpg') and result.jpg:
                    artifacts['plot.jpg'] = base64.b64decode(result.jpg)

            exec_time = time.time() - start_time

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
                execution_time=time.time() - start_time
            )

    def _execute_local(self, code: str, context: Optional[Dict[str, Any]] = None) -> ExecutionResult:
        import subprocess
        import tempfile

        start_time = time.time()

        if context:
            context_code = self._build_context_code(context)
            full_code = context_code + "\n\n" + code
        else:
            full_code = code

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

            exec_time = time.time() - start_time

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
                execution_time=time.time() - start_time
            )
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass

    def _build_context_code(self, context: Dict[str, Any]) -> str:
        lines = ["# Auto-generated context"]

        if 'file_path' in context:
            lines.append("import pandas as pd")
            lines.append(f"df = pd.read_csv('{context['file_path']}')")
            lines.append("print(f'Dataset loaded: {len(df)} rows, {len(df.columns)} columns')")
            lines.append("print(f'Columns: {list(df.columns)}')")

        if 'df_info' in context:
            lines.append(f"# Dataset info: {context['df_info']}")

        return "\n".join(lines)

    def _sanitize_code(self, code: str) -> str:
        dangerous = ['os.system', 'subprocess', 'eval(', 'exec(', '__import__', "open('/"]
        for d in dangerous:
            if d in code:
                code = code.replace(d, f"# BLOCKED: {d}")
        return code

    def close(self):
        if self.sandbox:
            self.sandbox.kill()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
