import io
import base64
import sys
import traceback
import json
import contextlib
from typing import Any, Dict

class SandboxExecutor:
    """Изолированное выполнение Python-кода для анализа данных."""
    
    def __init__(self, df, max_output_length: int = 15000):
        self.df = df
        self.max_output_length = max_output_length
        self._setup_safe_builtins()
    
    def _setup_safe_builtins(self):
        allowed_names = {
            'range', 'len', 'str', 'int', 'float', 'list', 'dict', 'tuple', 'set',
            'zip', 'enumerate', 'sum', 'min', 'max', 'abs', 'round', 'sorted',
            'filter', 'map', 'print', 'isinstance', 'type', 'hasattr', 'getattr',
            'dir', 'repr', 'format', 'divmod', 'pow', 'chr', 'ord', 'bool',
            'complex', 'frozenset', 'slice', 'next', 'iter', 'reversed',
            'all', 'any', 'bin', 'hex', 'oct', 'ascii', 'bytes', 'bytearray',
            'memoryview', 'hash', 'id', 'help', 'vars', 'enumerate', 'zip'
        }
        dangerous = {'exec', 'eval', 'open', 'input', 'compile', '__import__'}
        allowed_names = allowed_names - dangerous
        
        self.safe_builtins = {}
        builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        for name in allowed_names:
            if name in builtins_dict:
                self.safe_builtins[name] = builtins_dict[name]
    
    def _create_globals(self) -> Dict[str, Any]:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        return {
            '__builtins__': self.safe_builtins,
            'pd': pd,
            'np': np,
            'plt': plt,
            'df': self.df.copy(),
            'json': json,
            'base64': base64,
            'io': io,
        }
    
    def execute(self, code: str, timeout_seconds: int = 30) -> Dict[str, Any]:
        code = code.strip()
        if code.startswith('```'):
            code = code.split('```', 2)[-1]
            if code.startswith('python'):
                code = code[6:]
            code = code.strip('`').strip()
        
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        images = []
        
        original_show = None
        original_savefig = None
        
        try:
            import matplotlib.pyplot as plt
            original_show = plt.show
            original_savefig = plt.savefig
            
            def capture_show(*args, **kwargs):
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                images.append(base64.b64encode(buf.read()).decode('utf-8'))
                plt.close()
            
            def capture_savefig(*args, **kwargs):
                buf = io.BytesIO()
                original_savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                images.append(base64.b64encode(buf.read()).decode('utf-8'))
            
            plt.show = capture_show
            plt.savefig = capture_savefig
            
            globals_dict = self._create_globals()
            
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                exec(code, globals_dict)
            
            stdout = stdout_buffer.getvalue()
            stderr = stderr_buffer.getvalue()
            
            if len(stdout) > self.max_output_length:
                stdout = stdout[:self.max_output_length] + "\n... [вывод обрезан]"
            
            return {
                'success': True,
                'stdout': stdout,
                'stderr': stderr,
                'images': images,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'{type(e).__name__}: {str(e)}',
                'traceback': traceback.format_exc(),
                'stdout': stdout_buffer.getvalue(),
                'stderr': stderr_buffer.getvalue(),
                'images': images
            }
        finally:
            if original_show:
                plt.show = original_show
            if original_savefig:
                plt.savefig = original_savefig
            plt.close('all')
