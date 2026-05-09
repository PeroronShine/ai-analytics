import subprocess
import json
import tempfile
import os
import base64
import re
from typing import Optional, Dict, Any
from analytics_core import SecurityGuard
import pandas as pd
import plotly.io as pio

class CodeInterpreter:
    def __init__(self, timeout: int = 12, allowed_methods: list = None):
        self.timeout = timeout
        self.allowed_methods = allowed_methods or [
            'describe', 'head', 'tail', 'info', 'corr', 'groupby', 
            'mean', 'sum', 'count', 'value_counts', 'plot', 'sort_values',
            'fillna', 'dropna', 'astype', 'rename', 'merge', 'concat', 
            'to_html', 'to_dict', 'to_string', 'to_json'
        ]

    def _sanitize_code(self, raw_code: str) -> str:
        if not raw_code: 
            return ""
        code = raw_code.strip()
        # Убираем только markdown code blocks, но не трогаем содержимое
        code = re.sub(r'^```python\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'^```\s*', '', code, flags=re.MULTILINE)
        code = re.sub(r'\s*```$', '', code, flags=re.MULTILINE)
        # Убираем только лишние пробелы в начале/конце строк
        lines = [line.rstrip() for line in code.split('\n')]
        return '\n'.join(lines)

    def execute_python(self, code: str, df: pd.DataFrame) -> Dict[str, Any]:
        if df is None or df.empty:
            return {'success': False, 'error': "DataFrame пуст или не загружен"}
        
        is_safe, error_msg = SecurityGuard.validate_code(code)
        if not is_safe:
            return {'success': False, 'error': f"Security: {error_msg}"}
        
        is_safe_ops, error_msg = SecurityGuard.validate_dataframe_operations(
            self.allowed_methods, code
        )
        if not is_safe_ops:
            return {'success': False, 'error': f"Operations: {error_msg}"}
        
        cleaned_code = self._sanitize_code(code)
        
        try:
            df_sample = df.head(100).copy()
            df_json = df_sample.to_json(orient='split')
            df_b64 = base64.b64encode(df_json.encode('utf-8')).decode('ascii')
            code_b64 = base64.b64encode(cleaned_code.encode('utf-8')).decode('ascii')
        except Exception as e:
            return {'success': False, 'error': f"Ошибка подготовки данных: {e}"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write('# -*- coding: utf-8 -*-\n')
            f.write('import pandas as pd\n')
            f.write('import json\n')
            f.write('import warnings\n')
            f.write('import base64\n')
            f.write('warnings.filterwarnings("ignore")\n')
            f.write(f'df = pd.read_json(base64.b64decode("{df_b64}").decode("utf-8"), orient="split")\n')
            f.write(f'user_code = base64.b64decode("{code_b64}").decode("utf-8")\n')
            f.write('__result__ = None\n')
            f.write('__chart__ = None\n')
            f.write('__error__ = None\n')
            f.write('try:\n')
            f.write('    exec(user_code, {"df": df, "pd": pd})\n')
            f.write('    if "fig" in locals() and hasattr(fig, "to_json"):\n')
            f.write('        __chart__ = fig.to_json()\n')
            f.write('    # Проверяем все возможные переменные с результатом\n')
            f.write('    for var_name in ["result", "df_result", "output", "res"]:\n')
            f.write('        if var_name in locals() and __result__ is None:\n')
            f.write('            val = locals()[var_name]\n')
            f.write('            if isinstance(val, pd.DataFrame):\n')
            f.write('                __result__ = val.to_string()\n')
            f.write('            elif isinstance(val, dict):\n')
            f.write('                __result__ = json.dumps(val, ensure_ascii=False, default=str)\n')
            f.write('            elif __result__ is None:\n')
            f.write('                __result__ = str(val)\n')
            f.write('    # Если результат - последняя строка кода (implicit return)\n')
            f.write('    if __result__ is None:\n')
            f.write('        import sys\n')
            f.write('        from io import StringIO\n')
            f.write('        # Перехватываем stdout\n')
            f.write('        old_stdout = sys.stdout\n')
            f.write('        sys.stdout = StringIO()\n')
            f.write('        exec(user_code, {"df": df, "pd": pd})\n')
            f.write('        captured = sys.stdout.getvalue()\n')
            f.write('        sys.stdout = old_stdout\n')
            f.write('        if captured.strip():\n')
            f.write('            __result__ = captured.strip()\n')
            f.write('except Exception as e:\n')
            f.write('    __error__ = str(e)\n')
            f.write('output = {\n')
            f.write('    "result": __result__,\n')
            f.write('    "chart": __chart__,\n')
            f.write('    "error": __error__\n')
            f.write('}\n')
            f.write('print(json.dumps(output, ensure_ascii=False, default=str))\n')
            script_path = f.name
        
        try:
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
                encoding='utf-8'
            )
            
            if result.returncode != 0:
                return {'success': False, 'error': f"Process error: {result.stderr.strip()}"}
            
            if not result.stdout.strip():
                return {'success': False, 'error': "Пустой вывод"}
                
            output = json.loads(result.stdout.strip())
            return {
                'success': output['error'] is None,
                'result': output['result'],
                'error': output['error'],
                'chart': output['chart']
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': f"Timeout: >{self.timeout}с"}
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f"Parse error: {e}"}
        finally:
            if os.path.exists(script_path):
                try: 
                    os.unlink(script_path)
                except: 
                    pass