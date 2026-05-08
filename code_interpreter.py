import subprocess
import json
import tempfile
import os
from typing import Dict, Any
import pandas as pd

class CodeInterpreter:
    """Безопасный интерпретатор Python кода"""
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    def execute(self, code: str, timeout: int = None) -> Dict[str, Any]:
        """Выполнение Python кода в изолированной среде"""
        timeout = timeout or self.timeout
        
        wrapper_code = f"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from io import StringIO

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

try:
    {code}
    
    result = {{
        'success': True,
        'output': mystdout.getvalue(),
        'error': None
    }}
    
except Exception as e:
    result = {{
        'success': False,
        'output': None,
        'error': str(e)
    }}

finally:
    sys.stdout = old_stdout

print(json.dumps(result, default=str))
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(wrapper_code)
            temp_file = f.name
        
        try:
            process = subprocess.Popen(
                ['python', temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                
                if stderr:
                    return {
                        'success': False,
                        'error': stderr,
                        'output': None
                    }
                
                result = json.loads(stdout.strip())
                return result
                
            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    'success': False,
                    'error': f'Timeout: выполнение превысило {timeout} секунд',
                    'output': None
                }
        
        finally:
            os.unlink(temp_file)
    
    def execute_with_dataframe(
        self, 
        code: str, 
        df: pd.DataFrame,
        timeout: int = None
    ) -> Dict[str, Any]:
        """Выполнение кода с передачей DataFrame"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_path = f.name
            df.to_csv(f.name, index=False)
        
        try:
            load_code = f"df = pd.read_csv(r'{df_path}')"
            full_code = f"{load_code}\n{code}"
            
            return self.execute(full_code, timeout)
        
        finally:
            os.unlink(df_path)
