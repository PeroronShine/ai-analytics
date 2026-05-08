import subprocess
import json
import tempfile
import os
import sys
from typing import Dict, Any
import pandas as pd


class CodeInterpreter:
    """Безопасный интерпретатор Python кода"""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def execute(self, code: str, timeout: int = None) -> Dict[str, Any]:
        """Выполнение Python кода в изолированной среде"""
        timeout = timeout or self.timeout

        wrapper_template = '''
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from io import StringIO

old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()

agent_output_df = None
agent_output_fig = None

try:
{user_code}

    result = {
        "success": True,
        "output": mystdout.getvalue(),
        "error": None,
        "data": None,
        "plot": None
    }

    if agent_output_df is not None:
        try:
            result["data"] = agent_output_df.head(100).to_dict(orient="records")
        except Exception as e:
            result["output"] += "\n[Ошибка сериализации DataFrame: " + str(e) + "]"

    if agent_output_fig is not None:
        try:
            result["plot"] = agent_output_fig.to_json()
        except Exception as e:
            result["output"] += "\n[Ошибка сериализации Figure: " + str(e) + "]"

except Exception as e:
    result = {
        "success": False,
        "output": mystdout.getvalue(),
        "error": str(e),
        "data": None,
        "plot": None
    }

finally:
    sys.stdout = old_stdout

print("---RESULT_JSON_START---")
print(json.dumps(result, default=str))
print("---RESULT_JSON_END---")
'''

        indented_code = "\n".join("    " + line for line in code.split("\n"))
        wrapper_code = wrapper_template.replace("{user_code}", indented_code)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(wrapper_code)
            temp_file = f.name

        try:
            python_executable = sys.executable
            process = subprocess.Popen(
                [python_executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(timeout=timeout)

                if stderr and not stdout:
                    return {
                        "success": False,
                        "error": stderr,
                        "output": None,
                        "data": None,
                        "plot": None
                    }

                json_start = stdout.find("---RESULT_JSON_START---")
                json_end = stdout.find("---RESULT_JSON_END---")

                if json_start != -1 and json_end != -1:
                    json_str = stdout[json_start + len("---RESULT_JSON_START---"):json_end].strip()
                    result = json.loads(json_str)
                    pre_output = stdout[:json_start].strip()
                    if pre_output:
                        result["output"] = pre_output + "\n" + (result.get("output") or "")
                    return result
                else:
                    try:
                        result = json.loads(stdout.strip())
                        return result
                    except json.JSONDecodeError:
                        return {
                            "success": True,
                            "output": stdout,
                            "error": stderr if stderr else None,
                            "data": None,
                            "plot": None
                        }

            except subprocess.TimeoutExpired:
                process.kill()
                return {
                    "success": False,
                    "error": f"Timeout: выполнение превысило {timeout} секунд",
                    "output": None,
                    "data": None,
                    "plot": None
                }

        finally:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

    def execute_with_dataframe(
        self, 
        code: str, 
        df: pd.DataFrame,
        timeout: int = None
    ) -> Dict[str, Any]:
        """Выполнение кода с передачей DataFrame"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            df_path = f.name
            df.to_csv(f.name, index=False)

        try:
            load_code = f"df = pd.read_csv(r'{df_path}')"
            full_code = f"{load_code}\n{code}"

            return self.execute(full_code, timeout)

        finally:
            try:
                os.unlink(df_path)
            except Exception:
                pass
