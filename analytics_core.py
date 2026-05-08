import json
import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

@dataclass
class ChartSpec:
    type: str
    title: str
    data: Dict[str, Any]
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    width: int = 800
    height: int = 500

class AnalyticsCore:
    SUPPORTED_CHARTS = ['bar', 'line', 'scatter', 'histogram', 'pie', 'heatmap', 'box', 'violin']
    MAX_CHARTS_PER_ANALYSIS = 5

    def __init__(self):
        self.charts_generated: List[ChartSpec] = []

    def process_agent_result(self, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        steps = agent_result.get('steps', [])
        final_answer = agent_result.get('final_answer', '')

        code_snippets = []
        observations = []

        for step in steps:
            if not step.is_final:
                for tc in step.tool_calls:
                    if tc.name == 'execute_python':
                        code_snippets.append(tc.arguments.get('code', ''))
                if step.observation:
                    observations.append(step.observation)

        charts = self._extract_charts_from_steps(steps)

        report = {
            "summary": self._generate_summary(final_answer, observations),
            "key_findings": self._extract_findings(final_answer),
            "code_executed": code_snippets,
            "observations": observations,
            "charts": charts,
            "statistics": {
                "steps_count": len(steps),
                "code_executions": len(code_snippets),
                "charts_generated": len(charts)
            }
        }

        return report

    def _extract_charts_from_steps(self, steps: List[Any]) -> List[Dict[str, Any]]:
        charts = []
        return charts

    def _generate_summary(self, final_answer: str, observations: List[str]) -> str:
        if not final_answer:
            return "Анализ не завершён."
        summary = final_answer[:500]
        if len(final_answer) > 500:
            summary += "..."
        return summary

    def _extract_findings(self, text: str) -> List[str]:
        findings = []
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '* ', '• ', '1. ', '2. ', '3. ', '4. ', '5. ')):
                findings.append(line[2:].strip())
            elif any(kw in line.lower() for kw in ['вывод', 'находка', 'результат', 'ключевой', 'важно', 'замечание']):
                if len(line) > 20:
                    findings.append(line)
        return findings[:10]

    def create_plotly_chart(self, spec: ChartSpec) -> go.Figure:
        df = spec.data if isinstance(spec.data, pd.DataFrame) else pd.DataFrame(spec.data)

        if spec.type == 'bar':
            fig = px.bar(df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
                        title=spec.title, width=spec.width, height=spec.height)
        elif spec.type == 'line':
            fig = px.line(df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
                         title=spec.title, width=spec.width, height=spec.height)
        elif spec.type == 'scatter':
            fig = px.scatter(df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
                           title=spec.title, width=spec.width, height=spec.height)
        elif spec.type == 'histogram':
            fig = px.histogram(df, x=spec.x_column, color=spec.color_column,
                             title=spec.title, width=spec.width, height=spec.height)
        elif spec.type == 'pie':
            fig = px.pie(df, values=spec.y_column, names=spec.x_column,
                        title=spec.title, width=spec.width, height=spec.height)
        elif spec.type == 'heatmap':
            fig = px.imshow(df, title=spec.title, width=spec.width, height=spec.height)
        elif spec.type == 'box':
            fig = px.box(df, x=spec.x_column, y=spec.y_column, color=spec.color_column,
                        title=spec.title, width=spec.width, height=spec.height)
        else:
            fig = go.Figure()
            fig.add_annotation(text=f"Unsupported chart type: {spec.type}", showarrow=False)

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,30,46,0.8)',
            font_color='#FAFAFA',
            title_font_color='#FAFAFA',
            legend_font_color='#FAFAFA'
        )

        return fig

    def validate_chart_request(self, chart_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        valid = []
        for spec in chart_specs:
            if spec.get('type') in self.SUPPORTED_CHARTS:
                valid.append(spec)
            if len(valid) >= self.MAX_CHARTS_PER_ANALYSIS:
                break
        return valid

    def format_number(self, value: Any, precision: int = 2) -> str:
        if isinstance(value, (int, float)):
            if abs(value) >= 1e6:
                return f"{value:.{precision}e}"
            return f"{value:.{precision}f}"
        return str(value)
