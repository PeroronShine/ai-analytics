import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional

class AnalyticsEngine:
    """Ядро для аналитики данных"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.metadata = self._extract_metadata()
    
    def _extract_metadata(self) -> Dict:
        """Извлечение метаданных о датасете"""
        return {
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(self.df.select_dtypes(include=['datetime64']).columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'shape': self.df.shape
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики по датасету"""
        return {
            'describe': self.df.describe().to_dict(),
            'info': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'memory_usage': self.df.memory_usage(deep=True).sum() / 1024 ** 2,
                'missing_total': self.df.isnull().sum().sum()
            }
        }
    
    def create_visualization(
        self,
        chart_type: str,
        x: str = None,
        y: str = None,
        title: str = None,
        **kwargs
    ) -> Optional[go.Figure]:
        """Создание визуализации"""
        try:
            if chart_type == 'histogram':
                fig = px.histogram(self.df, x=x or self.metadata['numeric_columns'][0], 
                                   title=title or f'Распределение {x}')
            
            elif chart_type == 'scatter':
                fig = px.scatter(self.df, x=x, y=y, title=title or f'{x} vs {y}')
            
            elif chart_type == 'line':
                fig = px.line(self.df, x=x, y=y, title=title or f'Временной ряд {y}')
            
            elif chart_type == 'bar':
                fig = px.bar(self.df, x=x, y=y, title=title or f'{y} по {x}')
            
            elif chart_type == 'box':
                fig = px.box(self.df, y=y or self.metadata['numeric_columns'][0], 
                            title=title or 'Box plot')
            
            elif chart_type == 'heatmap':
                corr_matrix = self.df[self.metadata['numeric_columns']].corr()
                fig = px.imshow(corr_matrix, text_auto=True, 
                               title='Корреляционная матрица')
            else:
                return None
            
            fig.update_layout(
                template='plotly_dark',
                height=500
            )
            
            return fig
        
        except Exception as e:
            print(f"Ошибка при создании графика: {e}")
            return None
    
    def calculate_correlation(self) -> pd.DataFrame:
        """Расчет корреляционной матрицы"""
        numeric_df = self.df[self.metadata['numeric_columns']]
        return numeric_df.corr()
    
    def group_statistics(self, group_by: str, agg_columns: list = None) -> pd.DataFrame:
        """Группировка и агрегация данных"""
        if agg_columns is None:
            agg_columns = self.metadata['numeric_columns']
        
        return self.df.groupby(group_by)[agg_columns].agg(['mean', 'sum', 'count', 'std'])
