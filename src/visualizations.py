"""
Visualization Module
Creates plots, tables, and heatmaps for the cascade analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from config import FIGURES_DIR, FIGURE_SIZE, DPI, PLOT_STYLE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
try:
    plt.style.use(PLOT_STYLE)
except:
    try:
        plt.style.use('seaborn-v0_8')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
sns.set_palette("husl")


class Visualizer:
    """
    Creates visualizations for cascade analysis results.
    """
    
    def __init__(self, output_dir: Path = FIGURES_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_completion_rates(self, completion_df: pd.DataFrame,
                             title: str = "Update Completion Rates by Window") -> Path:
        """
        Plot completion rates for each update type.
        
        Args:
            completion_df: DataFrame with completion rates
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        x = np.arange(len(completion_df))
        width = 0.35
        
        completed = ax.bar(x - width/2, completion_df['completion_rate_pct'], 
                          width, label='Completed', color='#2ecc71')
        missing = ax.bar(x + width/2, completion_df['missing_rate_pct'],
                        width, label='Missing', color='#e74c3c')
        
        ax.set_xlabel('Update Type', fontsize=12)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(completion_df['update_type'], rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Add value labels
        for i, (comp, miss) in enumerate(zip(completed, missing)):
            height_comp = comp.get_height()
            height_miss = miss.get_height()
            ax.text(comp.get_x() + comp.get_width()/2., height_comp,
                   f'{height_comp:.1f}%', ha='center', va='bottom', fontsize=10)
            ax.text(miss.get_x() + miss.get_width()/2., height_miss,
                   f'{height_miss:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        filepath = self.output_dir / "completion_rates.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved completion rates plot to {filepath}")
        return filepath
    
    def plot_cohort_completion(self, cohort_df: pd.DataFrame,
                              cohort_type: str = 'gender',
                              title: Optional[str] = None) -> Path:
        """
        Plot completion rates by cohort.
        
        Args:
            cohort_df: DataFrame with cohort completion rates
            cohort_type: Type of cohort (e.g., 'gender', 'urban_rural')
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if title is None:
            title = f"Completion Rates by {cohort_type.title()}"
        
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        pivot_df = cohort_df.pivot(index='update_type', columns='cohort', values='completion_rate_pct')
        
        pivot_df.plot(kind='bar', ax=ax, width=0.8)
        
        ax.set_xlabel('Update Type', fontsize=12)
        ax.set_ylabel('Completion Rate (%)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(title=cohort_type.title(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"cohort_completion_{cohort_type}.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved cohort completion plot to {filepath}")
        return filepath
    
    def plot_gender_divergence(self, gender_df: pd.DataFrame) -> Path:
        """
        Plot gender divergence metrics.
        
        Args:
            gender_df: DataFrame with gender divergence data
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Completion rates by gender
        x = np.arange(len(gender_df))
        width = 0.35
        
        ax1.bar(x - width/2, gender_df['male_completion_rate_pct'], 
               width, label='Male', color='#3498db')
        ax1.bar(x + width/2, gender_df['female_completion_rate_pct'],
               width, label='Female', color='#e91e63')
        
        ax1.set_xlabel('Update Type', fontsize=12)
        ax1.set_ylabel('Completion Rate (%)', fontsize=12)
        ax1.set_title('Completion Rates by Gender', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(gender_df['update_type'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Divergence metric
        colors = ['#e74c3c' if d > 0 else '#2ecc71' for d in gender_df['relative_divergence']]
        ax2.barh(x, gender_df['divergence_pct_points'], color=colors)
        ax2.set_yticks(x)
        ax2.set_yticklabels(gender_df['update_type'])
        ax2.set_xlabel('Divergence (Percentage Points)', fontsize=12)
        ax2.set_title('Gender Divergence', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / "gender_divergence.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved gender divergence plot to {filepath}")
        return filepath
    
    def plot_rural_urban_lag(self, lag_df: pd.DataFrame) -> Path:
        """
        Plot rural-urban lag metrics.
        
        Args:
            lag_df: DataFrame with rural-urban lag data
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Completion rates
        x = np.arange(len(lag_df))
        width = 0.35
        
        ax1.bar(x - width/2, lag_df['urban_completion_rate_pct'],
               width, label='Urban', color='#3498db')
        ax1.bar(x + width/2, lag_df['rural_completion_rate_pct'],
               width, label='Rural', color='#e67e22')
        
        ax1.set_xlabel('Update Type', fontsize=12)
        ax1.set_ylabel('Completion Rate (%)', fontsize=12)
        ax1.set_title('Completion Rates: Urban vs Rural', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(lag_df['update_type'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Lag days
        colors = ['#e74c3c' if lag > 0 else '#2ecc71' for lag in lag_df['lag_days']]
        ax2.barh(x, lag_df['lag_days'], color=colors)
        ax2.set_yticks(x)
        ax2.set_yticklabels(lag_df['update_type'])
        ax2.set_xlabel('Lag Days (Rural - Urban)', fontsize=12)
        ax2.set_title('Temporal Lag Between Urban and Rural', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / "rural_urban_lag.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved rural-urban lag plot to {filepath}")
        return filepath
    
    def plot_high_risk_districts_heatmap(self, df: pd.DataFrame,
                                        state_col: str = 'state',
                                        district_col: str = 'district',
                                        risk_col: str = 'transition_failure') -> Path:
        """
        Create heatmap of high-risk districts.
        
        Args:
            df: DataFrame with district-level risk data
            state_col: State column name
            district_col: District column name
            risk_col: Risk metric column name
            
        Returns:
            Path to saved figure
        """
        # Calculate district-level failure rates
        district_risk = df.groupby([state_col, district_col])[risk_col].agg(['mean', 'count']).reset_index()
        district_risk.columns = [state_col, district_col, 'failure_rate', 'count']
        
        # Filter districts with sufficient samples
        district_risk = district_risk[district_risk['count'] >= 100]
        
        # Get top 20 high-risk districts
        top_districts = district_risk.nlargest(20, 'failure_rate')
        
        # Create pivot table for heatmap
        pivot_df = top_districts.pivot_table(
            values='failure_rate',
            index=district_col,
            columns=state_col,
            aggfunc='mean'
        ).fillna(0)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='YlOrRd',
                   cbar_kws={'label': 'Failure Rate'}, ax=ax)
        
        ax.set_title('High-Risk Districts Heatmap (Top 20)', fontsize=14, fontweight='bold')
        ax.set_xlabel('State', fontsize=12)
        ax.set_ylabel('District', fontsize=12)
        
        plt.tight_layout()
        
        filepath = self.output_dir / "district_risk_heatmap.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved district heatmap to {filepath}")
        return filepath
    
    def plot_feature_importance(self, feature_importance: pd.DataFrame,
                               model_name: str = 'model',
                               top_n: int = 20) -> Path:
        """
        Plot feature importance.
        
        Args:
            feature_importance: DataFrame with feature names and importance scores
            model_name: Name of the model
            top_n: Number of top features to show
            
        Returns:
            Path to saved figure
        """
        top_features = feature_importance.head(top_n).sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.barh(range(len(top_features)), top_features['importance'], color='#3498db')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance - {model_name.title()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f"feature_importance_{model_name}.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance plot to {filepath}")
        return filepath
    
    def plot_roc_curves(self, model_results: Dict[str, Dict],
                       title: str = "ROC Curves Comparison") -> Path:
        """
        Plot ROC curves for all models.
        
        Args:
            model_results: Dictionary with model results (contains 'roc_curve' data)
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        
        for i, (model_name, results) in enumerate(model_results.items()):
            if 'roc_curve' in results:
                roc_data = results['roc_curve']
                ax.plot(roc_data['fpr'], roc_data['tpr'],
                       label=f"{model_name} (AUC = {results['auc']:.3f})",
                       color=colors[i % len(colors)], linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        filepath = self.output_dir / "roc_curves.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved ROC curves plot to {filepath}")
        return filepath
    
    def plot_model_comparison(self, model_results: Dict[str, Dict]) -> Path:
        """
        Create bar chart comparing model performance metrics.
        
        Args:
            model_results: Dictionary with model results
            
        Returns:
            Path to saved figure
        """
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        model_names = list(model_results.keys())
        metric_values = {metric: [model_results[m].get(metric, 0) for m in model_names]
                        for metric in metrics_to_plot}
        
        x = np.arange(len(model_names))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot)/2) * width + width/2
            ax.bar(x + offset, metric_values[metric], width, label=metric.upper())
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        filepath = self.output_dir / "model_comparison.png"
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved model comparison plot to {filepath}")
        return filepath
