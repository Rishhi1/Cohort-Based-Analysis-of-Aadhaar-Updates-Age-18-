"""
Statistical Analysis Module
Computes completion rates, divergence metrics, and identifies high-risk cohorts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on cascade completion data.
    Computes completion rates, gender divergence, rural-urban lag, and high-risk cohorts.
    """
    
    def __init__(self):
        self.update_types = ['biometric', 'mobile', 'address', 'name']
        self.window_days = {
            'biometric': (0, 0),
            'mobile': (0, 30),
            'address': (31, 60),
            'name': (61, 90)
        }
    
    def compute_completion_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute completion rates for each update type within each window.
        
        Args:
            df: DataFrame with cascade tracking results
            
        Returns:
            DataFrame with completion rates by update type
        """
        results = []
        
        for update_type in self.update_types:
            window_col = f'{update_type}_in_window'
            
            if window_col in df.columns:
                total = len(df)
                completed = df[window_col].sum()
                completion_rate = (completed / total * 100) if total > 0 else 0
                
                results.append({
                    'update_type': update_type,
                    'window_days': f"{self.window_days[update_type][0]}-{self.window_days[update_type][1]}",
                    'total_individuals': total,
                    'completed': completed,
                    'completion_rate_pct': completion_rate,
                    'missing': total - completed,
                    'missing_rate_pct': 100 - completion_rate
                })
        
        completion_df = pd.DataFrame(results)
        logger.info(f"Computed completion rates for {len(completion_df)} update types")
        
        return completion_df
    
    def compute_cohort_completion_rates(self, 
                                       df: pd.DataFrame,
                                       cohort_col: str) -> pd.DataFrame:
        """
        Compute completion rates broken down by cohort (e.g., gender, urban_rural, state).
        
        Args:
            df: DataFrame with cascade tracking results
            cohort_col: Column name to group by (e.g., 'gender', 'urban_rural', 'state')
            
        Returns:
            DataFrame with completion rates by cohort and update type
        """
        results = []
        
        if cohort_col not in df.columns:
            logger.warning(f"Cohort column '{cohort_col}' not found in DataFrame")
            return pd.DataFrame()
        
        for cohort_value in df[cohort_col].dropna().unique():
            cohort_df = df[df[cohort_col] == cohort_value]
            
            for update_type in self.update_types:
                window_col = f'{update_type}_in_window'
                
                if window_col in cohort_df.columns:
                    total = len(cohort_df)
                    completed = cohort_df[window_col].sum()
                    completion_rate = (completed / total * 100) if total > 0 else 0
                    
                    results.append({
                        'cohort': cohort_value,
                        'update_type': update_type,
                        'total': total,
                        'completed': completed,
                        'completion_rate_pct': completion_rate,
                        'missing': total - completed
                    })
        
        cohort_completion_df = pd.DataFrame(results)
        
        if len(cohort_completion_df) > 0:
            logger.info(f"Computed completion rates for {len(cohort_completion_df)} cohort-update combinations")
        
        return cohort_completion_df
    
    def compute_gender_divergence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute gender-based divergence metrics for each update type.
        
        Args:
            df: DataFrame with cascade tracking results
            
        Returns:
            DataFrame with gender divergence metrics
        """
        if 'gender' not in df.columns:
            logger.warning("Gender column not found")
            return pd.DataFrame()
        
        results = []
        
        for update_type in self.update_types:
            window_col = f'{update_type}_in_window'
            
            if window_col not in df.columns:
                continue
            
            # Completion rates by gender
            gender_rates = df.groupby('gender')[window_col].agg(['sum', 'count', 'mean'])
            gender_rates['completion_rate'] = gender_rates['mean'] * 100
            
            # Extract rates for each gender
            male_rate = gender_rates.loc['M', 'completion_rate'] if 'M' in gender_rates.index else 0
            female_rate = gender_rates.loc['F', 'completion_rate'] if 'F' in gender_rates.index else 0
            
            # Divergence metric (absolute difference)
            divergence = abs(male_rate - female_rate)
            
            # Relative divergence (percentage point difference)
            relative_divergence = male_rate - female_rate
            
            results.append({
                'update_type': update_type,
                'male_completion_rate_pct': male_rate,
                'female_completion_rate_pct': female_rate,
                'divergence_pct_points': divergence,
                'relative_divergence': relative_divergence,
                'male_count': gender_rates.loc['M', 'count'] if 'M' in gender_rates.index else 0,
                'female_count': gender_rates.loc['F', 'count'] if 'F' in gender_rates.index else 0
            })
        
        gender_divergence_df = pd.DataFrame(results)
        logger.info(f"Computed gender divergence for {len(gender_divergence_df)} update types")
        
        return gender_divergence_df
    
    def compute_rural_urban_lag(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute rural vs urban lag in completion rates.
        
        Args:
            df: DataFrame with cascade tracking results
            
        Returns:
            DataFrame with rural-urban lag metrics
        """
        if 'urban_rural' not in df.columns:
            logger.warning("urban_rural column not found")
            return pd.DataFrame()
        
        results = []
        
        for update_type in self.update_types:
            window_col = f'{update_type}_in_window'
            days_col = f'{update_type}_days_after_18'
            
            if window_col not in df.columns:
                continue
            
            # Completion rates
            urban_rural_rates = df.groupby('urban_rural')[window_col].mean() * 100
            
            urban_rate = urban_rural_rates.get('Urban', 0)
            rural_rate = urban_rural_rates.get('Rural', 0)
            
            # Average days to update (for completed updates only)
            lag_data = []
            if days_col in df.columns:
                for area_type in ['Urban', 'Rural']:
                    area_df = df[(df['urban_rural'] == area_type) & df[window_col]]
                    if len(area_df) > 0:
                        avg_days = area_df[days_col].mean()
                        lag_data.append({
                            'area_type': area_type,
                            'avg_days_to_update': avg_days
                        })
            
            # Lag metric (rural - urban, positive means rural is slower)
            completion_lag = rural_rate - urban_rate  # Negative means rural is behind
            avg_lag_days = 0
            if lag_data:
                lag_df = pd.DataFrame(lag_data)
                if len(lag_df) == 2:
                    rural_days = lag_df[lag_df['area_type'] == 'Rural']['avg_days_to_update'].values[0]
                    urban_days = lag_df[lag_df['area_type'] == 'Urban']['avg_days_to_update'].values[0]
                    avg_lag_days = rural_days - urban_days
            
            results.append({
                'update_type': update_type,
                'urban_completion_rate_pct': urban_rate,
                'rural_completion_rate_pct': rural_rate,
                'completion_lag_pct_points': completion_lag,
                'rural_avg_days_to_update': lag_data[1]['avg_days_to_update'] if len(lag_data) > 1 else np.nan,
                'urban_avg_days_to_update': lag_data[0]['avg_days_to_update'] if len(lag_data) > 0 else np.nan,
                'lag_days': avg_lag_days
            })
        
        lag_df = pd.DataFrame(results)
        logger.info(f"Computed rural-urban lag for {len(lag_df)} update types")
        
        return lag_df
    
    def identify_high_risk_cohorts(self, df: pd.DataFrame,
                                   min_size: int = 100) -> pd.DataFrame:
        """
        Identify high-risk cohorts with low completion rates.
        
        Args:
            df: DataFrame with cascade tracking results
            min_size: Minimum cohort size to consider
            
        Returns:
            DataFrame with high-risk cohorts
        """
        high_risk_cohorts = []
        
        # Check by state
        if 'state' in df.columns and 'transition_failure' in df.columns:
            state_risk = df.groupby('state').agg({
                'transition_failure': ['mean', 'count']
            }).reset_index()
            state_risk.columns = ['state', 'failure_rate', 'count']
            state_risk = state_risk[state_risk['count'] >= min_size]
            state_risk = state_risk.nlargest(10, 'failure_rate')
            state_risk['cohort_type'] = 'state'
            high_risk_cohorts.append(state_risk)
        
        # Check by district
        if 'district' in df.columns and 'transition_failure' in df.columns:
            district_risk = df.groupby('district').agg({
                'transition_failure': ['mean', 'count']
            }).reset_index()
            district_risk.columns = ['district', 'failure_rate', 'count']
            district_risk = district_risk[district_risk['count'] >= min_size]
            district_risk = district_risk.nlargest(20, 'failure_rate')
            district_risk['cohort_type'] = 'district'
            high_risk_cohorts.append(district_risk)
        
        # Check by gender
        if 'gender' in df.columns and 'transition_failure' in df.columns:
            gender_risk = df.groupby('gender').agg({
                'transition_failure': ['mean', 'count']
            }).reset_index()
            gender_risk.columns = ['gender', 'failure_rate', 'count']
            gender_risk['cohort_type'] = 'gender'
            high_risk_cohorts.append(gender_risk)
        
        # Check by urban_rural
        if 'urban_rural' in df.columns and 'transition_failure' in df.columns:
            urban_rural_risk = df.groupby('urban_rural').agg({
                'transition_failure': ['mean', 'count']
            }).reset_index()
            urban_rural_risk.columns = ['urban_rural', 'failure_rate', 'count']
            urban_rural_risk['cohort_type'] = 'urban_rural'
            high_risk_cohorts.append(urban_rural_risk)
        
        if high_risk_cohorts:
            high_risk_df = pd.concat(high_risk_cohorts, ignore_index=True)
            high_risk_df = high_risk_df.sort_values('failure_rate', ascending=False)
            logger.info(f"Identified {len(high_risk_df)} high-risk cohorts")
            return high_risk_df
        
        return pd.DataFrame()
    
    def compute_statistical_tests(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform statistical significance tests between cohorts.
        
        Args:
            df: DataFrame with cascade tracking results
            
        Returns:
            DataFrame with test results
        """
        test_results = []
        
        # Gender comparison
        if 'gender' in df.columns and 'transition_failure' in df.columns:
            male_failures = df[df['gender'] == 'M']['transition_failure'].sum()
            male_total = len(df[df['gender'] == 'M'])
            female_failures = df[df['gender'] == 'F']['transition_failure'].sum()
            female_total = len(df[df['gender'] == 'F'])
            
            if male_total > 0 and female_total > 0:
                contingency = [[male_failures, male_total - male_failures],
                              [female_failures, female_total - female_failures]]
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                
                test_results.append({
                    'test': 'gender_failure_chi2',
                    'statistic': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'cohort1': f'Male (n={male_total})',
                    'cohort2': f'Female (n={female_total})'
                })
        
        # Urban vs Rural comparison
        if 'urban_rural' in df.columns and 'transition_failure' in df.columns:
            urban_failures = df[df['urban_rural'] == 'Urban']['transition_failure'].sum()
            urban_total = len(df[df['urban_rural'] == 'Urban'])
            rural_failures = df[df['urban_rural'] == 'Rural']['transition_failure'].sum()
            rural_total = len(df[df['urban_rural'] == 'Rural'])
            
            if urban_total > 0 and rural_total > 0:
                contingency = [[urban_failures, urban_total - urban_failures],
                              [rural_failures, rural_total - rural_failures]]
                chi2, p_value = stats.chi2_contingency(contingency)[:2]
                
                test_results.append({
                    'test': 'urban_rural_failure_chi2',
                    'statistic': chi2,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'cohort1': f'Urban (n={urban_total})',
                    'cohort2': f'Rural (n={rural_total})'
                })
        
        test_results_df = pd.DataFrame(test_results)
        logger.info(f"Computed {len(test_results_df)} statistical tests")
        
        return test_results_df
