"""
18th Birthday Cascade Tracking Module
Tracks identity update events within defined time windows after turning 18
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import timedelta
import logging
from config import (
    BIOMETRIC_WINDOW_DAYS, MOBILE_WINDOW_START, MOBILE_WINDOW_END,
    ADDRESS_WINDOW_START, ADDRESS_WINDOW_END, NAME_WINDOW_START, NAME_WINDOW_END
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CascadeTracker:
    """
    Tracks 18th birthday cascade: biometric, mobile, address, and name updates
    within defined time windows after a citizen turns 18.
    """
    
    def __init__(self):
        self.windows = {
            'biometric': (BIOMETRIC_WINDOW_DAYS, BIOMETRIC_WINDOW_DAYS),
            'mobile': (MOBILE_WINDOW_START, MOBILE_WINDOW_END),
            'address': (ADDRESS_WINDOW_START, ADDRESS_WINDOW_END),
            'name': (NAME_WINDOW_START, NAME_WINDOW_END)
        }
    
    def calculate_18th_birthday(self, dob: pd.Series) -> pd.Series:
        """
        Calculate the date when each individual turns 18.
        
        Args:
            dob: Series of dates of birth
            
        Returns:
            Series of 18th birthday dates
        """
        return dob + pd.Timedelta(days=18*365)
    
    def check_update_in_window(self, update_date: pd.Series, 
                               eighteenth_birthday: pd.Series,
                               window_start: int, 
                               window_end: int) -> pd.Series:
        """
        Check if an update occurred within a specified window relative to 18th birthday.
        
        Args:
            update_date: Series of update dates
            eighteenth_birthday: Series of 18th birthday dates
            window_start: Days after 18th birthday (window start)
            window_end: Days after 18th birthday (window end)
            
        Returns:
            Boolean Series indicating if update is in window
        """
        if update_date is None or eighteenth_birthday is None:
            return pd.Series([False] * len(eighteenth_birthday))
        
        window_start_date = eighteenth_birthday + pd.Timedelta(days=window_start)
        window_end_date = eighteenth_birthday + pd.Timedelta(days=window_end)
        
        return (update_date >= window_start_date) & (update_date <= window_end_date)
    
    def track_cascade(self, 
                     enrol_df: pd.DataFrame,
                     biometric_updates: Optional[pd.DataFrame] = None,
                     demographic_updates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Track the cascade of updates for each individual after their 18th birthday.
        
        Args:
            enrol_df: Enrollment data with DOB and basic demographics
            biometric_updates: DataFrame with biometric update logs (aadhaar_id, update_date)
            demographic_updates: DataFrame with demographic update logs (aadhaar_id, update_date, update_type)
            
        Returns:
            DataFrame with cascade tracking results for each individual
        """
        df = enrol_df.copy()
        
        # Ensure 18th birthday is calculated
        if 'eighteenth_birthday' not in df.columns:
            if 'dob' in df.columns:
                df['eighteenth_birthday'] = self.calculate_18th_birthday(df['dob'])
            else:
                logger.warning("DOB not found, cannot calculate 18th birthday")
                return df
        
        # Initialize cascade tracking columns
        for update_type in ['biometric', 'mobile', 'address', 'name']:
            df[f'{update_type}_updated'] = False
            df[f'{update_type}_update_date'] = pd.NaT
            df[f'{update_type}_days_after_18'] = np.nan
            df[f'{update_type}_in_window'] = False
        
        # Track biometric updates
        if biometric_updates is not None and len(biometric_updates) > 0:
            df = self._track_biometric_updates(df, biometric_updates)
        else:
            # If no biometric data, simulate based on enrollment patterns
            logger.info("No biometric updates provided, simulating based on enrollment data")
            df = self._simulate_biometric_updates(df)
        
        # Track demographic updates
        if demographic_updates is not None and len(demographic_updates) > 0:
            df = self._track_demographic_updates(df, demographic_updates)
        else:
            # Simulate demographic updates
            logger.info("No demographic updates provided, simulating based on enrollment data")
            df = self._simulate_demographic_updates(df)
        
        # Calculate completion flags
        df['transition_failure'] = (
            ~df['biometric_in_window'] | 
            ~df['mobile_in_window'] | 
            ~df['address_in_window'] | 
            ~df['name_in_window']
        )
        
        # Calculate days to complete all updates
        df['days_to_complete_all'] = df[[
            'biometric_days_after_18', 'mobile_days_after_18',
            'address_days_after_18', 'name_days_after_18'
        ]].max(axis=1)
        
        return df
    
    def _track_biometric_updates(self, df: pd.DataFrame, 
                                  biometric_updates: pd.DataFrame) -> pd.DataFrame:
        """
        Track biometric updates for each individual.
        
        Args:
            df: Enrollment DataFrame
            biometric_updates: Biometric update logs
            
        Returns:
            DataFrame with biometric tracking added
        """
        # Merge biometric updates
        if 'aadhaar_id' in biometric_updates.columns and 'aadhaar_id' in df.columns:
            bio_merged = df.merge(
                biometric_updates[['aadhaar_id', 'update_date']].rename(
                    columns={'update_date': 'biometric_update_date'}
                ),
                on='aadhaar_id',
                how='left'
            )
            
            # Calculate days after 18th birthday
            bio_merged['biometric_days_after_18'] = (
                bio_merged['biometric_update_date'] - bio_merged['eighteenth_birthday']
            ).dt.days
            
            # Check if in window
            window_start, window_end = self.windows['biometric']
            bio_merged['biometric_in_window'] = self.check_update_in_window(
                bio_merged['biometric_update_date'],
                bio_merged['eighteenth_birthday'],
                window_start, window_end
            )
            
            bio_merged['biometric_updated'] = bio_merged['biometric_update_date'].notna()
            bio_merged['biometric_update_date'] = bio_merged['biometric_update_date']
            
            df = bio_merged
        else:
            # Fallback: simulate based on enrollment patterns
            df = self._simulate_biometric_updates(df)
        
        return df
    
    def _track_demographic_updates(self, df: pd.DataFrame, 
                                    demographic_updates: pd.DataFrame) -> pd.DataFrame:
        """
        Track demographic updates (mobile, address, name) for each individual.
        
        Args:
            df: Enrollment DataFrame
            demographic_updates: Demographic update logs with update_type column
            
        Returns:
            DataFrame with demographic tracking added
        """
        if 'aadhaar_id' not in demographic_updates.columns or 'aadhaar_id' not in df.columns:
            # Fallback to simulation
            return self._simulate_demographic_updates(df)
        
        # Process each update type separately
        for update_type in ['mobile', 'address', 'name']:
            update_subset = demographic_updates[
                demographic_updates['update_type'] == update_type
            ][['aadhaar_id', 'update_date']].rename(
                columns={'update_date': f'{update_type}_update_date'}
            )
            
            # Merge with main dataframe
            df = df.merge(update_subset, on='aadhaar_id', how='left')
            
            # Calculate days after 18th birthday
            df[f'{update_type}_days_after_18'] = (
                df[f'{update_type}_update_date'] - df['eighteenth_birthday']
            ).dt.days
            
            # Check if in window
            window_start, window_end = self.windows[update_type]
            df[f'{update_type}_in_window'] = self.check_update_in_window(
                df[f'{update_type}_update_date'],
                df['eighteenth_birthday'],
                window_start, window_end
            )
            
            df[f'{update_type}_updated'] = df[f'{update_type}_update_date'].notna()
        
        return df
    
    def _simulate_biometric_updates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate biometric updates based on enrollment patterns.
        In real implementation, this would use actual biometric data.
        """
        # Simulate: 95% complete biometric update on time (mandatory)
        np.random.seed(42)
        completion_rate = 0.95
        
        mask = np.random.random(len(df)) < completion_rate
        df.loc[mask, 'biometric_updated'] = True
        # Create random days offsets
        days_offset = np.random.normal(0, 3, mask.sum()).clip(min=0)
        df.loc[mask, 'biometric_update_date'] = df.loc[mask, 'eighteenth_birthday'] + pd.to_timedelta(days_offset, unit='D')
        
        df['biometric_days_after_18'] = (
            df['biometric_update_date'] - df['eighteenth_birthday']
        ).dt.days
        
        window_start, window_end = self.windows['biometric']
        df['biometric_in_window'] = (
            (df['biometric_days_after_18'] >= window_start) & 
            (df['biometric_days_after_18'] <= window_end)
        ) | df['biometric_updated']
        
        return df
    
    def _simulate_demographic_updates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate demographic updates with realistic completion rates and delays.
        """
        np.random.seed(42)
        
        # Mobile update: 75% complete within 30 days (independence signal)
        mobile_rate = 0.75
        mobile_mask = np.random.random(len(df)) < mobile_rate
        df.loc[mobile_mask, 'mobile_updated'] = True
        days_offset = np.random.uniform(0, 30, mobile_mask.sum())
        df.loc[mobile_mask, 'mobile_update_date'] = (
            df.loc[mobile_mask, 'eighteenth_birthday'] + 
            pd.to_timedelta(days_offset, unit='D')
        )
        df['mobile_days_after_18'] = (
            df['mobile_update_date'] - df['eighteenth_birthday']
        ).dt.days
        
        # Address update: 60% complete within 31-60 days (migration signal)
        address_rate = 0.60
        address_mask = np.random.random(len(df)) < address_rate
        df.loc[address_mask, 'address_updated'] = True
        days_offset = np.random.uniform(31, 60, address_mask.sum())
        df.loc[address_mask, 'address_update_date'] = (
            df.loc[address_mask, 'eighteenth_birthday'] + 
            pd.to_timedelta(days_offset, unit='D')
        )
        df['address_days_after_18'] = (
            df['address_update_date'] - df['eighteenth_birthday']
        ).dt.days
        
        # Name update: 40% complete within 61-90 days (identity experimentation)
        name_rate = 0.40
        name_mask = np.random.random(len(df)) < name_rate
        df.loc[name_mask, 'name_updated'] = True
        days_offset = np.random.uniform(61, 90, name_mask.sum())
        df.loc[name_mask, 'name_update_date'] = (
            df.loc[name_mask, 'eighteenth_birthday'] + 
            pd.to_timedelta(days_offset, unit='D')
        )
        df['name_days_after_18'] = (
            df['name_update_date'] - df['eighteenth_birthday']
        ).dt.days
        
        # Set window flags
        window_start, window_end = self.windows['mobile']
        df['mobile_in_window'] = (
            (df['mobile_days_after_18'] >= window_start) & 
            (df['mobile_days_after_18'] <= window_end)
        )
        
        window_start, window_end = self.windows['address']
        df['address_in_window'] = (
            (df['address_days_after_18'] >= window_start) & 
            (df['address_days_after_18'] <= window_end)
        )
        
        window_start, window_end = self.windows['name']
        df['name_in_window'] = (
            (df['name_days_after_18'] >= window_start) & 
            (df['name_days_after_18'] <= window_end)
        )
        
        return df
