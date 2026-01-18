"""
Feature Engineering Module
Creates features for ML modeling based on cascade tracking results.

CRITICAL: Split-First Design for Production ML
==============================================
This class uses a strict fit()/transform() pattern to prevent data leakage:

1. fit(train_df): Learn statistics, encodings, and aggregates ONLY from training data
2. transform(df): Apply learned transformations to any dataframe (train/test/inference)

Why this is mandatory:
- Aggregation leakage: Geographic completion rates must be learned from training only
- Target leakage: Features that encode the target must be excluded
- Future information: Features requiring knowledge of outcomes are forbidden
- Encoder fitting: LabelEncoders must see all training categories, not test data
- Scaling: StandardScaler must compute mean/std from training only

This ensures:
- Proper train-test split (no test data influences training features)
- Valid cross-validation (each fold trains on independent feature stats)
- Production inference (features computed without future knowledge)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Engineers features for transition failure prediction.
    
    Uses fit()/transform() pattern to prevent data leakage:
    - fit(train_df): Learn from training data only
    - transform(df): Apply transformations to any data
    
    REMOVED FEATURES (due to leakage):
    - all_updates_completed: Direct target encoding
    - num_completed_updates: Proxy of target
    - completion_rate: Proxy of target (num_completed_updates / 4)
    - num_missing_updates: Inverse proxy of target
    - days_to_complete_all: Requires knowing all outcomes
    - early_completion/late_completion: Requires knowing full outcome
    - state/district/urban_rural_completion_rate: Aggregation leakage (now learned in fit())
    """
    
    def __init__(self):
        # Learned from training data (fit only)
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.geographic_aggregates: Dict[str, pd.Series] = {}  # State/district/urban_rural completion rates
        self.feature_names_: List[str] = []  # Stored feature order
        self.is_fitted: bool = False
        
        # Safe fallback values for unseen categories/regions
        self.DEFAULT_COMPLETION_RATE = 0.5  # Neutral prior
        self.UNSEEN_CATEGORY_VALUE = -1
    
    def fit(self, train_df: pd.DataFrame, target_col: str = 'transition_failure'):
        """
        Learn feature transformations from training data only.
        
        Args:
            train_df: Training DataFrame with cascade tracking results
            target_col: Name of target column (for leakage detection)
        """
        logger.info("Fitting FeatureEngineer on training data...")
        
        # Safety check: ensure target is present for leakage detection
        if target_col not in train_df.columns:
            logger.warning(f"Target column '{target_col}' not found. Cannot perform leakage checks.")
        
        # Create features without aggregation (to learn aggregates)
        feature_df = self._create_base_features(train_df.copy())
        
        # Learn geographic aggregates from training data only
        self._fit_geographic_aggregates(feature_df, target_col)
        
        # Apply geographic features
        feature_df = self._apply_geographic_features(feature_df)
        
        # Fit categorical encoders on training data
        self._fit_categorical_encoders(feature_df)
        feature_df = self._transform_categorical_features(feature_df)
        
        # Fit scaler on training data
        feature_list = self._get_feature_list_safe(feature_df, target_col)
        if feature_list:
            numeric_features = feature_df[feature_list].select_dtypes(include=[np.number]).columns.tolist()
            if numeric_features:
                self.scaler = StandardScaler()
                self.scaler.fit(feature_df[numeric_features])
        
        # Store feature names in order (critical for transform())
        self.feature_names_ = self._get_feature_list_safe(feature_df, target_col)
        
        # Leakage safety checks
        self._check_leakage(train_df, target_col)
        
        self.is_fitted = True
        logger.info(f"FeatureEngineer fitted. {len(self.feature_names_)} features created.")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned transformations to data (test/inference).
        
        Args:
            df: DataFrame to transform
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            ValueError: If fit() has not been called
        """
        if not self.is_fitted:
            raise ValueError(
                "FeatureEngineer has not been fitted. Call fit() before transform()."
            )
        
        logger.info(f"Transforming {len(df)} samples...")
        
        # Create base features (no aggregation)
        feature_df = self._create_base_features(df.copy())
        
        # Apply learned geographic features
        feature_df = self._apply_geographic_features(feature_df)
        
        # Apply learned categorical encodings
        feature_df = self._transform_categorical_features(feature_df)
        
        # Ensure all expected features exist
        feature_df = self._ensure_feature_parity(feature_df)
        
        # Apply scaling
        if self.scaler is not None and self.feature_names_:
            numeric_features = [f for f in self.feature_names_ 
                              if f in feature_df.columns and pd.api.types.is_numeric_dtype(feature_df[f])]
            if numeric_features:
                feature_df[numeric_features] = self.scaler.transform(feature_df[numeric_features])
        
        logger.info(f"Transformation complete. {len(self.feature_names_)} features created.")
        return feature_df
    
    def fit_transform(self, train_df: pd.DataFrame, target_col: str = 'transition_failure') -> pd.DataFrame:
        """
        Fit on training data and transform it.
        
        Args:
            train_df: Training DataFrame
            target_col: Name of target column
            
        Returns:
            Transformed training DataFrame
        """
        self.fit(train_df, target_col)
        return self.transform(train_df)
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names (in correct order).
        
        Returns:
            List of feature column names
            
        Raises:
            ValueError: If fit() has not been called
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer has not been fitted. Call fit() first.")
        return self.feature_names_.copy()
    
    def _create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create base features that don't require aggregation or target knowledge.
        These are safe to compute on any dataframe.
        """
        # 1. Time-to-update features
        df = self._create_time_to_update_features(df)
        
        # 2. Binary window flags (based on time-to-update, not outcomes)
        df = self._create_window_flags_safe(df)
        
        # 3. Gap features (time between updates)
        df = self._create_gap_features(df)
        
        # 4. Missing update indicators (based on presence of update dates)
        df = self._create_missing_indicators(df)
        
        # 5. Temporal features
        df = self._create_temporal_features(df)
        
        return df
    
    def _create_time_to_update_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-to-update features for each update type.
        Safe: Only uses days_after_18, no outcome knowledge required.
        """
        update_types = ['biometric', 'mobile', 'address', 'name']
        
        for update_type in update_types:
            days_col = f'{update_type}_days_after_18'
            
            if days_col in df.columns:
                # Time to update (fill missing with large value to indicate never updated)
                df[f'{update_type}_time_to_update'] = df[days_col].fillna(999)
                
                # Log transform (add 1 to avoid log(0))
                df[f'{update_type}_time_to_update_log'] = np.log1p(
                    df[f'{update_type}_time_to_update'].clip(lower=0)
                )
                
                # Is update late? (>90 days) - safe if computed from days_after_18
                df[f'{update_type}_is_late'] = (df[days_col] > 90).astype(int).fillna(1)
        
        return df
    
    def _create_window_flags_safe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binary flags for each update window completion.
        REMOVED: all_updates_completed, num_completed_updates (target leakage)
        
        Safe: Only individual window flags, no aggregate completion features.
        """
        update_types = ['biometric', 'mobile', 'address', 'name']
        
        for update_type in update_types:
            window_flag = f'{update_type}_in_window'
            if window_flag in df.columns:
                df[f'{update_type}_window_completed'] = df[window_flag].astype(int).fillna(0)
            else:
                # Fallback: use time-to-update to infer window completion
                time_col = f'{update_type}_time_to_update'
                if time_col in df.columns:
                    # Define windows
                    if update_type == 'biometric':
                        window_end = 0
                    elif update_type == 'mobile':
                        window_end = 30
                    elif update_type == 'address':
                        window_end = 60
                    else:  # name
                        window_end = 90
                    
                    df[f'{update_type}_window_completed'] = (
                        (df[time_col] <= window_end) & (df[time_col] < 999)
                    ).astype(int)
        
        # REMOVED: all_updates_completed (direct target encoding)
        # REMOVED: num_completed_updates (proxy of target)
        
        return df
    
    def _create_gap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create gap features: time between consecutive updates.
        Safe: Computed from time-to-update features, no outcome knowledge.
        """
        # Gap between biometric and mobile
        if 'biometric_time_to_update' in df.columns and 'mobile_time_to_update' in df.columns:
            df['gap_biometric_to_mobile'] = (
                df['mobile_time_to_update'] - df['biometric_time_to_update']
            ).fillna(0)
            # Handle missing updates (999 values)
            df['gap_biometric_to_mobile'] = df['gap_biometric_to_mobile'].clip(lower=-999, upper=999)
        
        # Gap between mobile and address
        if 'mobile_time_to_update' in df.columns and 'address_time_to_update' in df.columns:
            df['gap_mobile_to_address'] = (
                df['address_time_to_update'] - df['mobile_time_to_update']
            ).fillna(0)
            df['gap_mobile_to_address'] = df['gap_mobile_to_address'].clip(lower=-999, upper=999)
        
        # Gap between address and name
        if 'address_time_to_update' in df.columns and 'name_time_to_update' in df.columns:
            df['gap_address_to_name'] = (
                df['name_time_to_update'] - df['address_time_to_update']
            ).fillna(0)
            df['gap_address_to_name'] = df['gap_address_to_name'].clip(lower=-999, upper=999)
        
        # Average gap between updates (excluding NaN/missing)
        gap_cols = [col for col in df.columns if col.startswith('gap_')]
        if gap_cols:
            df['avg_gap_between_updates'] = df[gap_cols].replace([np.inf, -np.inf], np.nan).mean(axis=1).fillna(0)
            df['max_gap_between_updates'] = df[gap_cols].replace([np.inf, -np.inf], np.nan).max(axis=1).fillna(0)
        
        return df
    
    def _create_missing_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create indicators for missing updates.
        REMOVED: num_missing_updates (inverse proxy of target)
        
        Safe: Only individual missing indicators.
        """
        update_types = ['biometric', 'mobile', 'address', 'name']
        
        for update_type in update_types:
            updated_col = f'{update_type}_updated'
            time_col = f'{update_type}_time_to_update'
            
            if updated_col in df.columns:
                df[f'{update_type}_missing'] = (~df[updated_col]).astype(int).fillna(1)
            elif time_col in df.columns:
                # Infer from time-to-update (999 means never updated)
                df[f'{update_type}_missing'] = (df[time_col] >= 999).astype(int)
            else:
                df[f'{update_type}_missing'] = 1  # Assume missing if no data
        
        # REMOVED: num_missing_updates (inverse proxy of target: 4 - num_completed_updates)
        
        return df
    
    def _fit_geographic_aggregates(self, train_df: pd.DataFrame, target_col: str):
        """
        Learn geographic completion rates from training data only.
        These are stored and applied via map() in transform().
        """
        if target_col not in train_df.columns:
            logger.warning("Cannot learn geographic aggregates: target column missing")
            return
        
        # State-level completion rate
        if 'state' in train_df.columns:
            state_completion = train_df.groupby('state')[target_col].mean()
            self.geographic_aggregates['state'] = state_completion
            logger.info(f"Learned completion rates for {len(state_completion)} states")
        
        # District-level completion rate
        if 'district' in train_df.columns:
            district_completion = train_df.groupby('district')[target_col].mean()
            self.geographic_aggregates['district'] = district_completion
            logger.info(f"Learned completion rates for {len(district_completion)} districts")
        
        # Urban vs rural completion rate
        if 'urban_rural' in train_df.columns:
            urban_rural_completion = train_df.groupby('urban_rural')[target_col].mean()
            self.geographic_aggregates['urban_rural'] = urban_rural_completion
            logger.info(f"Learned completion rates for {len(urban_rural_completion)} urban_rural categories")
    
    def _apply_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned geographic aggregates via map() with safe fallbacks.
        Safe: Uses pre-computed aggregates, no new aggregation.
        """
        # State-level completion rate
        if 'state' in df.columns and 'state' in self.geographic_aggregates:
            df['state_completion_rate'] = df['state'].map(
                self.geographic_aggregates['state']
            ).fillna(self.DEFAULT_COMPLETION_RATE)
        
        # District-level completion rate
        if 'district' in df.columns and 'district' in self.geographic_aggregates:
            df['district_completion_rate'] = df['district'].map(
                self.geographic_aggregates['district']
            ).fillna(self.DEFAULT_COMPLETION_RATE)
        
        # Urban vs rural completion rate
        if 'urban_rural' in df.columns and 'urban_rural' in self.geographic_aggregates:
            df['urban_rural_completion_rate'] = df['urban_rural'].map(
                self.geographic_aggregates['urban_rural']
            ).fillna(self.DEFAULT_COMPLETION_RATE)
        
        return df
    
    def _fit_categorical_encoders(self, train_df: pd.DataFrame):
        """
        Fit LabelEncoders on training data only.
        """
        categorical_cols = ['gender', 'state', 'district', 'urban_rural']
        
        for col in categorical_cols:
            if col in train_df.columns:
                encoder = LabelEncoder()
                # Fit on all unique values in training data
                unique_values = train_df[col].astype(str).fillna('Unknown').unique()
                encoder.fit(unique_values)
                self.label_encoders[col] = encoder
                logger.info(f"Fitted encoder for '{col}' with {len(encoder.classes_)} categories")
    
    def _transform_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned categorical encodings.
        Unseen categories are mapped to -1.
        """
        categorical_cols = ['gender', 'state', 'district', 'urban_rural']
        
        for col in categorical_cols:
            if col not in df.columns:
                continue
            
            encoded_col = f'{col}_encoded'
            
            if col in self.label_encoders:
                encoder = self.label_encoders[col]
                # Convert to string and handle missing
                values = df[col].astype(str).fillna('Unknown')
                
                # Map known categories, set unseen to -1
                df[encoded_col] = values.apply(
                    lambda x: encoder.transform([x])[0] if x in encoder.classes_ else self.UNSEEN_CATEGORY_VALUE
                )
            else:
                # Encoder not fitted, skip this column
                logger.warning(f"Encoder for '{col}' not fitted. Skipping encoding.")
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from dates.
        Safe: Only uses date components, no outcome knowledge.
        """
        if 'eighteenth_birthday' in df.columns:
            # Year and month of 18th birthday
            df['eighteenth_birthday_year'] = df['eighteenth_birthday'].dt.year
            df['eighteenth_birthday_month'] = df['eighteenth_birthday'].dt.month
            df['eighteenth_birthday_quarter'] = df['eighteenth_birthday'].dt.quarter
            
            # Day of week
            df['eighteenth_birthday_dow'] = df['eighteenth_birthday'].dt.dayofweek
        
        if 'enrolment_date' in df.columns:
            # Years between enrollment and 18th birthday
            if 'eighteenth_birthday' in df.columns:
                df['years_between_enrolment_and_18'] = (
                    df['eighteenth_birthday'] - pd.to_datetime(df['enrolment_date'])
                ).dt.days / 365.25
        
        return df
    
    def _get_feature_list_safe(self, df: pd.DataFrame, target_col: str = 'transition_failure') -> List[str]:
        """
        Get list of feature columns, excluding target and identifiers.
        """
        exclude_cols = [
            'aadhaar_id', 'dob', 'enrolment_date', 'eighteenth_birthday',
            'biometric_update_date', 'mobile_update_date', 
            'address_update_date', 'name_update_date',
            'state', 'district', 'urban_rural',  # Use encoded versions
            'gender',  # Use encoded version
            target_col,  # Target variable
            # REMOVED LEAKAGE FEATURES:
            'all_updates_completed',  # Direct target encoding
            'num_completed_updates',  # Proxy of target
            'completion_rate',  # Proxy of target
            'num_missing_updates',  # Inverse proxy of target
            'days_to_complete_all',  # Requires knowing all outcomes
            'early_completion',  # Requires knowing full outcome
            'late_completion',  # Requires knowing full outcome
        ]
        
        # Exclude raw update columns (use derived features instead)
        exclude_patterns = [
            '_days_after_18',  # Use time_to_update features instead
            '_in_window',  # Use window_completed instead
            '_updated',  # Use missing indicators instead
        ]
        
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if any(pattern in col for pattern in exclude_patterns):
                continue
            # Prefer encoded versions over raw categoricals
            is_raw_categorical = col in ['state', 'district', 'gender', 'urban_rural']
            if is_raw_categorical and f'{col}_encoded' in df.columns:
                continue
            feature_cols.append(col)
        
        return sorted(feature_cols)
    
    def _ensure_feature_parity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all expected features exist (add missing with default values).
        Critical for feature order consistency between train/test.
        """
        for feature_name in self.feature_names_:
            if feature_name not in df.columns:
                # Add missing feature with safe default
                if 'encoded' in feature_name or 'completion_rate' in feature_name:
                    df[feature_name] = self.DEFAULT_COMPLETION_RATE
                else:
                    df[feature_name] = 0
                logger.warning(f"Missing feature '{feature_name}' added with default value")
        
        # Ensure feature order matches
        if self.feature_names_:
            # Reorder and select only expected features
            available_features = [f for f in self.feature_names_ if f in df.columns]
            if len(available_features) < len(self.feature_names_):
                missing = set(self.feature_names_) - set(available_features)
                logger.warning(f"Missing features: {missing}")
        
        return df
    
    def _check_leakage(self, train_df: pd.DataFrame, target_col: str):
        """
        Perform leakage safety checks.
        """
        if target_col not in train_df.columns:
            return
        
        # Get feature list
        feature_list = self._get_feature_list_safe(train_df, target_col)
        
        # Check 1: Target not in feature list
        if target_col in feature_list:
            raise ValueError(f"CRITICAL: Target column '{target_col}' found in feature list!")
        
        # Check 2: High correlation with target (potential leakage)
        if len(feature_list) > 0:
            feature_df = train_df[feature_list].select_dtypes(include=[np.number])
            target = train_df[target_col]
            
            for feature in feature_df.columns:
                if feature in train_df.columns:
                    corr = abs(train_df[feature].corr(target))
                    if corr > 0.95:
                        logger.warning(
                            f"WARNING: Feature '{feature}' has correlation {corr:.3f} with target. "
                            f"Potential leakage - consider removing."
                        )
