"""
Day-0 Prediction Model
Predicts transition failure risk when citizen turns 18, using ONLY features available at prediction time.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve,
    average_precision_score, brier_score_loss, roc_curve
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    xgb = None

from config import RANDOM_STATE, CV_FOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Day0Predictor:
    """
    Day-0 prediction model using ONLY features available when citizen turns 18.
    
    Features Excluded (require future knowledge):
    - All *_time_to_update features
    - All *_window_completed flags
    - All gap features
    - All missing/late indicators
    
    Features Included (Day-0 available):
    - Demographics (gender, state, district, urban_rural)
    - Temporal (birthday components, enrollment context)
    - Geographic aggregates (learned from training)
    """
    
    # Explicit exclusion patterns for Day-0 model
    DAY0_EXCLUDED_PATTERNS = [
        '*_time_to_update*',
        '*_window_completed',
        'gap_*',
        '*_missing',
        '*_is_late',
        '*_days_after_18',
        '*_updated',
        'days_to_complete_all'
    ]
    
    def __init__(self):
        self.models = {}
        self.day0_feature_names = []
        self.is_fitted = False
        
    def get_day0_features(self, df: pd.DataFrame, feature_list: List[str]) -> List[str]:
        """
        Filter features to only those available at Day-0.
        
        Args:
            df: DataFrame with features
            feature_list: List of all features
            
        Returns:
            List of Day-0 available features
        """
        day0_features = []
        
        # Allowed patterns (Day-0 available)
        allowed_patterns = [
            'gender_encoded',
            'state_encoded',
            'district_encoded',
            'urban_rural_encoded',
            'eighteenth_birthday_*',
            'years_between_enrolment_and_18',
            '*_completion_rate',  # Geographic aggregates (learned from training)
            'pincode'
        ]
        
        for feature in feature_list:
            # Check if explicitly excluded
            excluded = any(
                pattern.replace('*', '') in feature or feature.startswith(pattern.replace('*', ''))
                for pattern in self.DAY0_EXCLUDED_PATTERNS
            )
            
            # Check if allowed
            allowed = any(
                pattern.replace('*', '') in feature or feature.startswith(pattern.replace('*', ''))
                for pattern in allowed_patterns
            )
            
            # Must be allowed AND not excluded AND exist in dataframe
            if allowed and not excluded and feature in df.columns:
                day0_features.append(feature)
        
        logger.info(f"Day-0 features: {len(day0_features)} out of {len(feature_list)} total features")
        logger.info(f"Day-0 feature list: {day0_features}")
        
        self.day0_feature_names = sorted(day0_features)
        return self.day0_feature_names
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series,
              feature_list: List[str]) -> Dict:
        """
        Train Day-0 prediction models.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_list: Full feature list (will be filtered to Day-0)
            
        Returns:
            Dictionary with model results
        """
        logger.info("="*80)
        logger.info("Training Day-0 Prediction Model (Features Available at Prediction Time)")
        logger.info("="*80)
        
        # Filter to Day-0 features only
        day0_features = self.get_day0_features(X_train, feature_list)
        
        X_train_day0 = X_train[day0_features].copy()
        X_test_day0 = X_test[day0_features].copy()
        
        # Handle missing values
        X_train_day0 = X_train_day0.fillna(X_train_day0.median()).fillna(0)
        X_test_day0 = X_test_day0.fillna(X_test_day0.median()).fillna(0)
        
        logger.info(f"Training with {len(day0_features)} Day-0 features")
        logger.info(f"Training samples: {len(X_train_day0)}, Test samples: {len(X_test_day0)}")
        
        results = {}
        
        # Train XGBoost (primary model)
        if XGBOOST_AVAILABLE:
            logger.info("\nTraining XGBoost (Day-0)...")
            model_xgb = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            
            model_xgb.fit(
                X_train_day0, y_train,
                eval_set=[(X_test_day0, y_test)],
                verbose=False
            )
            
            # Calibrate probabilities
            calibrated_xgb = CalibratedClassifierCV(model_xgb, method='isotonic', cv=3)
            calibrated_xgb.fit(X_train_day0, y_train)
            
            # Predictions
            y_pred = calibrated_xgb.predict(X_test_day0)
            y_pred_proba = calibrated_xgb.predict_proba(X_test_day0)[:, 1]
            
            # Evaluate
            metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, 'XGBoost_Day0')
            
            results['xgboost'] = {
                'model': calibrated_xgb,
                'metrics': metrics,
                'predictions': y_pred,
                'pred_proba': y_pred_proba
            }
            
            self.models['xgboost'] = calibrated_xgb
        
        # Train Random Forest
        logger.info("\nTraining Random Forest (Day-0)...")
        model_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        model_rf.fit(X_train_day0, y_train)
        
        # Calibrate
        calibrated_rf = CalibratedClassifierCV(model_rf, method='isotonic', cv=3)
        calibrated_rf.fit(X_train_day0, y_train)
        
        y_pred = calibrated_rf.predict(X_test_day0)
        y_pred_proba = calibrated_rf.predict_proba(X_test_day0)[:, 1]
        
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, 'RandomForest_Day0')
        
        results['random_forest'] = {
            'model': calibrated_rf,
            'metrics': metrics,
            'predictions': y_pred,
            'pred_proba': y_pred_proba
        }
        
        self.models['random_forest'] = calibrated_rf
        
        # Train Logistic Regression (baseline)
        logger.info("\nTraining Logistic Regression (Day-0)...")
        model_lr = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        
        model_lr.fit(X_train_day0, y_train)
        
        # Calibrate
        calibrated_lr = CalibratedClassifierCV(model_lr, method='isotonic', cv=3)
        calibrated_lr.fit(X_train_day0, y_train)
        
        y_pred = calibrated_lr.predict(X_test_day0)
        y_pred_proba = calibrated_lr.predict_proba(X_test_day0)[:, 1]
        
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba, 'LogisticRegression_Day0')
        
        results['logistic_regression'] = {
            'model': calibrated_lr,
            'metrics': metrics,
            'predictions': y_pred,
            'pred_proba': y_pred_proba
        }
        
        self.models['logistic_regression'] = calibrated_lr
        
        self.is_fitted = True
        
        # Print summary
        logger.info("\n" + "="*80)
        logger.info("Day-0 Model Performance Summary")
        logger.info("="*80)
        for model_name, result in results.items():
            m = result['metrics']
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  ROC-AUC: {m['roc_auc']:.4f}")
            logger.info(f"  Accuracy: {m['accuracy']:.4f}")
            logger.info(f"  Precision: {m['precision']:.4f}")
            logger.info(f"  Recall: {m['recall']:.4f}")
            logger.info(f"  F1-Score: {m['f1']:.4f}")
            logger.info(f"  Brier Score: {m['brier_score']:.4f} (lower is better)")
            logger.info(f"  Average Precision: {m['avg_precision']:.4f}")
        
        return results
    
    def _evaluate_model(self, y_true: pd.Series, y_pred: np.ndarray,
                       y_pred_proba: np.ndarray, model_name: str) -> Dict:
        """
        Evaluate model performance.
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'avg_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'brier_score': brier_score_loss(y_true, y_pred_proba),
        }
        
        # Recall at high precision
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        high_prec_idx = np.where(precision >= 0.8)[0]
        if len(high_prec_idx) > 0:
            metrics['recall_at_80pct_precision'] = recall[high_prec_idx[0]]
        else:
            metrics['recall_at_80pct_precision'] = 0.0
        
        return metrics
    
    def get_feature_names(self) -> List[str]:
        """Get Day-0 feature names."""
        if not self.day0_feature_names:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.day0_feature_names.copy()
