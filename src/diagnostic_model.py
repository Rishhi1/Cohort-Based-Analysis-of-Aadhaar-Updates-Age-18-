"""
Retrospective Diagnostic Model
Explains why transition failures occurred after the 90-day cascade completes.
Uses ALL features (including post-event) for root cause analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

try:
    import xgboost as xgb
    import shap
    XGBOOST_AVAILABLE = True
    SHAP_AVAILABLE = True
except:
    XGBOOST_AVAILABLE = False
    SHAP_AVAILABLE = False
    xgb = None
    shap = None

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiagnosticModel:
    """
    Retrospective diagnostic model using ALL features (including post-event).
    
    Purpose:
    - Explain WHY failures happened
    - Identify root causes (gender, geography, temporal patterns)
    - Generate policy-actionable insights
    
    Features Included:
    - All Day-0 features (demographics, temporal, geographic)
    - Post-event features (time-to-update, gaps, completion flags)
    - Missing/late indicators
    """
    
    def __init__(self):
        self.model = None
        self.shap_explainer = None
        self.feature_names = []
        self.is_fitted = False
        
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_test: pd.DataFrame, y_test: pd.Series,
              feature_list: List[str]) -> Dict:
        """
        Train diagnostic model with ALL features.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            feature_list: List of all features (including post-event)
            
        Returns:
            Dictionary with model results and explanations
        """
        logger.info("="*80)
        logger.info("Training Retrospective Diagnostic Model (All Features for Explanation)")
        logger.info("="*80)
        
        # Use ALL features (including post-event)
        available_features = [f for f in feature_list if f in X_train.columns]
        X_train_diag = X_train[available_features].copy()
        X_test_diag = X_test[available_features].copy()
        
        # Handle missing values
        X_train_diag = X_train_diag.fillna(X_train_diag.median()).fillna(0)
        X_test_diag = X_test_diag.fillna(X_test_diag.median()).fillna(0)
        
        self.feature_names = available_features
        
        logger.info(f"Training with {len(available_features)} features (including post-event)")
        logger.info(f"Training samples: {len(X_train_diag)}, Test samples: {len(X_test_diag)}")
        
        # Train XGBoost (best for explainability)
        if XGBOOST_AVAILABLE:
            logger.info("\nTraining XGBoost for diagnostic analysis...")
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(
                X_train_diag, y_train,
                eval_set=[(X_test_diag, y_test)],
                verbose=False
            )
            
            self.model = model
            
        else:
            # Fallback to Random Forest
            logger.info("\nTraining Random Forest for diagnostic analysis...")
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train_diag, y_train)
            self.model = model
        
        # Predictions
        y_pred = self.model.predict(X_test_diag)
        y_pred_proba = self.model.predict_proba(X_test_diag)[:, 1]
        
        # Evaluate
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            metrics['feature_importance'] = feature_importance
        
        # SHAP explainability
        if SHAP_AVAILABLE and XGBOOST_AVAILABLE:
            logger.info("\nComputing SHAP values for explainability...")
            try:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test_diag.iloc[:100])  # Sample for speed
                self.shap_explainer = explainer
                metrics['shap_available'] = True
                logger.info("SHAP values computed successfully")
            except Exception as e:
                logger.warning(f"SHAP computation failed: {e}")
                metrics['shap_available'] = False
        else:
            metrics['shap_available'] = False
        
        self.is_fitted = True
        
        logger.info("\n" + "="*80)
        logger.info("Diagnostic Model Performance Summary")
        logger.info("="*80)
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info("\nNote: High accuracy acceptable for diagnostic model (uses post-event features)")
        
        return {
            'model': self.model,
            'metrics': metrics,
            'predictions': y_pred,
            'pred_proba': y_pred_proba
        }
    
    def explain(self, X_sample: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Generate SHAP explanations for a sample of instances.
        
        Args:
            X_sample: Sample instances to explain
            
        Returns:
            SHAP values array, or None if SHAP unavailable
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.warning("SHAP explainer not available")
            return None
        
        try:
            shap_values = self.shap_explainer.shap_values(X_sample)
            return shap_values
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def get_feature_names(self) -> List[str]:
        """Get feature names used by diagnostic model."""
        return self.feature_names.copy()
