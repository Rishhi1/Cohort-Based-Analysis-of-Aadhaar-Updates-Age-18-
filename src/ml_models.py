"""
ML Modeling Module
Trains models to predict transition failure and provides SHAP explainability
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import pickle
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    xgb = None
    logging.warning(f"XGBoost not available: {e}. Install with: pip install xgboost (may require libomp on macOS)")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Install with: pip install shap")

from config import (
    RANDOM_STATE, TEST_SIZE, CV_FOLDS, XGBOOST_PARAMS, RF_PARAMS,
    MODELS_DIR
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLModeler:
    """
    Trains and evaluates ML models for transition failure prediction.
    Supports Logistic Regression, Random Forest, and XGBoost.
    """
    
    def __init__(self):
        self.models = {}
        self.feature_names = []
        self.model_results = {}
        self.shap_explainers = {}
        
    def prepare_data(self, df: pd.DataFrame, 
                    target_col: str = 'transition_failure',
                    feature_list: Optional[List[str]] = None) -> Tuple:
        """
        Prepare data for modeling: extract features and target.
        
        Args:
            df: DataFrame with features
            target_col: Name of target column
            feature_list: List of feature columns to use
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        if feature_list is None:
            from feature_engineering import FeatureEngineer
            fe = FeatureEngineer()
            feature_list = fe.get_feature_list(df, exclude_target=True)
        
        # Extract target
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame")
        
        y = df[target_col].astype(int)
        
        # Extract features
        available_features = [f for f in feature_list if f in df.columns]
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())  # Fill with median for numeric
        X = X.fillna(0)  # Fill remaining with 0
        
        self.feature_names = available_features
        logger.info(f"Prepared data: {len(X)} samples, {len(available_features)} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, available_features
    
    def train_test_split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        Split data into train and test sets.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_logistic_regression(self, X_train: pd.DataFrame, 
                                  y_train: pd.Series,
                                  X_test: pd.DataFrame,
                                  y_test: pd.Series) -> Dict:
        """
        Train Logistic Regression model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        logger.info("Training Logistic Regression model...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        self.models['logistic_regression'] = model
        self.model_results['logistic_regression'] = metrics
        
        logger.info(f"Logistic Regression - AUC: {metrics['auc']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'pred_proba': y_pred_proba
        }
    
    def train_random_forest(self, X_train: pd.DataFrame,
                           y_train: pd.Series,
                           X_test: pd.DataFrame,
                           y_test: pd.Series) -> Dict:
        """
        Train Random Forest model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and evaluation metrics
        """
        logger.info("Training Random Forest model...")
        
        model = RandomForestClassifier(**RF_PARAMS)
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        metrics['feature_importance'] = feature_importance
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        self.models['random_forest'] = model
        self.model_results['random_forest'] = metrics
        
        logger.info(f"Random Forest - AUC: {metrics['auc']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'pred_proba': y_pred_proba
        }
    
    def train_xgboost(self, X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_test: pd.DataFrame,
                     y_test: pd.Series) -> Optional[Dict]:
        """
        Train XGBoost model.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with model and evaluation metrics, or None if XGBoost unavailable
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available, skipping...")
            return None
        
        logger.info("Training XGBoost model...")
        
        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        metrics = self._evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        metrics['feature_importance'] = feature_importance
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring='roc_auc'
        )
        metrics['cv_auc_mean'] = cv_scores.mean()
        metrics['cv_auc_std'] = cv_scores.std()
        
        self.models['xgboost'] = model
        self.model_results['xgboost'] = metrics
        
        logger.info(f"XGBoost - AUC: {metrics['auc']:.4f}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': y_pred,
            'pred_proba': y_pred_proba
        }
    
    def _evaluate_model(self, y_true: pd.Series, 
                       y_pred: np.ndarray,
                       y_pred_proba: np.ndarray) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'avg_precision': average_precision_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # ROC curve data
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'thresholds': thresholds.tolist()}
            
            # Precision-Recall curve data
            precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_curve'] = {
                'precision': precision_curve.tolist(),
                'recall': recall_curve.tolist(),
                'thresholds': pr_thresholds.tolist()
            }
        
        return metrics
    
    def compute_shap_values(self, model_name: str,
                           X_sample: pd.DataFrame,
                           max_samples: int = 100) -> Optional[np.ndarray]:
        """
        Compute SHAP values for model explainability.
        
        Args:
            model_name: Name of the model ('random_forest' or 'xgboost')
            X_sample: Feature matrix to explain
            max_samples: Maximum number of samples for SHAP computation
            
        Returns:
            SHAP values array, or None if SHAP unavailable
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explainability analysis")
            return None
        
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        
        # Sample data for faster computation
        if len(X_sample) > max_samples:
            X_sample = X_sample.sample(n=max_samples, random_state=RANDOM_STATE)
        
        logger.info(f"Computing SHAP values for {model_name} on {len(X_sample)} samples...")
        
        try:
            if model_name == 'xgboost' or (xgb is not None and isinstance(model, xgb.XGBClassifier)):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                self.shap_explainers[model_name] = explainer
            elif model_name == 'random_forest' or isinstance(model, RandomForestClassifier):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                self.shap_explainers[model_name] = explainer
            else:
                # For linear models, use KernelExplainer
                explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:50])
                shap_values = explainer.shap_values(X_sample.iloc[:50])
                self.shap_explainers[model_name] = explainer
            
            logger.info(f"SHAP values computed successfully for {model_name}")
            return shap_values
            
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None
    
    def save_model(self, model_name: str, filepath: Optional[Path] = None):
        """
        Save trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save model (default: MODELS_DIR/model_name.pkl)
        """
        if model_name not in self.models:
            logger.warning(f"Model '{model_name}' not found")
            return
        
        if filepath is None:
            filepath = MODELS_DIR / f"{model_name}.pkl"
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.models[model_name],
                'feature_names': self.feature_names,
                'metrics': self.model_results.get(model_name, {})
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: Optional[Path] = None):
        """
        Load trained model from disk.
        
        Args:
            model_name: Name of the model
            filepath: Path to load model from
        """
        if filepath is None:
            filepath = MODELS_DIR / f"{model_name}.pkl"
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filepath}")
            return
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.models[model_name] = data['model']
        self.feature_names = data.get('feature_names', [])
        self.model_results[model_name] = data.get('metrics', {})
        
        logger.info(f"Model loaded from {filepath}")
