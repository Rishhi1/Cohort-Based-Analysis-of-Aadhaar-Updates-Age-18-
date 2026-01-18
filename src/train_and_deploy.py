"""
Train Day-0 Predictor and prepare for deployment
This script trains production-ready models and saves them for deployment
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR, RANDOM_STATE, TEST_SIZE, OUTPUT_DIR
from data_loader import DataLoader
from cascade_tracker import CascadeTracker
from feature_engineering import FeatureEngineer
from day0_predictor import Day0Predictor
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_deployment_models():
    """
    Train Day-0 predictor models and save for deployment.
    """
    logger.info("="*80)
    logger.info("Training Day-0 Predictor for Deployment")
    logger.info("="*80)
    
    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    data_loader = DataLoader()
    
    enrol_df = data_loader.load_enrolment_data()
    biometric_df = data_loader.load_biometric_data()
    demo_df = data_loader.load_demographic_data()
    
    enrol_df = data_loader.preprocess_enrolment(enrol_df)
    
    # Create individual-level simulation
    if 'aadhaar_id' not in enrol_df.columns:
        individual_df = data_loader.create_individual_level_simulation(
            enrol_df, biometric_df, demo_df
        )
    else:
        individual_df = enrol_df.copy()
    
    logger.info(f"Loaded {len(individual_df)} individual records")
    
    # Step 2: Track cascade
    logger.info("\nStep 2: Tracking 18th birthday cascade...")
    cascade_tracker = CascadeTracker()
    tracked_df = cascade_tracker.track_cascade(individual_df)
    logger.info(f"Tracked cascade for {len(tracked_df)} individuals")
    
    # Step 3: Feature engineering (split-first)
    logger.info("\nStep 3: Feature engineering (split-first to prevent leakage)...")
    
    # Split before feature engineering
    train_idx, test_idx = train_test_split(
        tracked_df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=tracked_df['transition_failure'] if 'transition_failure' in tracked_df.columns else None
    )
    
    train_df = tracked_df.loc[train_idx].copy()
    test_df = tracked_df.loc[test_idx].copy()
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(train_df, target_col='transition_failure')
    
    train_features_df = feature_engineer.transform(train_df)
    test_features_df = feature_engineer.transform(test_df)
    
    feature_list = feature_engineer.get_feature_names()
    logger.info(f"Created {len(feature_list)} features")
    
    # Step 4: Prepare data for Day-0 predictor
    logger.info("\nStep 4: Preparing data for Day-0 predictor...")
    
    X_train = train_features_df[feature_list].copy()
    y_train = train_df['transition_failure'].astype(int)
    X_test = test_features_df[feature_list].copy()
    y_test = test_df['transition_failure'].astype(int)
    
    X_train = X_train.fillna(X_train.median()).fillna(0)
    X_test = X_test.fillna(X_test.median()).fillna(0)
    
    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    # Step 5: Train Day-0 predictor
    logger.info("\nStep 5: Training Day-0 predictor (Day-0 features only)...")
    
    day0_predictor = Day0Predictor()
    results = day0_predictor.train(X_train, y_train, X_test, y_test, feature_list)
    
    # Step 6: Save models for deployment
    logger.info("\nStep 6: Saving models for deployment...")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save feature engineer
    with open(MODELS_DIR / 'feature_engineer.pkl', 'wb') as f:
        pickle.dump(feature_engineer, f)
    logger.info(f"Saved feature engineer to {MODELS_DIR / 'feature_engineer.pkl'}")
    
    # Save Day-0 predictor (best model: XGBoost or Random Forest)
    best_model_name = 'xgboost' if 'xgboost' in day0_predictor.models else 'random_forest'
    best_model = day0_predictor.models[best_model_name]
    
    day0_predictor_data = {
        'model': best_model,
        'feature_names': day0_predictor.get_feature_names(),
        'day0_feature_names': day0_predictor.get_feature_names(),
        'metadata': {
            'model_type': f'Day-0 Predictor ({best_model_name})',
            'training_date': pd.Timestamp.now().isoformat(),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'performance': results[best_model_name]['metrics'],
            'day0_features': day0_predictor.get_feature_names()
        }
    }
    
    model_path = MODELS_DIR / f'day0_predictor_{best_model_name}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(day0_predictor_data, f)
    logger.info(f"Saved Day-0 predictor to {model_path}")
    
    # Save all models (for ensemble or fallback)
    with open(MODELS_DIR / 'day0_predictor_all.pkl', 'wb') as f:
        pickle.dump(day0_predictor, f)
    logger.info(f"Saved all Day-0 predictor models to {MODELS_DIR / 'day0_predictor_all.pkl'}")
    
    # Step 7: Performance summary
    logger.info("\n" + "="*80)
    logger.info("Deployment Model Performance Summary")
    logger.info("="*80)
    
    best_metrics = results[best_model_name]['metrics']
    logger.info(f"\nBest Model: {best_model_name.upper()}")
    logger.info(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
    logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {best_metrics['f1']:.4f}")
    logger.info(f"  Brier Score: {best_metrics['brier_score']:.4f}")
    
    logger.info(f"\nDay-0 Features ({len(day0_predictor.get_feature_names())}):")
    for feat in day0_predictor.get_feature_names():
        logger.info(f"  - {feat}")
    
    logger.info("\n" + "="*80)
    logger.info("Models saved and ready for deployment!")
    logger.info(f"Feature Engineer: {MODELS_DIR / 'feature_engineer.pkl'}")
    logger.info(f"Day-0 Predictor: {model_path}")
    logger.info("="*80)
    
    return feature_engineer, day0_predictor, results


if __name__ == '__main__':
    train_deployment_models()
