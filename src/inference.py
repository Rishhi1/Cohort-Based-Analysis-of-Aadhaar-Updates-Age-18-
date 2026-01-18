"""
Inference Script for Day-0 Predictor
Command-line tool for making predictions on new data
"""

import pandas as pd
import numpy as np
import pickle
import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config import MODELS_DIR
from feature_engineering import FeatureEngineer
from day0_predictor import Day0Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_models():
    """Load trained models."""
    # Load feature engineer
    with open(MODELS_DIR / 'feature_engineer.pkl', 'rb') as f:
        feature_engineer = pickle.load(f)
    
    # Load Day-0 predictor
    model_path = MODELS_DIR / 'day0_predictor_xgboost.pkl'
    if not model_path.exists():
        model_path = MODELS_DIR / 'day0_predictor_random_forest.pkl'
    
    with open(model_path, 'rb') as f:
        day0_data = pickle.load(f)
    
    day0_predictor = Day0Predictor()
    day0_predictor.models['primary'] = day0_data['model']
    day0_predictor.day0_feature_names = day0_data['feature_names']
    day0_predictor.is_fitted = True
    
    return feature_engineer, day0_predictor


def predict_single(input_data: dict):
    """Predict for a single citizen."""
    feature_engineer, day0_predictor = load_models()
    
    # Create DataFrame
    input_df = pd.DataFrame([{
        'aadhaar_id': input_data.get('aadhaar_id', 'unknown'),
        'gender': input_data.get('gender'),
        'state': input_data.get('state'),
        'district': input_data.get('district'),
        'urban_rural': input_data.get('urban_rural'),
        'enrolment_date': input_data.get('enrolment_date'),
        'eighteenth_birthday': pd.to_datetime(input_data.get('eighteenth_birthday')),
        'pincode': input_data.get('pincode'),
        'dob': pd.to_datetime(input_data.get('eighteenth_birthday')) - pd.Timedelta(days=18*365)
    }])
    
    # Feature engineering
    features_df = feature_engineer.transform(input_df)
    
    # Get Day-0 features
    day0_features = day0_predictor.get_day0_features(features_df, features_df.columns.tolist())
    X_pred = features_df[day0_features].fillna(0)
    
    # Predict
    model = day0_predictor.models['primary']
    probability = model.predict_proba(X_pred)[0][1]
    prediction = 1 if probability >= 0.5 else 0
    
    # Risk level
    if probability >= 0.7:
        risk_level = "high"
    elif probability >= 0.5:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    return {
        'aadhaar_id': input_data.get('aadhaar_id', 'unknown'),
        'prediction': int(prediction),
        'predicted_class': 'failure' if prediction == 1 else 'success',
        'probability': float(probability),
        'risk_level': risk_level
    }


def predict_batch(input_file: str, output_file: str):
    """Predict for multiple citizens from CSV file."""
    # Load data
    df = pd.read_csv(input_file)
    
    # Predict for each row
    results = []
    for idx, row in df.iterrows():
        try:
            input_data = row.to_dict()
            result = predict_single(input_data)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            results.append({
                'aadhaar_id': row.get('aadhaar_id', f'row_{idx}'),
                'error': str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    logger.info(f"Saved predictions to {output_file}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Day-0 Predictor Inference')
    parser.add_argument('--input', type=str, help='Input JSON file or CSV file')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--single', action='store_true', help='Single prediction mode')
    
    args = parser.parse_args()
    
    if args.single and args.input:
        # Single prediction from JSON
        with open(args.input, 'r') as f:
            input_data = json.load(f)
        
        result = predict_single(input_data)
        print(json.dumps(result, indent=2))
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
    
    elif args.input:
        # Batch prediction from CSV
        if not args.output:
            args.output = args.input.replace('.csv', '_predictions.csv')
        
        predict_batch(args.input, args.output)
    
    else:
        # Interactive mode
        print("Day-0 Predictor Inference")
        print("Enter citizen details:")
        
        input_data = {
            'gender': input("Gender (M/F): "),
            'state': input("State: "),
            'district': input("District: "),
            'urban_rural': input("Urban/Rural: "),
            'eighteenth_birthday': input("18th Birthday (YYYY-MM-DD): "),
            'enrolment_date': input("Enrolment Date (YYYY-MM-DD, optional): "),
            'pincode': input("Pincode (optional): ")
        }
        
        result = predict_single(input_data)
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
