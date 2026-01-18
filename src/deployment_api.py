"""
Deployment API for Day-0 Prediction Model
Production-ready REST API for real-time failure risk prediction
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from config import MODELS_DIR, RANDOM_STATE
from feature_engineering import FeatureEngineer
from day0_predictor import Day0Predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)  # Enable CORS for cross-origin requests
else:
    app = None

# Global model and feature engineer (loaded at startup)
day0_predictor = None
feature_engineer = None
model_metadata = {}


def load_deployed_models():
    """
    Load trained models and feature engineer at startup.
    """
    global day0_predictor, feature_engineer, model_metadata
    
    try:
        # Load feature engineer
        fe_path = MODELS_DIR / 'feature_engineer.pkl'
        if fe_path.exists():
            with open(fe_path, 'rb') as f:
                feature_engineer = pickle.load(f)
            logger.info(f"Loaded feature engineer from {fe_path}")
        else:
            logger.warning(f"Feature engineer not found at {fe_path}. Train models first.")
            feature_engineer = FeatureEngineer()
        
        # Load Day-0 predictor (best model)
        day0_predictor = Day0Predictor()
        model_path = MODELS_DIR / 'day0_predictor_xgboost.pkl'
        
        if model_path.exists():
            with open(model_path, 'rb') as f:
                day0_data = pickle.load(f)
                day0_predictor.models['xgboost'] = day0_data['model']
                day0_predictor.day0_feature_names = day0_data['feature_names']
                day0_predictor.is_fitted = True
                model_metadata = day0_data.get('metadata', {})
            logger.info(f"Loaded Day-0 predictor from {model_path}")
        else:
            logger.warning(f"Day-0 predictor not found at {model_path}. Train models first.")
    
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


if FLASK_AVAILABLE and app is not None:
    @app.route('/health', methods=['GET'])
    def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': day0_predictor is not None and day0_predictor.is_fitted,
        'feature_engineer_loaded': feature_engineer is not None and feature_engineer.is_fitted
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict transition failure risk for a citizen turning 18.
    
    Expected input JSON:
    {
        "aadhaar_id": "sim_12345",
        "gender": "M",
        "state": "Karnataka",
        "district": "Bengaluru Urban",
        "urban_rural": "Urban",
        "enrolment_date": "2010-05-15",
        "eighteenth_birthday": "2028-05-15",
        "pincode": 560043
    }
    
    Returns:
    {
        "aadhaar_id": "sim_12345",
        "prediction": 1,
        "predicted_class": "failure",
        "probability": 0.75,
        "risk_level": "high",
        "features_used": [...],
        "timestamp": "..."
    }
    """
    try:
        if not day0_predictor or not day0_predictor.is_fitted:
            return jsonify({
                'error': 'Model not loaded. Train models first.'
            }), 503
        
        if not feature_engineer or not feature_engineer.is_fitted:
            return jsonify({
                'error': 'Feature engineer not loaded. Train models first.'
            }), 503
        
        # Parse input
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Validate required fields
        required_fields = ['gender', 'state', 'district', 'urban_rural', 'eighteenth_birthday']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}'
            }), 400
        
        # Create DataFrame from input
        input_df = pd.DataFrame([{
            'aadhaar_id': data.get('aadhaar_id', 'unknown'),
            'gender': data.get('gender'),
            'state': data.get('state'),
            'district': data.get('district'),
            'urban_rural': data.get('urban_rural'),
            'enrolment_date': data.get('enrolment_date'),
            'eighteenth_birthday': pd.to_datetime(data.get('eighteenth_birthday')),
            'pincode': data.get('pincode'),
            'dob': pd.to_datetime(data.get('eighteenth_birthday')) - pd.Timedelta(days=18*365)
        }])
        
        # Feature engineering
        try:
            features_df = feature_engineer.transform(input_df)
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            return jsonify({
                'error': f'Feature engineering failed: {str(e)}'
            }), 400
        
        # Get Day-0 features
        day0_features = day0_predictor.get_day0_features(features_df, features_df.columns.tolist())
        X_pred = features_df[day0_features].fillna(0)
        
        # Predict
        model = day0_predictor.models.get('xgboost') or day0_predictor.models.get('random_forest')
        if not model:
            return jsonify({'error': 'No trained model available'}), 503
        
        probability = model.predict_proba(X_pred)[0][1]
        prediction = 1 if probability >= 0.5 else 0
        
        # Risk level mapping
        if probability >= 0.7:
            risk_level = "high"
        elif probability >= 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        result = {
            'aadhaar_id': data.get('aadhaar_id', 'unknown'),
            'prediction': int(prediction),
            'predicted_class': 'failure' if prediction == 1 else 'success',
            'probability': float(probability),
            'risk_level': risk_level,
            'features_used': day0_features,
            'num_features': len(day0_features),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction for {data.get('aadhaar_id', 'unknown')}: {prediction} (prob: {probability:.3f})")
        
        return jsonify(result), 200
    
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({
            'error': f'Prediction failed: {str(e)}'
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple citizens at once.
    
    Expected input JSON:
    {
        "citizens": [
            {
                "aadhaar_id": "sim_12345",
                "gender": "M",
                "state": "Karnataka",
                ...
            },
            ...
        ]
    }
    """
    try:
        if not day0_predictor or not day0_predictor.is_fitted:
            return jsonify({'error': 'Model not loaded'}), 503
        
        data = request.get_json()
        if not data or 'citizens' not in data:
            return jsonify({'error': 'Expected JSON with "citizens" array'}), 400
        
        citizens = data['citizens']
        if not isinstance(citizens, list):
            return jsonify({'error': '"citizens" must be an array'}), 400
        
        # Process batch
        results = []
        for citizen in citizens:
            try:
                # Create single prediction request
                temp_request = type('obj', (object,), {'get_json': lambda: citizen})()
                original_request = request
                request = temp_request
                
                # Call predict internally
                response, status = predict()
                if status == 200:
                    results.append(json.loads(response.data))
                
                request = original_request
            except Exception as e:
                logger.error(f"Error processing citizen {citizen.get('aadhaar_id', 'unknown')}: {e}")
                results.append({
                    'aadhaar_id': citizen.get('aadhaar_id', 'unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'total': len(citizens),
            'processed': len([r for r in results if 'error' not in r]),
            'failed': len([r for r in results if 'error' in r]),
            'predictions': results,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information and metadata."""
    if not day0_predictor or not day0_predictor.is_fitted:
        return jsonify({'error': 'Model not loaded'}), 503
    
    return jsonify({
        'model_type': 'Day-0 Predictor',
        'purpose': 'Predict transition failure risk when citizen turns 18',
        'features': day0_predictor.get_feature_names(),
        'num_features': len(day0_predictor.get_feature_names()),
        'models_available': list(day0_predictor.models.keys()),
        'metadata': model_metadata,
        'timestamp': datetime.now().isoformat()
    }), 200


if __name__ == '__main__':
    if not FLASK_AVAILABLE:
        logger.error("Flask not available. Install with: pip install flask flask-cors")
        sys.exit(1)
    
    # Load models at startup
    logger.info("Loading models for deployment...")
    load_deployed_models()
    
    # Run Flask app
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting deployment API on {host}:{port}")
    app.run(host=host, port=port, debug=False)
