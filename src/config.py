"""
Configuration module for 18th Birthday Cascade Analysis
Defines constants, paths, and hyperparameters for the ML pipeline
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT
ENROL_DIR = DATA_DIR / "enrol"
BIOMETRIC_DIR = DATA_DIR / "biometric"
DEMOGRAPHIC_DIR = DATA_DIR / "demographic"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODELS_DIR = OUTPUT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

FIGURES_DIR = OUTPUT_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

# 18th Birthday Cascade Window Definitions
# Day 0: biometric update (mandatory)
BIOMETRIC_WINDOW_DAYS = 0  # On the exact 18th birthday

# Day 0-30: mobile update (independence signal)
MOBILE_WINDOW_START = 0
MOBILE_WINDOW_END = 30

# Day 31-60: address update (migration signal)
ADDRESS_WINDOW_START = 31
ADDRESS_WINDOW_END = 60

# Day 61-90: name update (identity experimentation)
NAME_WINDOW_START = 61
NAME_WINDOW_END = 90

# Update types
UPDATE_TYPES = {
    "biometric": "biometric",
    "mobile": "mobile",
    "address": "address",
    "name": "name"
}

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# XGBoost parameters
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# Random Forest parameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

# Visualization settings
FIGURE_SIZE = (12, 8)
DPI = 300
FONT_SIZE = 12
PLOT_STYLE = "seaborn-v0_8-darkgrid"
