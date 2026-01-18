# 18th Birthday Cascade Analysis: Detecting Identity Transition Failures in Aadhaar

## Project Overview

This project analyzes the 90-day behavioral cascade after citizens turn 18 to detect demographic, geographic, and gender-based divergence in identity update completion rates. The analysis predicts transition failures that lead to future KYC rejections and DBT issues.

## Problem Statement

When citizens turn 18, Aadhaar mandates a biometric update. However, downstream updates (mobile number, address, name) happen inconsistently and often fail, causing KYC rejections and DBT issues years later.

## Core Analysis Logic

The pipeline tracks identity updates in defined time windows:
- **Day 0**: Biometric update (mandatory)
- **Day 0–30**: Mobile update (independence signal)
- **Day 31–60**: Address update (migration signal)
- **Day 61–90**: Name update (identity experimentation)

## Project Structure

```
UIDIA HAckathon/
├── src/
│   ├── config.py                 # Configuration and constants
│   ├── data_loader.py            # Data loading and preprocessing
│   ├── cascade_tracker.py        # 18th birthday cascade tracking
│   ├── feature_engineering.py    # Feature creation for ML
│   ├── statistical_analysis.py   # Statistical insights
│   ├── ml_models.py              # ML model training and evaluation
│   ├── visualizations.py         # Plotting and visualization
│   └── main_pipeline.py          # Main execution script
├── enrol/                        # Enrollment datasets
├── biometric/                    # Biometric update logs
├── demographic/                  # Demographic update logs
├── outputs/
│   ├── models/                   # Saved ML models
│   ├── figures/                  # Generated plots
│   ├── reports/                  # Policy summary reports
│   └── *.csv                     # Analysis results
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

1. Install Python 3.8 or higher
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the complete pipeline:

```bash
cd src
python main_pipeline.py
```

The pipeline will:
1. Load and preprocess enrollment, biometric, and demographic data
2. Track 18th birthday cascade for each individual
3. Engineer features for ML modeling
4. Perform statistical analysis (completion rates, gender divergence, rural-urban lag)
5. Train ML models (Logistic Regression, Random Forest, XGBoost)
6. Generate visualizations and policy-ready reports

## Outputs

All outputs are saved to the `outputs/` directory:

- **Figures**: Completion rates, cohort analysis, gender divergence, rural-urban lag, district heatmaps, feature importance, ROC curves
- **Reports**: Policy summary report with key findings and recommendations
- **CSV Files**: Tracked cascade results, completion rates, divergence metrics, high-risk cohorts
- **Models**: Trained ML models (saved as pickle files)

## Key Features

### Statistical Analysis
- Completion rates by update type and time window
- Gender-based divergence metrics
- Rural-urban lag analysis
- High-risk cohort identification
- Statistical significance testing

### ML Modeling
- Multiple models: Logistic Regression, Random Forest, XGBoost
- Cross-validation and performance metrics
- SHAP-based explainability
- Feature importance analysis

### Visualizations
- Completion rate bar charts
- Cohort comparison plots
- Gender divergence analysis
- Rural-urban lag visualization
- High-risk district heatmaps
- ROC curves and model comparison
- Feature importance plots

## Methodology

### Feature Engineering
- Time-to-update features
- Binary window completion flags
- Gap days between updates
- Missing update indicators
- Geographic aggregation features
- Temporal features

### Models
- **Logistic Regression**: Baseline model with balanced class weights
- **Random Forest**: Ensemble method with feature importance
- **XGBoost**: Gradient boosting with regularization

### Evaluation Metrics
- AUC-ROC
- Accuracy, Precision, Recall, F1-Score
- Average Precision
- Cross-validation scores

## Policy Implications

The analysis provides actionable insights for:
1. **Targeted Outreach**: Focus on high-risk districts and rural areas
2. **Gender-Sensitive Interventions**: Address gender-based divergence
3. **Early Warning System**: Deploy predictive models for proactive intervention
4. **Infrastructure Improvements**: Enhance digital infrastructure in underserved areas
5. **Monitoring Framework**: Track intervention effectiveness

## Technical Details

- **Language**: Python 3.8+
- **Key Libraries**: Pandas, NumPy, Scikit-learn, XGBoost, SHAP, Matplotlib, Seaborn
- **Data Handling**: Supports both individual-level and aggregated data formats
- **Reproducibility**: Random seed fixed for consistent results

## Author

Machine Learning Engineer + Data Scientist  
Project for UIDAI-style population-scale dataset analysis

## License

This project is for analytical and research purposes.
