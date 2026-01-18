"""
Main Pipeline Script
End-to-end execution of the 18th Birthday Cascade Analysis
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, REPORTS_DIR
from data_loader import DataLoader
from cascade_tracker import CascadeTracker
from feature_engineering import FeatureEngineer
from statistical_analysis import StatisticalAnalyzer
from ml_models import MLModeler
from visualizations import Visualizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_DIR / 'pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Main pipeline execution function.
    """
    logger.info("=" * 80)
    logger.info("18th Birthday Cascade Analysis Pipeline - Starting")
    logger.info("=" * 80)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Data Loading
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Loading Data")
        logger.info("="*80)
        
        data_loader = DataLoader()
        
        enrol_df = data_loader.load_enrolment_data()
        biometric_df = data_loader.load_biometric_data()
        demo_df = data_loader.load_demographic_data()
        
        # Preprocess enrollment data
        enrol_df = data_loader.preprocess_enrolment(enrol_df)
        
        # Create individual-level simulation if needed
        if 'aadhaar_id' not in enrol_df.columns:
            logger.info("Creating individual-level simulation from aggregated data...")
            individual_df = data_loader.create_individual_level_simulation(
                enrol_df, biometric_df, demo_df
            )
        else:
            individual_df = enrol_df.copy()
        
        logger.info(f"Loaded {len(individual_df)} individual records")
        
        # Step 2: Cascade Tracking
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Tracking 18th Birthday Cascade")
        logger.info("="*80)
        
        cascade_tracker = CascadeTracker()
        
        # Prepare update dataframes (if available)
        biometric_updates = None
        demographic_updates = None
        
        # Track cascade
        tracked_df = cascade_tracker.track_cascade(
            individual_df,
            biometric_updates=biometric_updates,
            demographic_updates=demographic_updates
        )
        
        logger.info(f"Tracked cascade for {len(tracked_df)} individuals")
        logger.info(f"Transition failure rate: {tracked_df['transition_failure'].mean():.2%}")
        
        # Step 3: Feature Engineering
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Feature Engineering")
        logger.info("="*80)
        
        feature_engineer = FeatureEngineer()
        
        # Split data first for proper feature engineering (no leakage)
        from sklearn.model_selection import train_test_split
        from config import RANDOM_STATE, TEST_SIZE
        
        # Split before feature engineering to prevent leakage
        train_idx, test_idx = train_test_split(
            tracked_df.index, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE,
            stratify=tracked_df['transition_failure'] if 'transition_failure' in tracked_df.columns else None
        )
        
        train_df = tracked_df.loc[train_idx].copy()
        test_df = tracked_df.loc[test_idx].copy()
        
        # Fit on training data only
        logger.info(f"Fitting FeatureEngineer on {len(train_df)} training samples...")
        feature_engineer.fit(train_df, target_col='transition_failure')
        
        # Transform both train and test
        train_features_df = feature_engineer.transform(train_df)
        test_features_df = feature_engineer.transform(test_df)
        
        # Combine for statistical analysis (features already computed safely)
        features_df = pd.concat([train_features_df, test_features_df], ignore_index=True)
        feature_list = feature_engineer.get_feature_names()
        
        logger.info(f"Created {len(feature_list)} features (fit on training, applied to train+test)")
        
        # Step 4: Statistical Analysis
        logger.info("\n" + "="*80)
        logger.info("STEP 4: Statistical Analysis")
        logger.info("="*80)
        
        stats_analyzer = StatisticalAnalyzer()
        
        # Completion rates
        completion_rates = stats_analyzer.compute_completion_rates(tracked_df)
        logger.info("\nCompletion Rates:")
        logger.info(completion_rates.to_string())
        
        # Cohort analysis
        gender_completion = stats_analyzer.compute_cohort_completion_rates(tracked_df, 'gender')
        urban_rural_completion = stats_analyzer.compute_cohort_completion_rates(tracked_df, 'urban_rural')
        
        # Gender divergence
        gender_divergence = stats_analyzer.compute_gender_divergence(tracked_df)
        logger.info("\nGender Divergence:")
        logger.info(gender_divergence.to_string())
        
        # Rural-urban lag
        rural_urban_lag = stats_analyzer.compute_rural_urban_lag(tracked_df)
        logger.info("\nRural-Urban Lag:")
        logger.info(rural_urban_lag.to_string())
        
        # High-risk cohorts
        high_risk_cohorts = stats_analyzer.identify_high_risk_cohorts(tracked_df)
        logger.info("\nHigh-Risk Cohorts (Top 10):")
        logger.info(high_risk_cohorts.head(10).to_string())
        
        # Statistical tests
        statistical_tests = stats_analyzer.compute_statistical_tests(tracked_df)
        logger.info("\nStatistical Tests:")
        logger.info(statistical_tests.to_string())
        
        # Step 5: ML Modeling
        logger.info("\n" + "="*80)
        logger.info("STEP 5: ML Modeling")
        logger.info("="*80)
        
        ml_modeler = MLModeler()
        
        # Prepare data (already split for feature engineering)
        X_train = train_features_df[feature_list].copy()
        y_train = train_df['transition_failure'].astype(int)
        X_test = test_features_df[feature_list].copy()
        y_test = test_df['transition_failure'].astype(int)
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median()).fillna(0)
        X_test = X_test.fillna(X_test.median()).fillna(0)
        
        # Set feature names in MLModeler
        ml_modeler.feature_names = feature_list
        
        logger.info(f"Training set: {len(X_train)} samples, {len(feature_list)} features")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Target distribution - Train: {y_train.value_counts().to_dict()}, Test: {y_test.value_counts().to_dict()}")
        
        # Train models
        logger.info("\nTraining Logistic Regression...")
        lr_results = ml_modeler.train_logistic_regression(X_train, y_train, X_test, y_test)
        logger.info(f"Logistic Regression - Train Score: {lr_results['model'].score(X_train, y_train):.4f}")
        logger.info(f"Logistic Regression - Test Score: {lr_results['model'].score(X_test, y_test):.4f}")
        
        logger.info("\nTraining Random Forest...")
        rf_results = ml_modeler.train_random_forest(X_train, y_train, X_test, y_test)
        logger.info(f"Random Forest - Train Score: {rf_results['model'].score(X_train, y_train):.4f}")
        logger.info(f"Random Forest - Test Score: {rf_results['model'].score(X_test, y_test):.4f}")
        
        logger.info("\nTraining XGBoost...")
        xgb_results = ml_modeler.train_xgboost(X_train, y_train, X_test, y_test)
        if xgb_results is not None:
            logger.info(f"XGBoost - Train Score: {xgb_results['model'].score(X_train, y_train):.4f}")
            logger.info(f"XGBoost - Test Score: {xgb_results['model'].score(X_test, y_test):.4f}")
        
        # Sanity check: Train on shuffled labels (should perform poorly if model is learning correctly)
        logger.info("\n" + "="*80)
        logger.info("SANITY CHECK: Training on Shuffled Labels")
        logger.info("="*80)
        logger.info("If model is learning correctly, shuffled labels should give poor accuracy (~50% for binary)")
        
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        
        # Shuffle labels to break feature-target relationship
        y_shuffled_train = np.random.permutation(y_train.values)
        y_shuffled_test = np.random.permutation(y_test.values)
        
        # Train on shuffled labels
        sanity_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
        sanity_model.fit(X_train, y_shuffled_train)
        
        train_shuffled_score = sanity_model.score(X_train, y_shuffled_train)
        test_shuffled_score = sanity_model.score(X_test, y_shuffled_test)
        
        logger.info(f"Sanity Check - Train Score (shuffled labels): {train_shuffled_score:.4f}")
        logger.info(f"Sanity Check - Test Score (shuffled labels): {test_shuffled_score:.4f}")
        
        if test_shuffled_score > 0.6:
            logger.warning("WARNING: High accuracy on shuffled labels suggests potential data leakage or spurious correlations!")
        else:
            logger.info("✓ Sanity check passed: Model performs poorly on shuffled labels (as expected)")
        
        # SHAP explainability (for best model)
        if xgb_results is not None:
            logger.info("\nComputing SHAP values for XGBoost...")
            shap_values = ml_modeler.compute_shap_values('xgboost', X_test, max_samples=100)
            best_model = 'xgboost'
        else:
            logger.info("\nComputing SHAP values for Random Forest...")
            shap_values = ml_modeler.compute_shap_values('random_forest', X_test, max_samples=100)
            best_model = 'random_forest'
        
        # Feature Availability Analysis
        logger.info("\n" + "="*80)
        logger.info("FEATURE AVAILABILITY ANALYSIS AT PREDICTION TIME")
        logger.info("="*80)
        
        from feature_availability_check import FeatureAvailabilityAnalyzer
        
        analyzer = FeatureAvailabilityAnalyzer()
        availability_report = analyzer.analyze_features(feature_list)
        analyzer.print_report(availability_report)
        
        # Save report
        import json
        report_path = OUTPUT_DIR / 'feature_availability_report.json'
        with open(report_path, 'w') as f:
            json.dump(availability_report, f, indent=2)
        logger.info(f"\nFeature availability report saved to {report_path}")
        
        # Model comparison
        logger.info("\nModel Performance Summary:")
        for model_name, results in ml_modeler.model_results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  AUC: {results['auc']:.4f}")
            logger.info(f"  Accuracy: {results['accuracy']:.4f}")
            logger.info(f"  Precision: {results['precision']:.4f}")
            logger.info(f"  Recall: {results['recall']:.4f}")
            logger.info(f"  F1-Score: {results['f1']:.4f}")
        
        # Step 6: Visualizations
        logger.info("\n" + "="*80)
        logger.info("STEP 6: Creating Visualizations")
        logger.info("="*80)
        
        visualizer = Visualizer()
        
        # Plot completion rates
        visualizer.plot_completion_rates(completion_rates)
        
        # Plot cohort completion
        if len(gender_completion) > 0:
            visualizer.plot_cohort_completion(gender_completion, 'gender')
        if len(urban_rural_completion) > 0:
            visualizer.plot_cohort_completion(urban_rural_completion, 'urban_rural')
        
        # Plot gender divergence
        if len(gender_divergence) > 0:
            visualizer.plot_gender_divergence(gender_divergence)
        
        # Plot rural-urban lag
        if len(rural_urban_lag) > 0:
            visualizer.plot_rural_urban_lag(rural_urban_lag)
        
        # Plot district heatmap
        visualizer.plot_high_risk_districts_heatmap(tracked_df)
        
        # Plot feature importance
        if best_model in ml_modeler.model_results:
            if 'feature_importance' in ml_modeler.model_results[best_model]:
                visualizer.plot_feature_importance(
                    ml_modeler.model_results[best_model]['feature_importance'],
                    best_model
                )
        
        # Plot ROC curves
        visualizer.plot_roc_curves(ml_modeler.model_results)
        
        # Plot model comparison
        visualizer.plot_model_comparison(ml_modeler.model_results)
        
        logger.info(f"\nVisualizations saved to {visualizer.output_dir}")
        
        # Step 7: Generate Reports
        logger.info("\n" + "="*80)
        logger.info("STEP 7: Generating Reports")
        logger.info("="*80)
        
        generate_summary_report(
            tracked_df, completion_rates, gender_divergence, rural_urban_lag,
            high_risk_cohorts, ml_modeler.model_results, statistical_tests
        )
        
        # Save results to CSV
        tracked_df.to_csv(OUTPUT_DIR / 'tracked_cascade_results.csv', index=False)
        completion_rates.to_csv(OUTPUT_DIR / 'completion_rates.csv', index=False)
        gender_divergence.to_csv(OUTPUT_DIR / 'gender_divergence.csv', index=False)
        rural_urban_lag.to_csv(OUTPUT_DIR / 'rural_urban_lag.csv', index=False)
        high_risk_cohorts.to_csv(OUTPUT_DIR / 'high_risk_cohorts.csv', index=False)
        
        # Save feature importance
        if best_model in ml_modeler.model_results:
            if 'feature_importance' in ml_modeler.model_results[best_model]:
                ml_modeler.model_results[best_model]['feature_importance'].to_csv(
                    OUTPUT_DIR / f'feature_importance_{best_model}.csv', index=False
                )
        
        logger.info(f"\nResults saved to {OUTPUT_DIR}")
        
        # Step 8: Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Total execution time: {duration}")
        logger.info(f"Results saved to: {OUTPUT_DIR}")
        logger.info(f"Reports saved to: {REPORTS_DIR}")
        
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        raise


def generate_summary_report(tracked_df: pd.DataFrame,
                           completion_rates: pd.DataFrame,
                           gender_divergence: pd.DataFrame,
                           rural_urban_lag: pd.DataFrame,
                           high_risk_cohorts: pd.DataFrame,
                           model_results: dict,
                           statistical_tests: pd.DataFrame):
    """
    Generate policy-ready summary report.
    """
    report_path = REPORTS_DIR / 'policy_summary_report.md'
    
    with open(report_path, 'w') as f:
        f.write("# 18th Birthday Cascade Analysis: Policy Summary Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This analysis examines identity transition failures when citizens turn 18, ")
        f.write("focusing on the 90-day behavioral cascade after mandatory biometric updates. ")
        f.write("The study identifies demographic, geographic, and gender-based divergence in ")
        f.write("identity update completion rates.\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Overall failure rate
        failure_rate = tracked_df['transition_failure'].mean()
        f.write(f"### 1. Overall Transition Failure Rate: {failure_rate:.1%}\n\n")
        f.write(f"Approximately {failure_rate:.1%} of individuals fail to complete all required ")
        f.write("updates within the 90-day window after turning 18. This translates to significant ")
        f.write("potential KYC rejections and DBT failures in subsequent years.\n\n")
        
        # Completion rates
        f.write("### 2. Update Completion Rates by Window\n\n")
        f.write("| Update Type | Window (Days) | Completion Rate | Missing Rate |\n")
        f.write("|-------------|---------------|-----------------|--------------|\n")
        for _, row in completion_rates.iterrows():
            f.write(f"| {row['update_type']} | {row['window_days']} | "
                   f"{row['completion_rate_pct']:.1f}% | {row['missing_rate_pct']:.1f}% |\n")
        f.write("\n")
        
        # Gender divergence
        if len(gender_divergence) > 0:
            f.write("### 3. Gender-Based Divergence\n\n")
            f.write("Significant gender-based differences exist in completion rates:\n\n")
            for _, row in gender_divergence.iterrows():
                if abs(row['relative_divergence']) > 5:  # Highlight significant divergence
                    f.write(f"- **{row['update_type']}**: {row['divergence_pct_points']:.1f} percentage point ")
                    f.write(f"difference (Male: {row['male_completion_rate_pct']:.1f}%, ")
                    f.write(f"Female: {row['female_completion_rate_pct']:.1f}%)\n")
            f.write("\n")
        
        # Rural-Urban lag
        if len(rural_urban_lag) > 0:
            f.write("### 4. Rural-Urban Lag\n\n")
            f.write("Rural areas consistently lag behind urban areas in update completion:\n\n")
            for _, row in rural_urban_lag.iterrows():
                f.write(f"- **{row['update_type']}**: {row['completion_lag_pct_points']:.1f} percentage points ")
                f.write(f"lag (Urban: {row['urban_completion_rate_pct']:.1f}%, ")
                f.write(f"Rural: {row['rural_completion_rate_pct']:.1f}%)\n")
            f.write("\n")
        
        # High-risk cohorts
        f.write("### 5. High-Risk Cohorts\n\n")
        f.write("The following cohorts show elevated transition failure rates:\n\n")
        top_risks = high_risk_cohorts.head(10)
        for _, row in top_risks.iterrows():
            f.write(f"- **{row.get('state', row.get('district', row.get('gender', row.get('urban_rural', 'Unknown'))))}**: ")
            f.write(f"{row['failure_rate']:.1%} failure rate (n={row['count']})\n")
        f.write("\n")
        
        # Model performance
        f.write("### 6. Predictive Model Performance\n\n")
        f.write("ML models trained to predict transition failure:\n\n")
        for model_name, results in model_results.items():
            f.write(f"- **{model_name.upper()}**: AUC = {results['auc']:.3f}, ")
            f.write(f"F1-Score = {results['f1']:.3f}\n")
        f.write("\n")
        
        # Statistical significance
        if len(statistical_tests) > 0:
            f.write("### 7. Statistical Significance Tests\n\n")
            for _, test in statistical_tests.iterrows():
                sig = "✓" if test['significant'] else "✗"
                f.write(f"- **{test['test']}**: {sig} p-value = {test['p_value']:.4f} ")
                f.write(f"({test['cohort1']} vs {test['cohort2']})\n")
            f.write("\n")
        
        # Policy recommendations
        f.write("## Policy Recommendations\n\n")
        f.write("1. **Targeted Outreach**: Focus on high-risk districts and rural areas with ")
        f.write("proactive reminders about update deadlines.\n\n")
        f.write("2. **Gender-Sensitive Interventions**: Address gender-based divergence through ")
        f.write("tailored communication strategies.\n\n")
        f.write("3. **Early Warning System**: Deploy predictive models to identify individuals ")
        f.write("at risk of transition failure and intervene proactively.\n\n")
        f.write("4. **Infrastructure Improvements**: Reduce barriers in rural areas through ")
        f.write("enhanced digital infrastructure and support services.\n\n")
        f.write("5. **Monitoring Framework**: Establish continuous monitoring of the 90-day ")
        f.write("cascade to track intervention effectiveness.\n\n")
        
        f.write("## Data and Methodology\n\n")
        f.write(f"- **Sample Size**: {len(tracked_df):,} individuals\n")
        f.write(f"- **Analysis Period**: 90-day window post-18th birthday\n")
        f.write(f"- **Update Types**: Biometric (Day 0), Mobile (0-30), Address (31-60), Name (61-90)\n")
        f.write(f"- **ML Models**: Logistic Regression, Random Forest, XGBoost with SHAP explainability\n\n")
    
    logger.info(f"Policy summary report saved to {report_path}")


if __name__ == "__main__":
    main()
