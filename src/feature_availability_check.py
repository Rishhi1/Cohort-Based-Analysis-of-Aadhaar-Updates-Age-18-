"""
Feature Availability Analysis at Prediction Time

Critical Question: "Are these patterns available at prediction time?"

This script analyzes which features are available at different prediction times:
- Day 0: When citizen turns 18 (before any updates)
- Day 30: After mobile window closes
- Day 60: After address window closes  
- Day 90: After all windows close (retrospective)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureAvailabilityAnalyzer:
    """
    Analyzes which features are available at different prediction times.
    """
    
    def __init__(self):
        # Features available at different times
        self.day0_features = set()  # Available at prediction time (day 0)
        self.day30_features = set()  # Available after mobile window
        self.day60_features = set()  # Available after address window
        self.day90_features = set()  # Available after all windows (retrospective)
        
        # Feature categories
        self.requires_future_knowledge = set()
        self.available_at_prediction = set()
        self.geographic_features = set()
        self.demographic_features = set()
        self.temporal_features = set()
        
    def analyze_features(self, feature_list: List[str]) -> Dict:
        """
        Analyze feature availability at prediction time.
        
        Args:
            feature_list: List of feature names used in the model
            
        Returns:
            Dictionary with availability analysis
        """
        logger.info(f"Analyzing {len(feature_list)} features for prediction-time availability...")
        
        # Categorize features
        for feature in feature_list:
            # Demographic/static features (always available)
            if any(x in feature for x in ['gender', 'state', 'district', 'urban_rural', 'encoded']):
                if '_encoded' in feature:
                    self.demographic_features.add(feature)
                    self.day0_features.add(feature)  # Available at day 0
                    self.available_at_prediction.add(feature)
            
            # Geographic aggregation features (learned from training, available at day 0)
            elif 'completion_rate' in feature and any(x in feature for x in ['state', 'district', 'urban_rural']):
                self.geographic_features.add(feature)
                self.day0_features.add(feature)  # Available at day 0 (learned from training)
                self.available_at_prediction.add(feature)
            
            # Temporal features (available at day 0)
            elif any(x in feature for x in ['birthday', 'enrolment', 'year', 'month', 'quarter', 'dow']):
                self.temporal_features.add(feature)
                self.day0_features.add(feature)
                self.available_at_prediction.add(feature)
            
            # Time-to-update features (REQUIRES FUTURE KNOWLEDGE)
            elif 'time_to_update' in feature:
                update_type = feature.split('_time_to_update')[0]
                self.requires_future_knowledge.add(feature)
                
                # Available after respective window closes
                if update_type == 'biometric':
                    self.day0_features.add(feature)  # Day 0 window
                elif update_type == 'mobile':
                    self.day30_features.add(feature)  # Day 30 window
                elif update_type == 'address':
                    self.day60_features.add(feature)  # Day 60 window
                elif update_type == 'name':
                    self.day90_features.add(feature)  # Day 90 window
            
            # Window completion flags (REQUIRES FUTURE KNOWLEDGE)
            elif 'window_completed' in feature:
                update_type = feature.split('_window_completed')[0]
                self.requires_future_knowledge.add(feature)
                
                if update_type == 'biometric':
                    self.day0_features.add(feature)
                elif update_type == 'mobile':
                    self.day30_features.add(feature)
                elif update_type == 'address':
                    self.day60_features.add(feature)
                elif update_type == 'name':
                    self.day90_features.add(feature)
            
            # Gap features (REQUIRES FUTURE KNOWLEDGE - need multiple update times)
            elif feature.startswith('gap_') or 'gap' in feature:
                self.requires_future_knowledge.add(feature)
                # Gaps require knowing when at least 2 updates happened
                if 'biometric_to_mobile' in feature:
                    self.day30_features.add(feature)
                elif 'mobile_to_address' in feature:
                    self.day60_features.add(feature)
                elif 'address_to_name' in feature:
                    self.day90_features.add(feature)
                elif 'avg_gap' in feature or 'max_gap' in feature:
                    self.day90_features.add(feature)  # Need all updates
            
            # Missing indicators (REQUIRES FUTURE KNOWLEDGE - need to know if update happened)
            elif 'missing' in feature:
                update_type = feature.split('_missing')[0]
                self.requires_future_knowledge.add(feature)
                
                if update_type == 'biometric':
                    self.day0_features.add(feature)
                elif update_type == 'mobile':
                    self.day30_features.add(feature)
                elif update_type == 'address':
                    self.day60_features.add(feature)
                elif update_type == 'name':
                    self.day90_features.add(feature)
            
            # Late indicators
            elif 'is_late' in feature:
                self.requires_future_knowledge.add(feature)
                # Can only determine if late after window closes
                update_type = feature.split('_is_late')[0]
                if update_type == 'biometric':
                    self.day0_features.add(feature)
                elif update_type == 'mobile':
                    self.day30_features.add(feature)
                elif update_type == 'address':
                    self.day60_features.add(feature)
                elif update_type == 'name':
                    self.day90_features.add(feature)
            
            else:
                # Unknown feature - assume available at day 0 for now
                logger.warning(f"Unknown feature category: {feature}")
                self.day0_features.add(feature)
        
        # Generate report
        report = self._generate_report(feature_list)
        
        return report
    
    def _generate_report(self, all_features: List[str]) -> Dict:
        """
        Generate availability report.
        """
        total_features = len(all_features)
        day0_count = len(self.day0_features)
        day30_count = len(self.day30_features)
        day60_count = len(self.day60_features)
        day90_count = len(self.day90_features)
        future_knowledge_count = len(self.requires_future_knowledge)
        available_at_pred_count = len(self.available_at_prediction)
        
        # Calculate coverage
        day0_coverage = day0_count / total_features if total_features > 0 else 0
        day30_coverage = day30_count / total_features if total_features > 0 else 0
        day60_coverage = day60_count / total_features if total_features > 0 else 0
        day90_coverage = day90_count / total_features if total_features > 0 else 0
        
        report = {
            'total_features': total_features,
            'available_at_day0': {
                'count': day0_count,
                'percentage': day0_coverage * 100,
                'features': sorted(self.day0_features)
            },
            'available_at_day30': {
                'count': day30_count,
                'percentage': day30_coverage * 100,
                'features': sorted(self.day30_features)
            },
            'available_at_day60': {
                'count': day60_count,
                'percentage': day60_coverage * 100,
                'features': sorted(self.day60_features)
            },
            'available_at_day90': {
                'count': day90_count,
                'percentage': day90_coverage * 100,
                'features': sorted(self.day90_features)
            },
            'requires_future_knowledge': {
                'count': future_knowledge_count,
                'percentage': (future_knowledge_count / total_features * 100) if total_features > 0 else 0,
                'features': sorted(self.requires_future_knowledge)
            },
            'truly_available_at_prediction': {
                'count': available_at_pred_count,
                'percentage': (available_at_pred_count / total_features * 100) if total_features > 0 else 0,
                'features': sorted(self.available_at_prediction)
            },
            'categories': {
                'demographic': sorted(self.demographic_features),
                'geographic': sorted(self.geographic_features),
                'temporal': sorted(self.temporal_features)
            }
        }
        
        return report
    
    def print_report(self, report: Dict):
        """
        Print formatted availability report.
        """
        print("\n" + "="*80)
        print("FEATURE AVAILABILITY ANALYSIS AT PREDICTION TIME")
        print("="*80)
        
        print(f"\nTotal Features: {report['total_features']}")
        
        print("\n" + "-"*80)
        print("FEATURES AVAILABLE AT DAY 0 (Prediction Time - When Citizen Turns 18)")
        print("-"*80)
        print(f"Count: {report['available_at_day0']['count']} ({report['available_at_day0']['percentage']:.1f}%)")
        print("\nFeatures:")
        for feat in report['available_at_day0']['features']:
            if feat in report['truly_available_at_prediction']['features']:
                print(f"  ✓ {feat} (available at prediction)")
            else:
                print(f"  ⚠ {feat} (requires future knowledge)")
        
        print("\n" + "-"*80)
        print("FEATURES REQUIRING FUTURE KNOWLEDGE")
        print("-"*80)
        print(f"Count: {report['requires_future_knowledge']['count']} ({report['requires_future_knowledge']['percentage']:.1f}%)")
        print("\nThese features are NOT available at prediction time (day 0):")
        for feat in report['requires_future_knowledge']['features']:
            print(f"  ✗ {feat}")
        
        print("\n" + "-"*80)
        print("TRULY AVAILABLE AT PREDICTION TIME (Day 0)")
        print("-"*80)
        print(f"Count: {report['truly_available_at_prediction']['count']} ({report['truly_available_at_prediction']['percentage']:.1f}%)")
        print("\nFeatures:")
        for feat in report['truly_available_at_prediction']['features']:
            print(f"  ✓ {feat}")
        
        print("\n" + "-"*80)
        print("FEATURE CATEGORIES")
        print("-"*80)
        print(f"Demographic Features: {len(report['categories']['demographic'])}")
        print(f"Geographic Features: {len(report['categories']['geographic'])}")
        print(f"Temporal Features: {len(report['categories']['temporal'])}")
        
        print("\n" + "-"*80)
        print("AVAILABILITY BY TIMELINE")
        print("-"*80)
        print(f"Day 0 (Turn 18): {report['available_at_day0']['count']} features ({report['available_at_day0']['percentage']:.1f}%)")
        print(f"Day 30 (After mobile): {report['available_at_day30']['count']} features ({report['available_at_day30']['percentage']:.1f}%)")
        print(f"Day 60 (After address): {report['available_at_day60']['count']} features ({report['available_at_day60']['percentage']:.1f}%)")
        print(f"Day 90 (After all): {report['available_at_day90']['count']} features ({report['available_at_day90']['percentage']:.1f}%)")
        
        print("\n" + "="*80)
        print("CRITICAL INSIGHT")
        print("="*80)
        
        if report['truly_available_at_prediction']['percentage'] < 50:
            print("⚠ WARNING: Less than 50% of features are available at prediction time (day 0)!")
            print("   This means the model relies heavily on future knowledge.")
            print("   For production deployment, you need features that predict BEFORE the cascade completes.")
        else:
            print(f"✓ {report['truly_available_at_prediction']['percentage']:.1f}% of features are available at prediction time.")
            print("   However, many features still require future knowledge of update completions.")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    # Import and get feature list from the pipeline
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from feature_engineering import FeatureEngineer
    
    # This would need to be run after fitting the feature engineer
    # For demonstration, we'll use a sample feature list
    sample_features = [
        # Time-to-update (require future knowledge)
        'biometric_time_to_update', 'mobile_time_to_update', 'address_time_to_update', 'name_time_to_update',
        'biometric_time_to_update_log', 'mobile_time_to_update_log', 'address_time_to_update_log', 'name_time_to_update_log',
        # Window flags (require future knowledge)
        'biometric_window_completed', 'mobile_window_completed', 'address_window_completed', 'name_window_completed',
        # Gap features (require future knowledge)
        'gap_biometric_to_mobile', 'gap_mobile_to_address', 'gap_address_to_name',
        'avg_gap_between_updates', 'max_gap_between_updates',
        # Missing indicators (require future knowledge)
        'biometric_missing', 'mobile_missing', 'address_missing', 'name_missing',
        # Late indicators (require future knowledge)
        'biometric_is_late', 'mobile_is_late', 'address_is_late', 'name_is_late',
        # Geographic (available at day 0)
        'state_completion_rate', 'district_completion_rate', 'urban_rural_completion_rate',
        # Demographic (available at day 0)
        'gender_encoded', 'state_encoded', 'district_encoded', 'urban_rural_encoded',
        # Temporal (available at day 0)
        'eighteenth_birthday_year', 'eighteenth_birthday_month', 'eighteenth_birthday_quarter',
        'eighteenth_birthday_dow', 'years_between_enrolment_and_18',
    ]
    
    analyzer = FeatureAvailabilityAnalyzer()
    report = analyzer.analyze_features(sample_features)
    analyzer.print_report(report)
