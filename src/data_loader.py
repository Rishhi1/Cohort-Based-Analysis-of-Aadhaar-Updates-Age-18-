"""
Data loading and preprocessing module
Handles loading of enrollment, biometric, and demographic datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from config import ENROL_DIR, BIOMETRIC_DIR, DEMOGRAPHIC_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and preprocesses enrollment, biometric, and demographic datasets.
    Handles both aggregated and individual-level data formats.
    """
    
    def __init__(self, enrol_dir: Path = ENROL_DIR, 
                 biometric_dir: Path = BIOMETRIC_DIR,
                 demographic_dir: Path = DEMOGRAPHIC_DIR):
        self.enrol_dir = enrol_dir
        self.biometric_dir = biometric_dir
        self.demographic_dir = demographic_dir
        
    def load_enrolment_data(self, filenames: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load enrollment datasets from CSV files.
        
        Args:
            filenames: List of filenames to load. If None, loads all CSV files.
            
        Returns:
            Combined enrollment DataFrame
        """
        if filenames is None:
            filenames = sorted([f.name for f in self.enrol_dir.glob("*.csv")])
        
        dfs = []
        for filename in filenames:
            filepath = self.enrol_dir / filename
            if filepath.exists():
                logger.info(f"Loading enrollment data from {filename}")
                df = pd.read_csv(filepath, low_memory=False)
                dfs.append(df)
            else:
                logger.warning(f"File not found: {filepath}")
        
        if not dfs:
            raise FileNotFoundError("No enrollment files found")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} enrollment records from {len(dfs)} files")
        
        return combined_df
    
    def load_biometric_data(self, filenames: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load biometric update logs from CSV files.
        
        Args:
            filenames: List of filenames to load. If None, loads all CSV files.
            
        Returns:
            Combined biometric DataFrame
        """
        if filenames is None:
            filenames = sorted([f.name for f in self.biometric_dir.glob("*.csv")])
        
        dfs = []
        for filename in filenames:
            filepath = self.biometric_dir / filename
            if filepath.exists():
                logger.info(f"Loading biometric data from {filename}")
                df = pd.read_csv(filepath, low_memory=False)
                dfs.append(df)
            else:
                logger.warning(f"File not found: {filepath}")
        
        if not dfs:
            raise FileNotFoundError("No biometric files found")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} biometric records from {len(dfs)} files")
        
        return combined_df
    
    def load_demographic_data(self, filenames: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load demographic update logs from CSV files.
        
        Args:
            filenames: List of filenames to load. If None, loads all CSV files.
            
        Returns:
            Combined demographic DataFrame
        """
        if filenames is None:
            filenames = sorted([f.name for f in self.demographic_dir.glob("*.csv")])
        
        dfs = []
        for filename in filenames:
            filepath = self.demographic_dir / filename
            if filepath.exists():
                logger.info(f"Loading demographic data from {filename}")
                df = pd.read_csv(filepath, low_memory=False)
                dfs.append(df)
            else:
                logger.warning(f"File not found: {filepath}")
        
        if not dfs:
            raise FileNotFoundError("No demographic files found")
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} demographic records from {len(dfs)} files")
        
        return combined_df
    
    def preprocess_enrolment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess enrollment data: standardize dates, extract features.
        
        Args:
            df: Raw enrollment DataFrame
            
        Returns:
            Preprocessed enrollment DataFrame
        """
        df = df.copy()
        
        # Convert date column if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
        # Standardize state and district names
        if 'state' in df.columns:
            df['state'] = df['state'].str.strip().str.title()
        if 'district' in df.columns:
            df['district'] = df['district'].str.strip().str.title()
        
        # Create urban_rural indicator from pincode patterns
        # (simplified heuristic: pincodes < 200000 often rural, but this varies)
        if 'pincode' in df.columns:
            # This is a placeholder - real implementation would use proper mapping
            df['urban_rural'] = df['pincode'].apply(
                lambda x: 'Urban' if pd.notna(x) and x > 500000 else 'Rural'
            )
        
        return df
    
    def create_individual_level_simulation(self, 
                                          enrol_df: pd.DataFrame,
                                          biometric_df: pd.DataFrame,
                                          demo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform aggregated data into individual-level records for analysis.
        This simulates individual records from cohort-level aggregates.
        
        Args:
            enrol_df: Enrollment data
            biometric_df: Biometric update data
            demo_df: Demographic update data
            
        Returns:
            Individual-level DataFrame with simulated aadhaar_ids
        """
        logger.info("Creating individual-level simulation from aggregated data")
        
        # Merge datasets on common keys (date, state, district, pincode)
        merged = enrol_df.copy()
        
        # Ensure date columns are in same format for merging
        if 'date' in merged.columns:
            merged['date'] = pd.to_datetime(merged['date'], errors='coerce')
        
        # Merge biometric data
        if 'bio_age_17_' in biometric_df.columns:
            # Rename for clarity
            biometric_agg = biometric_df.rename(columns={'bio_age_17_': 'biometric_updates'})
            # Ensure date format is consistent
            if 'date' in biometric_agg.columns:
                biometric_agg['date'] = pd.to_datetime(biometric_agg['date'], format='%d-%m-%Y', errors='coerce')
            merged = merged.merge(
                biometric_agg[['date', 'state', 'district', 'pincode', 'biometric_updates']],
                on=['date', 'state', 'district', 'pincode'],
                how='left',
                suffixes=('', '_bio')
            )
        
        # Merge demographic data
        if 'demo_age_17_' in demo_df.columns:
            demo_agg = demo_df.rename(columns={'demo_age_17_': 'demographic_updates'})
            # Ensure date format is consistent
            if 'date' in demo_agg.columns:
                demo_agg['date'] = pd.to_datetime(demo_agg['date'], format='%d-%m-%Y', errors='coerce')
            merged = merged.merge(
                demo_agg[['date', 'state', 'district', 'pincode', 'demographic_updates']],
                on=['date', 'state', 'district', 'pincode'],
                how='left',
                suffixes=('', '_demo')
            )
        
        # For each row, expand into individual records based on age_18_greater count
        # This simulates individual-level data
        individual_records = []
        
        for idx, row in merged.iterrows():
            if pd.isna(row.get('age_18_greater', 0)) or row.get('age_18_greater', 0) <= 0:
                continue
            
            num_individuals = int(row.get('age_18_greater', 0))
            
            for i in range(num_individuals):
                record = {
                    'aadhaar_id': f"sim_{idx}_{i}_{hash((row.get('state', ''), row.get('district', ''), row.get('pincode', '')))}",
                    'enrolment_date': row.get('date'),
                    'state': row.get('state'),
                    'district': row.get('district'),
                    'pincode': row.get('pincode'),
                    'urban_rural': row.get('urban_rural', 'Unknown'),
                    # Simulate DOB: 18 years before enrollment date (simplified)
                    'dob': row.get('date') - pd.Timedelta(days=18*365 + np.random.randint(0, 365)),
                    # Simulate gender distribution (50-50 split)
                    'gender': 'M' if np.random.random() < 0.5 else 'F',
                    'biometric_updates_count': row.get('biometric_updates', 0) / max(num_individuals, 1),
                    'demographic_updates_count': row.get('demographic_updates', 0) / max(num_individuals, 1)
                }
                individual_records.append(record)
        
        individual_df = pd.DataFrame(individual_records)
        
        # Calculate 18th birthday date
        if 'dob' in individual_df.columns:
            individual_df['eighteenth_birthday'] = individual_df['dob'] + pd.Timedelta(days=18*365)
            individual_df['current_age'] = (pd.Timestamp.now() - individual_df['dob']).dt.days / 365.25
        
        logger.info(f"Created {len(individual_df)} individual-level records")
        
        return individual_df
