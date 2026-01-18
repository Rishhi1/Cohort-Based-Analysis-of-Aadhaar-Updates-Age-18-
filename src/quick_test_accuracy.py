"""
Quick test script to get accuracy scores from trained models
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from main_pipeline import main
import logging

# Suppress verbose logging for quick output
logging.getLogger().setLevel(logging.WARNING)

if __name__ == "__main__":
    try:
        main()
        print("\n" + "="*80)
        print("PIPELINE COMPLETED - Check outputs directory for accuracy scores")
        print("="*80)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
