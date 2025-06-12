"""
Main execution script for US Open 2025 prediction system.
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_collection.collect_historical_data import HistoricalDataCollector
from preprocessing.data_cleaner import GolfDataCleaner
from prediction.us_open_2025_predictor import USOpen2025Predictor


def collect_data():
    """Collect historical data from DataGolf API."""
    print("=" * 60)
    print("COLLECTING HISTORICAL DATA")
    print("=" * 60)
    
    collector = HistoricalDataCollector()
    collector.run_full_collection()
    
    print("\nData collection completed!")


def process_data():
    """Process and clean the collected data."""
    print("=" * 60)
    print("PROCESSING DATA")
    print("=" * 60)
    
    cleaner = GolfDataCleaner()
    processed_data = cleaner.process_all_data()
    
    print(f"\nProcessed {len(processed_data)} datasets:")
    for key, df in processed_data.items():
        print(f"  {key}: {df.shape}")
    
    print("\nData processing completed!")


def make_predictions():
    """Generate US Open 2025 predictions."""
    print("=" * 60)
    print("GENERATING US OPEN 2025 PREDICTIONS")
    print("=" * 60)
    
    predictor = USOpen2025Predictor()
    predictions_df, report = predictor.run_full_prediction()
    
    print("\n" + report)
    
    return predictions_df, report


def run_full_pipeline():
    """Run the complete pipeline from data collection to predictions."""
    print("STARTING FULL US OPEN 2025 PREDICTION PIPELINE")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Collect data
        collect_data()
        
        # Step 2: Process data
        process_data()
        
        # Step 3: Make predictions
        predictions_df, report = make_predictions()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Total time: {duration}")
        print(f"Predictions generated for {len(predictions_df)} players")
        
        return True
        
    except Exception as e:
        print(f"\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='US Open 2025 Prediction System')
    parser.add_argument('--step', choices=['collect', 'process', 'predict', 'full'], 
                       default='full', help='Which step to run')
    parser.add_argument('--data-dir', default='data', help='Data directory')
    
    args = parser.parse_args()
    
    # Set data directory
    if args.data_dir != 'data':
        os.environ['DATA_DIR'] = args.data_dir
    
    print(f"US Open 2025 Prediction System")
    print(f"Data directory: {args.data_dir}")
    print(f"Step: {args.step}")
    print()
    
    if args.step == 'collect':
        collect_data()
    elif args.step == 'process':
        process_data()
    elif args.step == 'predict':
        predictions_df, report = make_predictions()
    elif args.step == 'full':
        success = run_full_pipeline()
        if not success:
            sys.exit(1)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
