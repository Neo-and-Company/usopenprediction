"""
Alternative data collection strategy using available DataGolf API endpoints.
This works with basic DataGolf API subscriptions.
"""

import os
import pandas as pd
import time
from datetime import datetime
from tqdm import tqdm
from datagolf_client import DataGolfClient


class AvailableDataCollector:
    """Collects data using endpoints available with basic DataGolf subscription."""
    
    def __init__(self, data_dir: str = 'data/raw'):
        """Initialize the data collector.
        
        Args:
            data_dir: Directory to save raw data files
        """
        self.client = DataGolfClient()
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # US Open event ID
        self.us_open_event_id = 26
        
        # Years for prediction archives (what's available)
        self.prediction_years = list(range(2020, 2025))
        
    def collect_player_list(self) -> pd.DataFrame:
        """Collect and save player list with IDs."""
        print("Collecting player list...")
        
        players_df = self.client.get_player_list()
        
        # Save to file
        filepath = os.path.join(self.data_dir, 'players.csv')
        players_df.to_csv(filepath, index=False)
        print(f"Saved {len(players_df)} players to {filepath}")
        
        return players_df
    
    def collect_current_rankings(self) -> pd.DataFrame:
        """Collect current DataGolf rankings."""
        print("Collecting current DataGolf rankings...")
        
        rankings_df = self.client.get_dg_rankings()
        
        # Save to file
        filepath = os.path.join(self.data_dir, 'current_rankings.csv')
        rankings_df.to_csv(filepath, index=False)
        print(f"Saved {len(rankings_df)} rankings to {filepath}")
        
        return rankings_df
    
    def collect_skill_ratings(self) -> pd.DataFrame:
        """Collect current skill ratings."""
        print("Collecting skill ratings...")
        
        skills_df = self.client.get_skill_ratings()
        
        # Save to file
        filepath = os.path.join(self.data_dir, 'skill_ratings.csv')
        skills_df.to_csv(filepath, index=False)
        print(f"Saved {len(skills_df)} skill ratings to {filepath}")
        
        return skills_df
    
    def collect_tournament_schedule(self) -> pd.DataFrame:
        """Collect tournament schedules."""
        print("Collecting tournament schedules...")
        
        schedule_df = self.client.get_schedule(tour='all')
        
        # Save to file
        filepath = os.path.join(self.data_dir, 'tournament_schedule.csv')
        schedule_df.to_csv(filepath, index=False)
        print(f"Saved {len(schedule_df)} tournaments to {filepath}")
        
        return schedule_df
    
    def collect_field_updates(self) -> pd.DataFrame:
        """Collect current field updates."""
        print("Collecting field updates...")
        
        field_df = self.client.get_field_updates(tour='pga')
        
        # Save to file
        filepath = os.path.join(self.data_dir, 'field_updates.csv')
        field_df.to_csv(filepath, index=False)
        print(f"Saved {len(field_df)} field updates to {filepath}")
        
        return field_df
    
    def collect_prediction_archives(self) -> pd.DataFrame:
        """Collect historical pre-tournament predictions."""
        print("Collecting prediction archives...")
        
        all_predictions = []
        
        for year in tqdm(self.prediction_years, desc="Prediction archives"):
            try:
                # Try to get US Open predictions for this year
                predictions = self.client.get_prediction_archive(
                    event_id=self.us_open_event_id,
                    year=year
                )
                
                if not predictions.empty:
                    predictions['year'] = year
                    predictions['event_name'] = 'US Open'
                    all_predictions.append(predictions)
                    print(f"  Collected predictions for {year}")
                else:
                    print(f"  No predictions available for {year}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error collecting predictions for {year}: {e}")
                continue
        
        if all_predictions:
            predictions_df = pd.concat(all_predictions, ignore_index=True)
            
            # Save to file
            filepath = os.path.join(self.data_dir, 'us_open_prediction_archives.csv')
            predictions_df.to_csv(filepath, index=False)
            print(f"Saved {len(predictions_df)} prediction records to {filepath}")
            
            return predictions_df
        else:
            print("No prediction archives collected")
            return pd.DataFrame()
    
    def collect_current_predictions(self) -> pd.DataFrame:
        """Collect current pre-tournament predictions if available."""
        print("Collecting current pre-tournament predictions...")
        
        try:
            predictions_df = self.client.get_pre_tournament_predictions(tour='pga')
            
            if not predictions_df.empty:
                # Save to file
                filepath = os.path.join(self.data_dir, 'current_predictions.csv')
                predictions_df.to_csv(filepath, index=False)
                print(f"Saved {len(predictions_df)} current predictions to {filepath}")
            else:
                print("No current predictions available")
            
            return predictions_df
            
        except Exception as e:
            print(f"Error collecting current predictions: {e}")
            return pd.DataFrame()
    
    def run_available_collection(self):
        """Run data collection using available endpoints."""
        print("Starting data collection with available endpoints...")
        print(f"Data will be saved to: {self.data_dir}")
        
        start_time = datetime.now()
        
        # Collect all available data
        players_df = self.collect_player_list()
        rankings_df = self.collect_current_rankings()
        skills_df = self.collect_skill_ratings()
        schedule_df = self.collect_tournament_schedule()
        field_df = self.collect_field_updates()
        predictions_df = self.collect_prediction_archives()
        current_preds_df = self.collect_current_predictions()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nData collection completed in {duration}")
        print(f"Files saved to {self.data_dir}:")
        
        # List all files created
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                filepath = os.path.join(self.data_dir, file)
                size = os.path.getsize(filepath)
                print(f"  {file}: {size:,} bytes")
        
        return {
            'players': players_df,
            'rankings': rankings_df,
            'skills': skills_df,
            'schedule': schedule_df,
            'field_updates': field_df,
            'prediction_archives': predictions_df,
            'current_predictions': current_preds_df
        }


if __name__ == "__main__":
    collector = AvailableDataCollector()
    data = collector.run_available_collection()
    
    print("\nCollection Summary:")
    for key, df in data.items():
        if not df.empty:
            print(f"  {key}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"  {key}: No data")
