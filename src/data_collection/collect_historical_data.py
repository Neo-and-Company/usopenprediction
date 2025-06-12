"""
Collect historical tournament data for US Open prediction model.
"""

import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from tqdm import tqdm
import json

from .datagolf_client import DataGolfClient


class HistoricalDataCollector:
    """Collects and organizes historical golf tournament data."""
    
    def __init__(self, data_dir: str = 'data/raw'):
        """Initialize the data collector.
        
        Args:
            data_dir: Directory to save raw data files
        """
        self.client = DataGolfClient()
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # US Open event ID (this is typically 26 for US Open)
        self.us_open_event_id = 26
        
        # Years to collect data for (adjust based on available data)
        self.years = list(range(2017, 2025))  # 2017-2024
        
        # Major championship event IDs (for broader context)
        self.major_event_ids = {
            'masters': 14,
            'pga_championship': 33,
            'us_open': 26,
            'open_championship': 28
        }
    
    def collect_player_list(self) -> pd.DataFrame:
        """Collect and save player list with IDs."""
        print("Collecting player list...")
        
        players_df = self.client.get_player_list()
        
        # Save to file
        filepath = os.path.join(self.data_dir, 'players.csv')
        players_df.to_csv(filepath, index=False)
        print(f"Saved {len(players_df)} players to {filepath}")
        
        return players_df
    
    def collect_event_list(self) -> pd.DataFrame:
        """Collect and save available events list."""
        print("Collecting event list...")
        
        events_df = self.client.get_event_list()
        
        # Save to file
        filepath = os.path.join(self.data_dir, 'events.csv')
        events_df.to_csv(filepath, index=False)
        print(f"Saved {len(events_df)} events to {filepath}")
        
        return events_df
    
    def collect_us_open_history(self) -> pd.DataFrame:
        """Collect historical US Open data."""
        print("Collecting US Open historical data...")
        
        all_data = []
        
        for year in tqdm(self.years, desc="Collecting US Open data"):
            try:
                # Get US Open data for this year
                data = self.client.get_historical_data(
                    tour='pga',
                    event_id=self.us_open_event_id,
                    year=year
                )
                
                if not data.empty:
                    data['year'] = year
                    data['event_name'] = 'US Open'
                    data['event_id'] = self.us_open_event_id
                    all_data.append(data)
                    print(f"  Collected {len(data)} records for {year}")
                else:
                    print(f"  No data available for {year}")
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Error collecting data for {year}: {e}")
                continue
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save to file
            filepath = os.path.join(self.data_dir, 'us_open_history.csv')
            combined_df.to_csv(filepath, index=False)
            print(f"Saved {len(combined_df)} US Open records to {filepath}")
            
            return combined_df
        else:
            print("No US Open data collected")
            return pd.DataFrame()
    
    def collect_major_championships(self) -> pd.DataFrame:
        """Collect data from all major championships for broader context."""
        print("Collecting major championship data...")
        
        all_majors_data = []
        
        for major_name, event_id in self.major_event_ids.items():
            print(f"  Collecting {major_name} data...")
            
            for year in tqdm(self.years, desc=f"{major_name}", leave=False):
                try:
                    data = self.client.get_historical_data(
                        tour='pga',
                        event_id=event_id,
                        year=year
                    )
                    
                    if not data.empty:
                        data['year'] = year
                        data['event_name'] = major_name
                        data['event_id'] = event_id
                        data['is_major'] = True
                        all_majors_data.append(data)
                    
                    # Rate limiting
                    time.sleep(0.3)
                    
                except Exception as e:
                    print(f"    Error collecting {major_name} {year}: {e}")
                    continue
        
        if all_majors_data:
            combined_df = pd.concat(all_majors_data, ignore_index=True)
            
            # Save to file
            filepath = os.path.join(self.data_dir, 'major_championships.csv')
            combined_df.to_csv(filepath, index=False)
            print(f"Saved {len(combined_df)} major championship records to {filepath}")
            
            return combined_df
        else:
            print("No major championship data collected")
            return pd.DataFrame()
    
    def collect_current_rankings_and_skills(self) -> tuple:
        """Collect current player rankings and skill ratings."""
        print("Collecting current rankings and skills...")
        
        # Get current rankings
        rankings_df = self.client.get_dg_rankings()
        rankings_filepath = os.path.join(self.data_dir, 'current_rankings.csv')
        rankings_df.to_csv(rankings_filepath, index=False)
        print(f"Saved {len(rankings_df)} rankings to {rankings_filepath}")
        
        # Get skill ratings
        skills_df = self.client.get_skill_ratings()
        skills_filepath = os.path.join(self.data_dir, 'current_skills.csv')
        skills_df.to_csv(skills_filepath, index=False)
        print(f"Saved {len(skills_df)} skill ratings to {skills_filepath}")
        
        return rankings_df, skills_df
    
    def collect_pga_tour_data(self, num_years: int = 3) -> pd.DataFrame:
        """Collect broader PGA Tour data for additional context.
        
        Args:
            num_years: Number of recent years to collect
            
        Returns:
            DataFrame with PGA Tour tournament data
        """
        print(f"Collecting PGA Tour data for last {num_years} years...")
        
        recent_years = list(range(2025 - num_years, 2025))
        all_pga_data = []
        
        for year in tqdm(recent_years, desc="PGA Tour data"):
            try:
                # Get all PGA Tour events for this year
                data = self.client.get_historical_data(
                    tour='pga',
                    event_id='all',  # All events
                    year=year
                )
                
                if not data.empty:
                    data['year'] = year
                    # Filter out majors (we have those separately)
                    data = data[~data.get('event_id', 0).isin(self.major_event_ids.values())]
                    all_pga_data.append(data)
                    print(f"  Collected {len(data)} PGA Tour records for {year}")
                
                # Rate limiting
                time.sleep(1.0)  # Longer delay for large requests
                
            except Exception as e:
                print(f"  Error collecting PGA Tour data for {year}: {e}")
                continue
        
        if all_pga_data:
            combined_df = pd.concat(all_pga_data, ignore_index=True)
            
            # Save to file
            filepath = os.path.join(self.data_dir, 'pga_tour_recent.csv')
            combined_df.to_csv(filepath, index=False)
            print(f"Saved {len(combined_df)} PGA Tour records to {filepath}")
            
            return combined_df
        else:
            print("No PGA Tour data collected")
            return pd.DataFrame()
    
    def run_full_collection(self):
        """Run the complete data collection process."""
        print("Starting full historical data collection...")
        print(f"Data will be saved to: {self.data_dir}")
        
        start_time = datetime.now()
        
        # Collect all data
        players_df = self.collect_player_list()
        events_df = self.collect_event_list()
        us_open_df = self.collect_us_open_history()
        majors_df = self.collect_major_championships()
        rankings_df, skills_df = self.collect_current_rankings_and_skills()
        pga_tour_df = self.collect_pga_tour_data()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nData collection completed in {duration}")
        print(f"Files saved to {self.data_dir}:")
        
        # Print summary
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_dir, filename)
                df = pd.read_csv(filepath)
                print(f"  {filename}: {len(df)} records")


if __name__ == "__main__":
    collector = HistoricalDataCollector()
    collector.run_full_collection()
