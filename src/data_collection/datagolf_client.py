"""
DataGolf API Client for collecting golf tournament and player data.
"""

import requests
import pandas as pd
import time
import os
from typing import Dict, List, Optional, Union
from dotenv import load_dotenv
import json

load_dotenv()


class DataGolfClient:
    """Client for interacting with the DataGolf API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the DataGolf client.
        
        Args:
            api_key: DataGolf API key. If None, will try to load from environment.
        """
        self.api_key = api_key or os.getenv('DATAGOLF_API_KEY')
        if not self.api_key:
            raise ValueError("DataGolf API key is required. Set DATAGOLF_API_KEY environment variable.")
        
        self.base_url = "https://feeds.datagolf.com"
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Union[Dict, List]:
        """Make a request to the DataGolf API.
        
        Args:
            endpoint: API endpoint path
            params: Query parameters
            
        Returns:
            JSON response data
        """
        if params is None:
            params = {}
        
        params['key'] = self.api_key
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request to {url}: {e}")
            raise
    
    def get_player_list(self, file_format: str = 'json') -> pd.DataFrame:
        """Get list of all players with IDs.
        
        Args:
            file_format: Response format ('json' or 'csv')
            
        Returns:
            DataFrame with player information
        """
        data = self._make_request('get-player-list', {'file_format': file_format})
        return pd.DataFrame(data)
    
    def get_schedule(self, tour: str = 'all', file_format: str = 'json') -> pd.DataFrame:
        """Get tournament schedule.
        
        Args:
            tour: Tour to get schedule for ('all', 'pga', 'euro', 'kft', 'alt')
            file_format: Response format
            
        Returns:
            DataFrame with schedule information
        """
        data = self._make_request('get-schedule', {
            'tour': tour,
            'file_format': file_format
        })
        return pd.DataFrame(data)

    def get_field_updates(self, tour: str = 'pga', file_format: str = 'json') -> pd.DataFrame:
        """Get field updates for current tournaments.

        Args:
            tour: Tour to get updates for ('pga', 'euro', 'kft', 'opp', 'alt')
            file_format: Response format

        Returns:
            DataFrame with field update information
        """
        data = self._make_request('field-updates', {
            'tour': tour,
            'file_format': file_format
        })
        return pd.DataFrame(data)

    def get_event_list(self, file_format: str = 'json') -> pd.DataFrame:
        """Get list of available historical events.
        
        Returns:
            DataFrame with event IDs and information
        """
        data = self._make_request('historical-raw-data/event-list', {
            'file_format': file_format
        })
        return pd.DataFrame(data)
    
    def get_historical_data(self, tour: str, event_id: Union[str, int],
                          year: int, file_format: str = 'json') -> pd.DataFrame:
        """Get historical round-level data for a tournament.

        NOTE: This requires a premium DataGolf subscription with historical data access.

        Args:
            tour: Tour code ('pga', 'euro', etc.)
            event_id: Event ID or 'all' for all events
            year: Calendar year
            file_format: Response format

        Returns:
            DataFrame with historical tournament data
        """
        try:
            data = self._make_request('historical-raw-data/rounds', {
                'tour': tour,
                'event_id': str(event_id),
                'year': str(year),
                'file_format': file_format
            })
            return pd.DataFrame(data)
        except Exception as e:
            if "api key does not have access" in str(e).lower():
                print(f"WARNING: Historical data access requires premium subscription.")
                print(f"Returning empty DataFrame for {tour} event {event_id} year {year}")
                return pd.DataFrame()
            else:
                raise
    
    def get_dg_rankings(self, file_format: str = 'json') -> pd.DataFrame:
        """Get current DataGolf rankings.
        
        Returns:
            DataFrame with current player rankings
        """
        data = self._make_request('preds/get-dg-rankings', {
            'file_format': file_format
        })
        return pd.DataFrame(data)
    
    def get_skill_ratings(self, display: str = 'value', 
                         file_format: str = 'json') -> pd.DataFrame:
        """Get player skill ratings.
        
        Args:
            display: How to display stats ('value' or 'rank')
            file_format: Response format
            
        Returns:
            DataFrame with skill ratings
        """
        data = self._make_request('preds/skill-ratings', {
            'display': display,
            'file_format': file_format
        })
        return pd.DataFrame(data)
    
    def get_pre_tournament_predictions(self, tour: str = 'pga', 
                                     odds_format: str = 'percent',
                                     file_format: str = 'json') -> pd.DataFrame:
        """Get pre-tournament predictions for current event.
        
        Args:
            tour: Tour code
            odds_format: Format for odds ('percent', 'american', 'decimal', 'fraction')
            file_format: Response format
            
        Returns:
            DataFrame with predictions
        """
        data = self._make_request('preds/pre-tournament', {
            'tour': tour,
            'odds_format': odds_format,
            'file_format': file_format
        })
        return pd.DataFrame(data)
    
    def get_prediction_archive(self, event_id: Optional[int] = None, 
                             year: int = 2025, odds_format: str = 'percent',
                             file_format: str = 'json') -> pd.DataFrame:
        """Get historical pre-tournament predictions.
        
        Args:
            event_id: Event ID (if None, gets most recent)
            year: Calendar year
            odds_format: Format for odds
            file_format: Response format
            
        Returns:
            DataFrame with historical predictions
        """
        params = {
            'year': str(year),
            'odds_format': odds_format,
            'file_format': file_format
        }
        if event_id:
            params['event_id'] = str(event_id)
            
        data = self._make_request('preds/pre-tournament-archive', params)
        return pd.DataFrame(data)
