"""
Data cleaning and preprocessing for golf tournament data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class GolfDataCleaner:
    """Cleans and preprocesses raw golf tournament data."""
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.required_columns = [
            'player_id', 'player_name', 'event_id', 'year'
        ]
        
        self.numeric_columns = [
            'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total',
            'distance', 'accuracy', 'gir', 'scrambling', 'putts_per_round',
            'finish_position', 'total_score', 'rounds_played'
        ]
    
    def clean_tournament_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw tournament data.
        
        Args:
            df: Raw tournament data
            
        Returns:
            Cleaned tournament data
        """
        print(f"Cleaning tournament data: {df.shape}")
        
        # Make a copy
        cleaned_df = df.copy()
        
        # Remove completely empty rows
        cleaned_df = cleaned_df.dropna(how='all')
        
        # Check for required columns
        missing_required = [col for col in self.required_columns if col not in cleaned_df.columns]
        if missing_required:
            print(f"Warning: Missing required columns: {missing_required}")
        
        # Clean player names and IDs
        if 'player_name' in cleaned_df.columns:
            cleaned_df['player_name'] = cleaned_df['player_name'].str.strip()
            cleaned_df = cleaned_df[cleaned_df['player_name'].notna()]
        
        # Ensure numeric columns are numeric
        for col in self.numeric_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Create made_cut indicator
        if 'finish_position' in cleaned_df.columns:
            # Assume made cut if finish position is not null and reasonable
            cleaned_df['made_cut'] = (
                cleaned_df['finish_position'].notna() & 
                (cleaned_df['finish_position'] > 0) &
                (cleaned_df['finish_position'] <= 200)  # Reasonable finish position
            ).astype(int)
        
        # Clean finish positions
        if 'finish_position' in cleaned_df.columns:
            # Remove unrealistic finish positions
            cleaned_df.loc[cleaned_df['finish_position'] <= 0, 'finish_position'] = np.nan
            cleaned_df.loc[cleaned_df['finish_position'] > 200, 'finish_position'] = np.nan
        
        # Clean strokes gained data
        for sg_col in ['sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total']:
            if sg_col in cleaned_df.columns:
                # Remove extreme outliers (beyond reasonable golf performance)
                q1 = cleaned_df[sg_col].quantile(0.01)
                q99 = cleaned_df[sg_col].quantile(0.99)
                cleaned_df.loc[cleaned_df[sg_col] < q1, sg_col] = np.nan
                cleaned_df.loc[cleaned_df[sg_col] > q99, sg_col] = np.nan
        
        # Clean traditional stats
        if 'accuracy' in cleaned_df.columns:
            # Driving accuracy should be between 0 and 1 (or 0 and 100 if percentage)
            if cleaned_df['accuracy'].max() > 1:
                cleaned_df['accuracy'] = cleaned_df['accuracy'] / 100
            cleaned_df.loc[cleaned_df['accuracy'] < 0, 'accuracy'] = np.nan
            cleaned_df.loc[cleaned_df['accuracy'] > 1, 'accuracy'] = np.nan
        
        if 'gir' in cleaned_df.columns:
            # GIR should be between 0 and 1 (or 0 and 100 if percentage)
            if cleaned_df['gir'].max() > 1:
                cleaned_df['gir'] = cleaned_df['gir'] / 100
            cleaned_df.loc[cleaned_df['gir'] < 0, 'gir'] = np.nan
            cleaned_df.loc[cleaned_df['gir'] > 1, 'gir'] = np.nan
        
        if 'scrambling' in cleaned_df.columns:
            # Scrambling should be between 0 and 1 (or 0 and 100 if percentage)
            if cleaned_df['scrambling'].max() > 1:
                cleaned_df['scrambling'] = cleaned_df['scrambling'] / 100
            cleaned_df.loc[cleaned_df['scrambling'] < 0, 'scrambling'] = np.nan
            cleaned_df.loc[cleaned_df['scrambling'] > 1, 'scrambling'] = np.nan
        
        if 'distance' in cleaned_df.columns:
            # Driving distance should be reasonable (200-350 yards typically)
            cleaned_df.loc[cleaned_df['distance'] < 200, 'distance'] = np.nan
            cleaned_df.loc[cleaned_df['distance'] > 400, 'distance'] = np.nan
        
        # Create date column if not exists
        if 'date' not in cleaned_df.columns and 'year' in cleaned_df.columns:
            # Create a basic date (we'll need to enhance this with actual tournament dates)
            cleaned_df['date'] = pd.to_datetime(cleaned_df['year'].astype(str) + '-01-01')
        
        # Remove rows with no useful data
        key_cols = ['finish_position', 'sg_total', 'total_score']
        available_key_cols = [col for col in key_cols if col in cleaned_df.columns]
        if available_key_cols:
            cleaned_df = cleaned_df.dropna(subset=available_key_cols, how='all')
        
        print(f"Cleaned data shape: {cleaned_df.shape}")
        return cleaned_df
    
    def standardize_player_ids(self, df: pd.DataFrame, 
                             players_df: pd.DataFrame) -> pd.DataFrame:
        """Standardize player IDs across datasets.
        
        Args:
            df: Tournament data
            players_df: Player list with standardized IDs
            
        Returns:
            Data with standardized player IDs
        """
        print("Standardizing player IDs...")
        
        # Create player name to ID mapping
        if 'player_name' in players_df.columns and 'player_id' in players_df.columns:
            name_to_id = dict(zip(players_df['player_name'], players_df['player_id']))
            
            # Map player names to standardized IDs
            if 'player_name' in df.columns:
                df['standardized_player_id'] = df['player_name'].map(name_to_id)
                
                # Use standardized ID where available, otherwise keep original
                if 'player_id' in df.columns:
                    df['player_id'] = df['standardized_player_id'].fillna(df['player_id'])
                else:
                    df['player_id'] = df['standardized_player_id']
                
                # Remove temporary column
                df = df.drop('standardized_player_id', axis=1)
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for modeling.
        
        Args:
            df: Tournament data
            
        Returns:
            Data with target variables
        """
        print("Creating target variables...")
        
        if 'finish_position' in df.columns:
            # Binary targets
            df['won'] = (df['finish_position'] == 1).astype(int)
            df['top_5'] = (df['finish_position'] <= 5).astype(int)
            df['top_10'] = (df['finish_position'] <= 10).astype(int)
            df['top_20'] = (df['finish_position'] <= 20).astype(int)
            
            # Continuous target (inverse of finish position for better modeling)
            df['performance_score'] = 1 / (df['finish_position'] + 1)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Data with missing values
            strategy: Strategy for handling missing values ('median', 'mean', 'drop')
            
        Returns:
            Data with missing values handled
        """
        print(f"Handling missing values with strategy: {strategy}")
        
        if strategy == 'drop':
            # Drop rows with too many missing values
            threshold = len(df.columns) * 0.5  # Drop if more than 50% missing
            df = df.dropna(thresh=threshold)
        
        elif strategy in ['median', 'mean']:
            # Fill numeric columns with median/mean
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if df[col].isna().sum() > 0:
                    if strategy == 'median':
                        fill_value = df[col].median()
                    else:
                        fill_value = df[col].mean()
                    
                    df[col] = df[col].fillna(fill_value)
            
            # Fill categorical columns with mode
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isna().sum() > 0:
                    mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
        
        return df
    
    def remove_outliers(self, df: pd.DataFrame, 
                       method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
        """Remove outliers from numeric columns.
        
        Args:
            df: Data potentially containing outliers
            method: Method for outlier detection ('iqr' or 'zscore')
            factor: Factor for outlier threshold
            
        Returns:
            Data with outliers removed
        """
        print(f"Removing outliers using {method} method...")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.Series([False] * len(df))
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > factor
            
            outlier_mask = outlier_mask | col_outliers.fillna(False)
        
        print(f"Removing {outlier_mask.sum()} outlier rows")
        return df[~outlier_mask]
    
    def process_all_data(self, raw_data_dir: str = 'data/raw', 
                        output_dir: str = 'data/processed') -> Dict[str, pd.DataFrame]:
        """Process all raw data files.
        
        Args:
            raw_data_dir: Directory containing raw data
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary of processed DataFrames
        """
        import os
        
        print("Processing all data files...")
        os.makedirs(output_dir, exist_ok=True)
        
        processed_data = {}
        
        # Load and process each data file
        data_files = {
            'players': 'players.csv',
            'us_open': 'us_open_history.csv',
            'majors': 'major_championships.csv',
            'pga_tour': 'pga_tour_recent.csv',
            'rankings': 'current_rankings.csv',
            'skills': 'current_skills.csv'
        }
        
        for key, filename in data_files.items():
            filepath = os.path.join(raw_data_dir, filename)
            
            if os.path.exists(filepath):
                print(f"Processing {filename}...")
                df = pd.read_csv(filepath)
                
                if key in ['us_open', 'majors', 'pga_tour']:
                    # Tournament data needs full cleaning
                    df = self.clean_tournament_data(df)
                    df = self.create_target_variables(df)
                    df = self.handle_missing_values(df)
                    
                    # Standardize player IDs if players data is available
                    if 'players' in processed_data:
                        df = self.standardize_player_ids(df, processed_data['players'])
                
                processed_data[key] = df
                
                # Save processed data
                output_path = os.path.join(output_dir, f'processed_{filename}')
                df.to_csv(output_path, index=False)
                print(f"Saved processed data to {output_path}")
            
            else:
                print(f"File not found: {filepath}")
        
        return processed_data
