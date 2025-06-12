"""
2025 US Open prediction system.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_collection.datagolf_client import DataGolfClient
from preprocessing.feature_engineering import GolfFeatureEngineer
from preprocessing.data_cleaner import GolfDataCleaner
from modeling.tournament_predictor import TournamentPredictor


class USOpen2025Predictor:
    """Complete prediction system for the 2025 US Open."""
    
    def __init__(self, data_dir: str = 'data'):
        """Initialize the US Open predictor.
        
        Args:
            data_dir: Base directory for data files
        """
        self.data_dir = data_dir
        self.raw_data_dir = os.path.join(data_dir, 'raw')
        self.processed_data_dir = os.path.join(data_dir, 'processed')
        self.predictions_dir = os.path.join(data_dir, 'predictions')
        
        # Create directories
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.predictions_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize components
        self.client = DataGolfClient()
        self.feature_engineer = GolfFeatureEngineer()
        self.data_cleaner = GolfDataCleaner()
        self.predictor = TournamentPredictor()
        
        # US Open specific information
        self.us_open_2025_info = {
            'event_name': 'US Open',
            'year': 2025,
            'course': 'Oakmont Country Club',
            'dates': 'June 12-15, 2025'
        }
    
    def load_processed_data(self) -> Dict[str, pd.DataFrame]:
        """Load all processed data files.

        Returns:
            Dictionary of processed DataFrames
        """
        print("Loading processed data...")

        data = {}

        # Define data files to load
        data_files = {
            'players': 'processed_players.csv',
            'us_open_history': 'processed_us_open_history.csv',
            'major_championships': 'processed_major_championships.csv',
            'pga_tour_recent': 'processed_pga_tour_recent.csv',
            'current_rankings': 'processed_current_rankings.csv',
            'current_skills': 'processed_current_skills.csv'
        }

        for key, filename in data_files.items():
            filepath = os.path.join(self.processed_data_dir, filename)

            if os.path.exists(filepath):
                data[key] = pd.read_csv(filepath)
                print(f"Loaded {key}: {data[key].shape}")
            else:
                print(f"File not found: {filepath}")
                data[key] = pd.DataFrame()

        # If we don't have historical data, create demo data
        if data['us_open_history'].empty and data['major_championships'].empty:
            print("No historical data found. Creating demo data for analysis...")
            data = self._create_demo_data(data)

        return data

    def _create_demo_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create demo historical data for analysis when real data is not available.

        Args:
            data: Existing data dictionary

        Returns:
            Updated data dictionary with demo historical data
        """
        import numpy as np
        from datetime import datetime, timedelta

        # Get current players and rankings for demo
        if not data['players'].empty and not data['current_rankings'].empty:
            # Merge players with rankings to get top players
            rankings_df = data['current_rankings'].copy()
            if 'rankings' in rankings_df.columns:
                # Extract player info from rankings column if it's nested
                rankings_list = []
                for _, row in rankings_df.iterrows():
                    if isinstance(row['rankings'], str):
                        import json
                        try:
                            ranking_data = json.loads(row['rankings'].replace("'", '"'))
                            rankings_list.append(ranking_data)
                        except:
                            continue
                    elif isinstance(row['rankings'], dict):
                        rankings_list.append(row['rankings'])

                if rankings_list:
                    rankings_df = pd.DataFrame(rankings_list)

            # Create demo US Open history (last 5 years)
            us_open_history = []
            years = [2019, 2020, 2021, 2022, 2023, 2024]

            # Get top 150 players for demo
            top_players = rankings_df.head(150) if not rankings_df.empty else data['players'].head(150)

            for year in years:
                # Simulate tournament results
                field_size = min(156, len(top_players))
                year_players = top_players.head(field_size).copy()

                # Add some randomness to simulate different performance levels
                np.random.seed(year)  # For reproducible results

                for idx, player_row in year_players.iterrows():
                    player_id = player_row.get('dg_id', player_row.get('player_id', idx))
                    player_name = player_row.get('player_name', f'Player {idx}')

                    # Simulate tournament performance
                    base_skill = player_row.get('datagolf_rank', idx) if 'datagolf_rank' in player_row else idx
                    skill_factor = 1 / (1 + base_skill * 0.01)  # Better rank = higher skill

                    # Simulate rounds (some players miss cut)
                    made_cut = np.random.random() > 0.35  # ~65% make cut
                    rounds_played = 4 if made_cut else 2

                    total_score = 0
                    for round_num in range(1, rounds_played + 1):
                        # Simulate round score (par is ~70-72 for US Open)
                        par = 71
                        skill_adjustment = np.random.normal(0, 3) * (1 - skill_factor)
                        difficulty_adjustment = np.random.normal(2, 2)  # US Open is tough
                        round_score = par + skill_adjustment + difficulty_adjustment
                        total_score += round_score

                        us_open_history.append({
                            'year': year,
                            'date': f'{year}-06-15',  # US Open typically in June
                            'player_id': player_id,
                            'player_name': player_name,
                            'round': round_num,
                            'score': int(round_score),
                            'total_score': int(total_score),
                            'made_cut': made_cut,
                            'tournament': 'US Open'
                        })

                    # Add final position
                    if made_cut:
                        # Simulate final position based on total score
                        position = max(1, int(np.random.exponential(20) * skill_factor))
                        won = position == 1
                        top_5 = position <= 5
                        top_10 = position <= 10
                        top_20 = position <= 20
                    else:
                        position = None
                        won = top_5 = top_10 = top_20 = False

                    # Add summary record
                    us_open_history.append({
                        'year': year,
                        'date': f'{year}-06-15',  # US Open typically in June
                        'player_id': player_id,
                        'player_name': player_name,
                        'round': 'final',
                        'score': int(total_score) if made_cut else None,
                        'total_score': int(total_score) if made_cut else None,
                        'position': position,
                        'made_cut': made_cut,
                        'won': won,
                        'top_5': top_5,
                        'top_10': top_10,
                        'top_20': top_20,
                        'tournament': 'US Open'
                    })

            data['us_open_history'] = pd.DataFrame(us_open_history)
            print(f"Created demo US Open history: {data['us_open_history'].shape}")

            # Create demo major championships data (simplified)
            major_data = data['us_open_history'].copy()
            major_data['major_type'] = 'US Open'
            data['major_championships'] = major_data
            print(f"Created demo major championships data: {data['major_championships'].shape}")

            # Create demo recent PGA Tour data
            pga_data = []
            recent_years = [2022, 2023, 2024]

            for year in recent_years:
                for event_num in range(1, 25):  # ~24 events per year
                    event_name = f"PGA Event {event_num} {year}"

                    # Sample of players for each event
                    event_field = top_players.sample(min(144, len(top_players)), random_state=year*100+event_num)

                    for idx, player_row in event_field.iterrows():
                        player_id = player_row.get('dg_id', player_row.get('player_id', idx))
                        player_name = player_row.get('player_name', f'Player {idx}')

                        # Simulate performance
                        base_skill = player_row.get('datagolf_rank', idx) if 'datagolf_rank' in player_row else idx
                        skill_factor = 1 / (1 + base_skill * 0.01)

                        made_cut = np.random.random() > 0.3
                        if made_cut:
                            total_score = int(np.random.normal(280, 8) - skill_factor * 5)
                            position = max(1, int(np.random.exponential(15) * skill_factor))
                        else:
                            total_score = None
                            position = None

                        pga_data.append({
                            'year': year,
                            'event': event_name,
                            'player_id': player_id,
                            'player_name': player_name,
                            'total_score': total_score,
                            'position': position,
                            'made_cut': made_cut,
                            'earnings': np.random.randint(5000, 500000) if made_cut else 0
                        })

            data['pga_tour_recent'] = pd.DataFrame(pga_data)
            print(f"Created demo PGA Tour data: {data['pga_tour_recent'].shape}")

        return data

    def get_current_field(self) -> pd.DataFrame:
        """Get the current field for the US Open (or simulate it).
        
        Returns:
            DataFrame with current field information
        """
        print("Getting current US Open field...")
        
        try:
            # Try to get current field from DataGolf API
            field_df = self.client.get_pre_tournament_predictions(tour='pga')
            
            if not field_df.empty:
                print(f"Retrieved current field: {len(field_df)} players")
                return field_df
            
        except Exception as e:
            print(f"Could not retrieve current field: {e}")
        
        # Fallback: use top players from rankings
        print("Using top-ranked players as field...")
        
        try:
            rankings_df = self.client.get_dg_rankings()
            
            # Take top 156 players (typical US Open field size)
            field_df = rankings_df.head(156).copy()
            field_df['in_field'] = True
            
            print(f"Created simulated field: {len(field_df)} players")
            return field_df
            
        except Exception as e:
            print(f"Could not create field from rankings: {e}")
            return pd.DataFrame()
    
    def create_prediction_features(self, field_df: pd.DataFrame, 
                                 historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features for the current field.
        
        Args:
            field_df: Current tournament field
            historical_data: Historical tournament data
            
        Returns:
            DataFrame with prediction features
        """
        print("Creating prediction features...")
        
        # Combine all historical tournament data
        tournament_data_frames = []
        
        for key in ['us_open_history', 'major_championships', 'pga_tour_recent']:
            if key in historical_data and not historical_data[key].empty:
                df = historical_data[key].copy()
                df['data_source'] = key
                tournament_data_frames.append(df)
        
        if not tournament_data_frames:
            print("No historical tournament data available")
            return pd.DataFrame()
        
        combined_tournament_data = pd.concat(tournament_data_frames, ignore_index=True)
        print(f"Combined tournament data: {combined_tournament_data.shape}")
        
        # Get current rankings and skills
        current_rankings = historical_data.get('current_rankings', pd.DataFrame())
        current_skills = historical_data.get('current_skills', pd.DataFrame())
        
        # Create features using the feature engineer
        features_df = self.feature_engineer.combine_all_features(
            tournament_data=combined_tournament_data,
            current_rankings=current_rankings,
            current_skills=current_skills
        )
        
        # Merge with field information
        if 'player_id' in field_df.columns:
            features_df = features_df.merge(
                field_df[['player_id', 'player_name']], 
                on='player_id', 
                how='inner'
            )
        
        print(f"Final features shape: {features_df.shape}")
        return features_df
    
    def train_models(self, historical_data: Dict[str, pd.DataFrame]) -> Dict:
        """Train prediction models on historical data.
        
        Args:
            historical_data: Historical tournament data
            
        Returns:
            Dictionary of trained models
        """
        print("Training prediction models...")
        
        # Combine historical data for training
        training_data_frames = []
        
        # Use US Open history as primary training data
        if 'us_open_history' in historical_data and not historical_data['us_open_history'].empty:
            us_open_data = historical_data['us_open_history'].copy()
            us_open_data['weight'] = 2.0  # Higher weight for US Open data
            training_data_frames.append(us_open_data)
        
        # Add major championship data
        if 'major_championships' in historical_data and not historical_data['major_championships'].empty:
            majors_data = historical_data['major_championships'].copy()
            majors_data['weight'] = 1.5  # Medium weight for other majors
            training_data_frames.append(majors_data)
        
        # Add recent PGA Tour data (lower weight)
        if 'pga_tour_recent' in historical_data and not historical_data['pga_tour_recent'].empty:
            pga_data = historical_data['pga_tour_recent'].copy()
            pga_data['weight'] = 1.0  # Standard weight for regular events
            training_data_frames.append(pga_data)
        
        if not training_data_frames:
            print("No training data available")
            return {}
        
        combined_training_data = pd.concat(training_data_frames, ignore_index=True)
        print(f"Training data shape: {combined_training_data.shape}")
        
        # Create features for training
        # Get unique player IDs from training data
        player_ids = combined_training_data['player_id'].unique().tolist()

        # Use a default course profile for US Open (Oakmont-style)
        us_open_course_profile = {
            'length_type': 'long',
            'driving_skill': 'accuracy',
            'approach_skill': 'precision',
            'scrambling_difficulty': 'high'
        }

        training_features = self.feature_engineer.combine_all_features(
            historical_data=combined_training_data,
            player_ids=player_ids,
            course_id=1,  # Default course ID for US Open
            course_profile=us_open_course_profile
        )
        
        # Merge with target variables
        target_cols = ['won', 'top_5', 'top_10', 'top_20', 'finish_position', 'made_cut']
        available_targets = [col for col in target_cols if col in combined_training_data.columns]
        
        if available_targets:
            training_targets = combined_training_data[['player_id'] + available_targets]
            training_features = training_features.merge(training_targets, on='player_id', how='left')
        
        # Train models
        models = self.predictor.train_multiple_targets(training_features)
        
        return models

    def create_prediction_features(self, field_df: pd.DataFrame, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features for the current tournament field.

        Args:
            field_df: Current tournament field
            historical_data: Historical data dictionary

        Returns:
            Features DataFrame for prediction
        """
        print("Creating prediction features...")

        # Get player IDs from field
        player_ids = field_df['player_id'].unique().tolist() if 'player_id' in field_df.columns else field_df.index.tolist()

        # Combine historical data for feature engineering
        historical_frames = []
        for key, df in historical_data.items():
            if not df.empty and key in ['us_open_history', 'major_championships', 'pga_tour_recent']:
                historical_frames.append(df)

        if not historical_frames:
            print("No historical data available for feature creation")
            return pd.DataFrame()

        combined_historical = pd.concat(historical_frames, ignore_index=True)

        # Use US Open course profile
        us_open_course_profile = {
            'length_type': 'long',
            'driving_skill': 'accuracy',
            'approach_skill': 'precision',
            'scrambling_difficulty': 'high'
        }

        # Create features using feature engineer
        features_df = self.feature_engineer.combine_all_features(
            historical_data=combined_historical,
            player_ids=player_ids,
            course_id=1,  # US Open course ID
            course_profile=us_open_course_profile
        )

        # Add player names if available
        if 'player_name' in field_df.columns:
            player_names = field_df[['player_id', 'player_name']].drop_duplicates()
            features_df = features_df.merge(player_names, on='player_id', how='left')

        print(f"Created features for {len(features_df)} players")
        return features_df

    def make_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions for the US Open field.
        
        Args:
            features_df: Features for current field
            
        Returns:
            DataFrame with predictions
        """
        print("Making US Open predictions...")
        
        predictions_list = []
        
        # Define targets to predict
        targets = ['won', 'top_5', 'top_10', 'top_20']
        
        for target in targets:
            if target in self.predictor.models:
                try:
                    # Use the best performing model (try XGBoost first)
                    model_types = ['xgboost', 'random_forest']  # lightgbm removed due to installation issues
                    
                    for model_type in model_types:
                        if model_type in self.predictor.models[target]:
                            pred_df = self.predictor.predict_tournament(
                                features_df, target=target, model_type=model_type
                            )
                            
                            # Rename columns to include model type
                            pred_df = pred_df.rename(columns={
                                f'{target}_prediction': f'{target}_pred_{model_type}',
                                f'{target}_probability': f'{target}_prob_{model_type}'
                            })
                            
                            predictions_list.append(pred_df)
                            break
                    
                except Exception as e:
                    print(f"Error making predictions for {target}: {e}")
        
        if predictions_list:
            # Combine all predictions
            final_predictions = predictions_list[0]
            for pred_df in predictions_list[1:]:
                final_predictions = final_predictions.merge(
                    pred_df, on='player_id', how='outer'
                )
            
            # Add player names if available
            if 'player_name' in features_df.columns:
                player_names = features_df[['player_id', 'player_name']].drop_duplicates()
                final_predictions = final_predictions.merge(
                    player_names, on='player_id', how='left'
                )
            
            return final_predictions
        else:
            print("No predictions generated")
            return pd.DataFrame()
    
    def generate_us_open_report(self, predictions_df: pd.DataFrame) -> str:
        """Generate a comprehensive US Open prediction report.
        
        Args:
            predictions_df: Predictions DataFrame
            
        Returns:
            Report string
        """
        report = []
        report.append("=" * 60)
        report.append("2025 US OPEN PREDICTION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Course: {self.us_open_2025_info['course']}")
        report.append(f"Dates: {self.us_open_2025_info['dates']}")
        report.append("")
        
        if predictions_df.empty:
            report.append("No predictions available.")
            return "\n".join(report)
        
        # Top win candidates
        win_col = [col for col in predictions_df.columns if 'won_prob' in col]
        if win_col:
            report.append("TOP WIN CANDIDATES:")
            report.append("-" * 30)
            top_winners = predictions_df.nlargest(10, win_col[0])
            
            for i, (_, row) in enumerate(top_winners.iterrows(), 1):
                player_name = row.get('player_name', f"Player {row['player_id']}")
                win_prob = row[win_col[0]] * 100
                report.append(f"{i:2d}. {player_name:<25} {win_prob:5.1f}%")
            report.append("")
        
        # Top 10 candidates
        top10_col = [col for col in predictions_df.columns if 'top_10_prob' in col]
        if top10_col:
            report.append("TOP 10 CANDIDATES:")
            report.append("-" * 30)
            top_10_candidates = predictions_df.nlargest(20, top10_col[0])
            
            for i, (_, row) in enumerate(top_10_candidates.iterrows(), 1):
                player_name = row.get('player_name', f"Player {row['player_id']}")
                top10_prob = row[top10_col[0]] * 100
                report.append(f"{i:2d}. {player_name:<25} {top10_prob:5.1f}%")
            report.append("")
        
        # Summary statistics
        report.append("PREDICTION SUMMARY:")
        report.append("-" * 30)
        
        for col in predictions_df.columns:
            if 'prob' in col:
                target = col.replace('_prob_xgboost', '').replace('_prob_lightgbm', '').replace('_prob_random_forest', '')
                mean_prob = predictions_df[col].mean() * 100
                max_prob = predictions_df[col].max() * 100
                report.append(f"{target.upper():<15} Mean: {mean_prob:5.1f}%  Max: {max_prob:5.1f}%")
        
        return "\n".join(report)
    
    def run_full_prediction(self) -> Tuple[pd.DataFrame, str]:
        """Run the complete US Open prediction process.
        
        Returns:
            Tuple of (predictions DataFrame, report string)
        """
        print("Starting 2025 US Open prediction process...")
        
        # Load processed data
        historical_data = self.load_processed_data()
        
        # Get current field
        field_df = self.get_current_field()
        
        if field_df.empty:
            print("Could not get tournament field")
            return pd.DataFrame(), "Error: Could not get tournament field"
        
        # Train models
        models = self.train_models(historical_data)
        
        if not models:
            print("Could not train models")
            return pd.DataFrame(), "Error: Could not train models"
        
        # Create features for current field
        features_df = self.create_prediction_features(field_df, historical_data)
        
        if features_df.empty:
            print("Could not create features")
            return pd.DataFrame(), "Error: Could not create features"
        
        # Make predictions
        predictions_df = self.make_predictions(features_df)
        
        # Generate report
        report = self.generate_us_open_report(predictions_df)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        predictions_file = os.path.join(self.predictions_dir, f'us_open_2025_predictions_{timestamp}.csv')
        predictions_df.to_csv(predictions_file, index=False)
        
        report_file = os.path.join(self.predictions_dir, f'us_open_2025_report_{timestamp}.txt')
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Predictions saved to: {predictions_file}")
        print(f"Report saved to: {report_file}")
        
        return predictions_df, report


if __name__ == "__main__":
    predictor = USOpen2025Predictor()
    predictions, report = predictor.run_full_prediction()
    
    print("\n" + report)
