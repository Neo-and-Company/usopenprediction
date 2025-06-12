import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class GolfFeatureEngineer:
    """Creates predictive features from raw golf tournament data."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.strokes_gained_cols = [
            'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total'
        ]
        self.traditional_stats_cols = [
            'distance', 'accuracy', 'gir', 'scrambling', 'putts_per_round'
        ]

    def create_player_form_features(self, df: pd.DataFrame,
                                  lookback_periods: List[int] = [4, 8, 16]) -> pd.DataFrame:
        """Create recent form features for each player."""
        print("Creating player form features...")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # This operation is much faster without iterating through each player
        df_sorted = df.sort_values(['player_id', 'date'])
        
        all_features = []
        for period in lookback_periods:
            # Group by player and apply rolling window
            grouped = df_sorted.groupby('player_id')
            
            # Strokes Gained rolling features
            for col in self.strokes_gained_cols:
                if col in df.columns:
                    df_sorted[f'{col}_avg_{period}t'] = grouped[col].transform(lambda x: x.rolling(window=period, min_periods=1).mean().shift(1))
                    df_sorted[f'{col}_std_{period}t'] = grouped[col].transform(lambda x: x.rolling(window=period, min_periods=1).std().shift(1))
            
            # Traditional stats rolling features
            for col in self.traditional_stats_cols:
                 if col in df.columns:
                    df_sorted[f'{col}_avg_{period}t'] = grouped[col].transform(lambda x: x.rolling(window=period, min_periods=1).mean().shift(1))

            # Results rolling features
            if 'finish_position' in df.columns:
                df_sorted[f'avg_finish_{period}t'] = grouped['finish_position'].transform(lambda x: x.rolling(window=period, min_periods=1).mean().shift(1))
                df_sorted[f'top10_rate_{period}t'] = grouped['finish_position'].transform(lambda x: (x <= 10).rolling(window=period, min_periods=1).mean().shift(1))
                df_sorted[f'top20_rate_{period}t'] = grouped['finish_position'].transform(lambda x: (x <= 20).rolling(window=period, min_periods=1).mean().shift(1))

            if 'made_cut' in df.columns:
                df_sorted[f'mc_rate_{period}t'] = grouped['made_cut'].transform(lambda x: x.rolling(window=period, min_periods=1).mean().shift(1))
        
        # We only need the latest entry for each player for prediction
        # The above creates features for every row in history, useful for training
        # For a prediction script, you would typically take the last row for each player
        return df_sorted

    def create_course_history_features(self, df: pd.DataFrame,
                                     course_id: int = None) -> pd.DataFrame:
        """Create course-specific historical performance features."""
        print("Creating course history features...")
        course_data = df[df.get('course_id') == course_id] if course_id else df
        
        # Group by player and aggregate their history on the course
        agg_dict = {
            'finish_position': ['count', 'mean', 'min', 'max'],
            'made_cut': ['mean']
        }
        for col in self.strokes_gained_cols:
            if col in course_data.columns:
                agg_dict[col] = ['mean']

        player_course_data = course_data.groupby('player_id').agg(agg_dict).reset_index()
        
        # Flatten MultiIndex columns
        player_course_data.columns = ['_'.join(col).strip() if col[1] else col[0] for col in player_course_data.columns.values]
        
        # Rename for clarity
        rename_map = {
            'player_id_': 'player_id',
            'finish_position_count': 'course_appearances',
            'finish_position_mean': 'course_avg_finish',
            'finish_position_min': 'course_best_finish',
            'finish_position_max': 'course_worst_finish',
            'made_cut_mean': 'course_mc_rate'
        }
        for col in self.strokes_gained_cols:
             if f'{col}_mean' in player_course_data.columns:
                rename_map[f'{col}_mean'] = f'course_{col}_avg'
        
        player_course_data = player_course_data.rename(columns=rename_map)

        return player_course_data
    
    def create_major_championship_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to major championship performance."""
        print("Creating major championship features...")
        majors_data = df[df.get('is_major', False) == True]
        
        if majors_data.empty:
            print("No major championship data found.")
            return pd.DataFrame(columns=['player_id'])

        # Aggregate performance in majors
        agg_dict = {
            'finish_position': ['count', 'mean', 'min'],
            'made_cut': ['mean']
        }
        for col in self.strokes_gained_cols:
            if col in majors_data.columns:
                agg_dict[col] = ['mean']
        
        player_majors = majors_data.groupby('player_id').agg(agg_dict).reset_index()
        player_majors.columns = ['_'.join(col).strip() if col[1] else col[0] for col in player_majors.columns.values]

        # Add rates and wins manually
        wins = majors_data[majors_data['finish_position'] == 1].groupby('player_id').size().reset_index(name='major_wins')
        top10s = (majors_data[majors_data['finish_position'] <= 10].groupby('player_id').size() / majors_data.groupby('player_id').size()).reset_index(name='major_top10_rate')
        
        player_majors = player_majors.merge(wins, on='player_id', how='left').fillna(0)
        player_majors = player_majors.merge(top10s, on='player_id', how='left').fillna(0)

        # Rename for clarity
        rename_map = {
            'player_id_': 'player_id',
            'finish_position_count': 'major_appearances',
            'finish_position_mean': 'major_avg_finish',
            'finish_position_min': 'major_best_finish',
            'made_cut_mean': 'major_mc_rate'
        }
        for col in self.strokes_gained_cols:
             if f'{col}_mean' in player_majors.columns:
                rename_map[f'{col}_mean'] = f'major_{col}_avg'

        player_majors = player_majors.rename(columns=rename_map)
        
        return player_majors

    def create_skill_trend_features(self, df: pd.DataFrame, window_size: int = 8) -> pd.DataFrame:
        """Create features comparing short-term vs. long-term form."""
        print("Creating skill trend features (short-term vs long-term form)...")

        trend_features = df[['player_id', 'date']].copy()
        
        # Define short and long lookback periods
        short_period, long_period = 4, 16 

        for col in self.strokes_gained_cols:
            if col in df.columns:
                # Calculate short-term and long-term rolling averages
                short_form = df.groupby('player_id')[col].transform(lambda x: x.rolling(window=short_period, min_periods=1).mean().shift(1))
                long_form = df.groupby('player_id')[col].transform(lambda x: x.rolling(window=long_period, min_periods=1).mean().shift(1))
                
                # The trend is the difference between short-term and long-term form
                trend_features[f'{col}_trend'] = short_form - long_form
        
        return trend_features

    # --- NEW FEATURE CATEGORIES ---

    def create_course_fit_features(self, player_stats_df: pd.DataFrame, course_profile: Dict) -> pd.DataFrame:
        """
        Creates features based on how a player's skills match a course profile.
        
        Args:
            player_stats_df: DataFrame with player's recent performance stats (e.g., from create_player_form_features).
            course_profile: A dictionary defining the course characteristics.
                Example: {'length_type': 'long', 'driving_skill': 'accuracy', 
                          'approach_skill': 'precision', 'scrambling_difficulty': 'high'}
        
        Returns:
            DataFrame with course fit scores.
        """
        print("Creating course fit features...")
        fit_features = player_stats_df[['player_id']].copy()

        # Driving fit score
        if course_profile.get('driving_skill') == 'accuracy':
            fit_features['driving_fit'] = player_stats_df['accuracy_avg_8t']
        elif course_profile.get('driving_skill') == 'distance':
            fit_features['driving_fit'] = player_stats_df['distance_avg_8t']

        # Approach fit score (courses demanding precision favor good iron players)
        if course_profile.get('approach_skill') == 'precision':
            fit_features['approach_fit'] = player_stats_df['sg_app_avg_8t']

        # Scrambling fit score (difficult courses require good scrambling)
        if course_profile.get('scrambling_difficulty') == 'high':
             fit_features['scrambling_fit'] = player_stats_df['sg_arg_avg_8t'] + player_stats_df['scrambling_avg_8t']

        return fit_features.drop_duplicates(subset=['player_id'])


    def create_field_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates features based on performance against field strength.
        Assumes 'field_strength' is a column in the input df.
        
        Args:
            df: Tournament data including a 'field_strength' column.
            
        Returns:
            DataFrame with field strength-adjusted features.
        """
        print("Creating field strength features...")
        if 'field_strength' not in df.columns:
            print("Warning: 'field_strength' column not found. Skipping.")
            return pd.DataFrame(columns=['player_id'])
            
        # Weight performance by the strength of the field
        for col in self.strokes_gained_cols:
            if col in df.columns:
                df[f'weighted_{col}'] = df[col] * df['field_strength']

        agg_dict = {}
        for col in self.strokes_gained_cols:
             if f'weighted_{col}' in df.columns:
                agg_dict[f'weighted_{col}'] = 'mean'
        
        agg_dict['field_strength'] = 'mean' # Average field strength faced

        player_fs_data = df.groupby('player_id').agg(agg_dict).reset_index()
        player_fs_data = player_fs_data.rename(columns={'field_strength': 'avg_field_strength_faced'})

        return player_fs_data

    # --- UPDATED COMBINING FUNCTION ---
    
    def combine_all_features(self, 
                             historical_data: pd.DataFrame,
                             player_ids: List[int],
                             course_id: int,
                             course_profile: Dict) -> pd.DataFrame:
        """
        Combine all feature sets for a specific list of players for an upcoming tournament.
        
        Args:
            historical_data: DataFrame of all past tournament results.
            player_ids: List of player_ids for the upcoming tournament.
            course_id: The ID of the course for the tournament.
            course_profile: The profile dictionary for the course.
            
        Returns:
            A single DataFrame with one row per player, ready for prediction.
        """
        print("Combining all features for prediction...")

        # 1. Player Form & Trends (calculated on the fly from historicals)
        form_features_full = self.create_player_form_features(historical_data)
        trend_features_full = self.create_skill_trend_features(form_features_full)
        
        # Merge form and trends
        form_and_trends = form_features_full.merge(trend_features_full, on=['player_id', 'date'], how='left')

        # Get the most recent stats for each player
        latest_features = form_and_trends.sort_values('date').groupby('player_id').tail(1)
        
        # Filter for players in the current tournament
        model_input_df = latest_features[latest_features['player_id'].isin(player_ids)].copy()

        # 2. Course History
        course_features = self.create_course_history_features(historical_data, course_id)
        model_input_df = model_input_df.merge(course_features, on='player_id', how='left')

        # 3. Major History
        major_features = self.create_major_championship_features(historical_data)
        model_input_df = model_input_df.merge(major_features, on='player_id', how='left')
        
        # 4. Field Strength
        fs_features = self.create_field_strength_features(historical_data)
        model_input_df = model_input_df.merge(fs_features, on='player_id', how='left')

        # 5. Course Fit
        fit_features = self.create_course_fit_features(model_input_df, course_profile)
        model_input_df = model_input_df.merge(fit_features, on='player_id', how='left')
        
        # Handle missing values - a simple strategy is to fill with 0 or the median
        # A more complex strategy would be needed for a production model
        model_input_df = model_input_df.fillna(0)

        print(f"Final combined features shape for prediction: {model_input_df.shape}")
        return model_input_df

# --- Example Usage ---
if __name__ == '__main__':
    # This is a placeholder for your actual data loading
    # You would load your historical tournament data from DataGolf here
    print("Loading mock data for demonstration...")
    # Create a mock DataFrame
    num_players = 50
    num_tournaments = 20
    data = []
    for p_id in range(1, num_players + 1):
        for t_id in range(1, num_tournaments + 1):
            row = {
                'player_id': p_id,
                'tournament_id': t_id,
                'date': pd.to_datetime(f'2023-{t_id // 2 + 1}-01'),
                'course_id': (t_id % 3) + 1,
                'is_major': t_id % 5 == 0,
                'field_strength': np.random.uniform(300, 800),
                'finish_position': np.random.randint(1, 100),
                'made_cut': np.random.choice([0, 1]),
                'sg_putt': np.random.normal(0, 0.5),
                'sg_arg': np.random.normal(0, 0.5),
                'sg_app': np.random.normal(0, 0.8),
                'sg_ott': np.random.normal(0, 0.6),
                'sg_t2g': np.random.normal(0, 1.0),
                'sg_total': np.random.normal(0, 1.5),
                'distance': np.random.normal(295, 10),
                'accuracy': np.random.normal(60, 5),
                'gir': np.random.normal(65, 5),
                'scrambling': np.random.normal(60, 5),
                'putts_per_round': np.random.normal(29, 1)
            }
            data.append(row)
    historical_data = pd.DataFrame(data)

    # Initialize the engineer
    feature_engineer = GolfFeatureEngineer()
    
    # Define the upcoming tournament's profile (e.g., U.S. Open at Oakmont)
    oakmont_id = 1 # Example course_id for Oakmont
    oakmont_profile = {
        'length_type': 'long', 
        'driving_skill': 'accuracy', # Although long, fairways are narrow
        'approach_skill': 'precision', # Greens are treacherous
        'scrambling_difficulty': 'high'
    }
    
    # Get the list of players for the upcoming tournament
    players_in_field = list(range(1, num_players + 1))

    # Generate the final feature set for prediction
    final_prediction_df = feature_engineer.combine_all_features(
        historical_data=historical_data,
        player_ids=players_in_field,
        course_id=oakmont_id,
        course_profile=oakmont_profile
    )
    
    print("\n--- Example Output Row ---")
    print(final_prediction_df.head(1).T)

