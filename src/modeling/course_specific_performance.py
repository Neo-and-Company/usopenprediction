"""
Course-Specific Historical Performance Analysis for Golf Predictions.
Analyzes player performance at specific venues with recency weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class CourseSpecificPerformanceAnalyzer:
    """Analyzes historical performance at specific golf courses."""
    
    def __init__(self, course_name: str = "Oakmont Country Club"):
        """Initialize the course-specific performance analyzer.
        
        Args:
            course_name: Name of the course to analyze
        """
        self.course_name = course_name
        self.recency_decay_factor = 0.85  # Exponential decay for older performances
        self.min_rounds_for_reliability = 4  # Minimum rounds for reliable metrics
        
    def collect_course_historical_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Collect historical performance data for the specific course.
        
        Args:
            player_data: DataFrame with player information
            
        Returns:
            DataFrame with course-specific historical performance
        """
        print(f"Collecting historical data for {self.course_name}...")
        
        # This would typically query DataGolf API for course-specific data
        # For now, we'll simulate realistic historical data based on player profiles
        course_history = []
        
        for _, player in player_data.iterrows():
            player_name = player['player_name']
            dg_rank = player.get('datagolf_rank', 100)
            
            # Simulate historical rounds at Oakmont (realistic data)
            historical_rounds = self._simulate_oakmont_history(player_name, dg_rank)
            
            for round_data in historical_rounds:
                course_history.append({
                    'player_name': player_name,
                    'dg_id': player.get('dg_id', 0),
                    'tournament_name': round_data['tournament'],
                    'year': round_data['year'],
                    'date': round_data['date'],
                    'round_number': round_data['round'],
                    'score': round_data['score'],
                    'relative_to_par': round_data['relative_to_par'],
                    'finish_position': round_data['finish'],
                    'field_size': round_data['field_size'],
                    'course_name': self.course_name
                })
        
        return pd.DataFrame(course_history)
    
    def _simulate_oakmont_history(self, player_name: str, dg_rank: int) -> List[Dict]:
        """Simulate realistic historical performance at Oakmont.
        
        Args:
            player_name: Player name
            dg_rank: DataGolf ranking
            
        Returns:
            List of historical round data
        """
        np.random.seed(hash(player_name) % 2**32)  # Consistent simulation per player
        
        # Determine how many times player has played Oakmont
        if dg_rank <= 10:
            num_tournaments = np.random.randint(3, 6)  # Top players play more
        elif dg_rank <= 50:
            num_tournaments = np.random.randint(1, 4)
        elif dg_rank <= 100:
            num_tournaments = np.random.randint(0, 3)
        else:
            num_tournaments = np.random.randint(0, 2)
        
        if num_tournaments == 0:
            return []
        
        historical_rounds = []
        
        # Simulate tournaments from 2007-2024 (Oakmont hosted US Open in 2007, 2016)
        possible_years = [2007, 2016, 2019, 2022]  # Include some non-US Open events
        tournament_years = np.random.choice(possible_years, 
                                          size=min(num_tournaments, len(possible_years)), 
                                          replace=False)
        
        for year in sorted(tournament_years):
            # Determine tournament type
            if year in [2007, 2016]:
                tournament_name = f"US Open {year}"
                field_size = 156
            else:
                tournament_name = f"Oakmont Invitational {year}"
                field_size = 120
            
            # Player's skill level affects base scoring
            if dg_rank <= 5:
                base_score = np.random.normal(2, 2)  # Elite players
            elif dg_rank <= 20:
                base_score = np.random.normal(4, 3)  # Very good players
            elif dg_rank <= 50:
                base_score = np.random.normal(6, 4)  # Good players
            else:
                base_score = np.random.normal(8, 5)  # Average players
            
            # Course-specific adjustment (some players suit Oakmont better)
            course_fit_adjustment = np.random.normal(0, 2)
            
            # Simulate 4 rounds
            tournament_scores = []
            for round_num in range(1, 5):
                round_variance = np.random.normal(0, 2)
                round_score = int(base_score + course_fit_adjustment + round_variance)
                
                # Ensure realistic bounds
                round_score = max(-5, min(15, round_score))
                
                tournament_scores.append(round_score)
                
                historical_rounds.append({
                    'tournament': tournament_name,
                    'year': year,
                    'date': f"{year}-06-{14 + round_num - 1}",  # Simulate US Open dates
                    'round': round_num,
                    'score': 70 + round_score,  # Oakmont par 70
                    'relative_to_par': round_score,
                    'finish': self._calculate_finish_position(tournament_scores, round_num, field_size),
                    'field_size': field_size
                })
        
        return historical_rounds
    
    def _calculate_finish_position(self, scores: List[int], current_round: int, field_size: int) -> int:
        """Calculate realistic finish position based on scores."""
        total_score = sum(scores[:current_round])
        
        # Rough mapping of score to finish position
        if total_score <= 0:
            return np.random.randint(1, 5)
        elif total_score <= 4:
            return np.random.randint(1, 15)
        elif total_score <= 8:
            return np.random.randint(10, 30)
        elif total_score <= 12:
            return np.random.randint(20, 60)
        else:
            return np.random.randint(40, field_size)
    
    def calculate_course_performance_metrics(self, course_history: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive course-specific performance metrics.
        
        Args:
            course_history: Historical performance data at the course
            
        Returns:
            DataFrame with course performance metrics for each player
        """
        print(f"Calculating course performance metrics for {self.course_name}...")
        
        if course_history.empty:
            return pd.DataFrame()
        
        # Group by player
        player_metrics = []
        
        for player_name in course_history['player_name'].unique():
            player_data = course_history[course_history['player_name'] == player_name].copy()
            
            if len(player_data) == 0:
                continue
            
            # Sort by date for recency weighting
            player_data['date'] = pd.to_datetime(player_data['date'])
            player_data = player_data.sort_values('date')
            
            # Calculate recency weights
            weights = self._calculate_recency_weights(player_data['date'])
            
            # Basic performance metrics
            metrics = {
                'player_name': player_name,
                'dg_id': player_data['dg_id'].iloc[0],
                'total_rounds': len(player_data),
                'total_tournaments': player_data['tournament_name'].nunique(),
                'years_played': player_data['year'].nunique(),
                'most_recent_year': player_data['year'].max(),
                'years_since_last_played': 2025 - player_data['year'].max(),
            }
            
            # Scoring metrics with recency weighting
            scores = player_data['relative_to_par'].values
            metrics.update({
                'avg_score': np.average(scores, weights=weights),
                'best_round': scores.min(),
                'worst_round': scores.max(),
                'scoring_std': np.sqrt(np.average((scores - np.average(scores, weights=weights))**2, weights=weights)),
                'recent_avg_score': np.mean(scores[-4:]) if len(scores) >= 4 else np.mean(scores),
                'early_avg_score': np.mean(scores[:4]) if len(scores) >= 8 else np.mean(scores),
            })
            
            # Tournament-level metrics
            tournament_scores = []
            tournament_finishes = []
            
            for tournament in player_data['tournament_name'].unique():
                tourn_data = player_data[player_data['tournament_name'] == tournament]
                tournament_total = tourn_data['relative_to_par'].sum()
                tournament_scores.append(tournament_total)
                
                final_finish = tourn_data['finish_position'].iloc[-1]  # Final round finish
                tournament_finishes.append(final_finish)
            
            if tournament_scores:
                tourn_weights = self._calculate_recency_weights(
                    player_data.groupby('tournament_name')['date'].max().values
                )
                
                metrics.update({
                    'avg_tournament_score': np.average(tournament_scores, weights=tourn_weights),
                    'best_tournament_score': min(tournament_scores),
                    'worst_tournament_score': max(tournament_scores),
                    'avg_finish_position': np.average(tournament_finishes, weights=tourn_weights),
                    'best_finish': min(tournament_finishes),
                    'made_cut_rate': len([f for f in tournament_finishes if f <= 70]) / len(tournament_finishes),
                    'top_10_rate': len([f for f in tournament_finishes if f <= 10]) / len(tournament_finishes),
                    'top_20_rate': len([f for f in tournament_finishes if f <= 20]) / len(tournament_finishes),
                })
            
            # Trend analysis
            if len(scores) >= 4:
                # Linear trend in recent performances
                recent_scores = scores[-8:] if len(scores) >= 8 else scores
                x = np.arange(len(recent_scores))
                trend_slope = np.polyfit(x, recent_scores, 1)[0]
                metrics['scoring_trend'] = trend_slope
            else:
                metrics['scoring_trend'] = 0
            
            # Reliability score based on sample size and recency
            reliability = min(1.0, len(player_data) / self.min_rounds_for_reliability)
            recency_factor = max(0.3, 1.0 - (metrics['years_since_last_played'] * 0.1))
            metrics['reliability_score'] = reliability * recency_factor
            
            player_metrics.append(metrics)
        
        return pd.DataFrame(player_metrics)

    def _calculate_recency_weights(self, dates) -> np.ndarray:
        """Calculate exponential decay weights based on recency.

        Args:
            dates: Array of dates

        Returns:
            Array of weights (more recent = higher weight)
        """
        if len(dates) == 0:
            return np.array([])

        dates = pd.to_datetime(dates)
        current_date = pd.Timestamp('2025-06-15')  # US Open 2025 date

        # Calculate years since each performance
        years_ago = []
        for date in dates:
            days_diff = (current_date - date).days
            years_ago.append(days_diff / 365.25)

        years_ago = np.array(years_ago)

        # Apply exponential decay
        weights = self.recency_decay_factor ** years_ago

        # Normalize weights to sum to 1
        return weights / weights.sum()

    def handle_missing_course_data(self, player_data: pd.DataFrame,
                                 course_metrics: pd.DataFrame) -> pd.DataFrame:
        """Handle players with no course-specific historical data.

        Args:
            player_data: All player data
            course_metrics: Existing course performance metrics

        Returns:
            Complete metrics with estimated values for missing players
        """
        print("Handling players with missing course-specific data...")

        # Find players without course history
        players_with_history = set(course_metrics['player_name'].unique())
        all_players = set(player_data['player_name'].unique())
        missing_players = all_players - players_with_history

        if not missing_players:
            return course_metrics

        # Calculate baseline metrics from players with history
        baseline_metrics = self._calculate_baseline_metrics(course_metrics)

        # Create estimated metrics for missing players
        missing_player_metrics = []

        for player_name in missing_players:
            player_info = player_data[player_data['player_name'] == player_name].iloc[0]
            dg_rank = player_info.get('datagolf_rank', 100)

            # Estimate performance based on general skill level
            estimated_metrics = self._estimate_course_performance(
                player_name, dg_rank, baseline_metrics
            )

            missing_player_metrics.append(estimated_metrics)

        # Combine existing and estimated metrics
        if missing_player_metrics:
            missing_df = pd.DataFrame(missing_player_metrics)
            complete_metrics = pd.concat([course_metrics, missing_df], ignore_index=True)
        else:
            complete_metrics = course_metrics

        return complete_metrics

    def _calculate_baseline_metrics(self, course_metrics: pd.DataFrame) -> Dict:
        """Calculate baseline performance metrics from existing data."""
        if course_metrics.empty:
            # Default baseline if no historical data exists
            return {
                'avg_score': 5.0,
                'scoring_std': 3.0,
                'avg_tournament_score': 20.0,
                'avg_finish_position': 50.0,
                'made_cut_rate': 0.6,
                'top_10_rate': 0.1,
                'top_20_rate': 0.2
            }

        # Calculate percentile-based baselines
        return {
            'avg_score': course_metrics['avg_score'].median(),
            'scoring_std': course_metrics['scoring_std'].median(),
            'avg_tournament_score': course_metrics['avg_tournament_score'].median(),
            'avg_finish_position': course_metrics['avg_finish_position'].median(),
            'made_cut_rate': course_metrics['made_cut_rate'].median(),
            'top_10_rate': course_metrics['top_10_rate'].median(),
            'top_20_rate': course_metrics['top_20_rate'].median()
        }

    def _estimate_course_performance(self, player_name: str, dg_rank: int,
                                   baseline_metrics: Dict) -> Dict:
        """Estimate course performance for players without historical data.

        Args:
            player_name: Player name
            dg_rank: DataGolf ranking
            baseline_metrics: Baseline performance metrics

        Returns:
            Estimated course performance metrics
        """
        # Adjust baseline based on general skill level
        if dg_rank <= 10:
            skill_adjustment = -2.0  # Elite players score better
            finish_adjustment = 0.7  # Better finishes
        elif dg_rank <= 25:
            skill_adjustment = -1.0
            finish_adjustment = 0.8
        elif dg_rank <= 50:
            skill_adjustment = 0.0
            finish_adjustment = 1.0
        elif dg_rank <= 100:
            skill_adjustment = 1.0
            finish_adjustment = 1.3
        else:
            skill_adjustment = 2.0
            finish_adjustment = 1.6

        return {
            'player_name': player_name,
            'dg_id': 0,  # Unknown for missing players
            'total_rounds': 0,
            'total_tournaments': 0,
            'years_played': 0,
            'most_recent_year': 0,
            'years_since_last_played': 999,  # Never played
            'avg_score': baseline_metrics['avg_score'] + skill_adjustment,
            'best_round': baseline_metrics['avg_score'] + skill_adjustment - 2,
            'worst_round': baseline_metrics['avg_score'] + skill_adjustment + 4,
            'scoring_std': baseline_metrics['scoring_std'],
            'recent_avg_score': baseline_metrics['avg_score'] + skill_adjustment,
            'early_avg_score': baseline_metrics['avg_score'] + skill_adjustment,
            'avg_tournament_score': baseline_metrics['avg_tournament_score'] + (skill_adjustment * 4),
            'best_tournament_score': baseline_metrics['avg_tournament_score'] + (skill_adjustment * 4) - 8,
            'worst_tournament_score': baseline_metrics['avg_tournament_score'] + (skill_adjustment * 4) + 12,
            'avg_finish_position': baseline_metrics['avg_finish_position'] * finish_adjustment,
            'best_finish': max(1, int(baseline_metrics['avg_finish_position'] * finish_adjustment * 0.3)),
            'made_cut_rate': max(0.1, baseline_metrics['made_cut_rate'] / finish_adjustment),
            'top_10_rate': max(0.01, baseline_metrics['top_10_rate'] / finish_adjustment),
            'top_20_rate': max(0.02, baseline_metrics['top_20_rate'] / finish_adjustment),
            'scoring_trend': 0.0,
            'reliability_score': 0.1  # Low reliability for estimated data
        }

    def create_course_performance_features(self, course_metrics: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from course performance metrics.

        Args:
            course_metrics: Course performance metrics

        Returns:
            DataFrame with engineered course performance features
        """
        print("Creating course performance features...")

        if course_metrics.empty:
            return pd.DataFrame()

        features_df = course_metrics.copy()

        # Consistency features
        features_df['consistency_score'] = 1 / (1 + features_df['scoring_std'])
        features_df['tournament_consistency'] = 1 / (1 + (features_df['worst_tournament_score'] - features_df['best_tournament_score']))

        # Performance level features
        features_df['elite_performance_rate'] = features_df['top_10_rate']
        features_df['solid_performance_rate'] = features_df['top_20_rate']
        features_df['reliability_adjusted_avg'] = features_df['avg_score'] * features_df['reliability_score']

        # Recency features
        features_df['recency_factor'] = np.exp(-features_df['years_since_last_played'] * 0.2)
        features_df['recent_form_indicator'] = (features_df['early_avg_score'] - features_df['recent_avg_score']) * features_df['recency_factor']

        # Experience features
        features_df['course_experience'] = np.log1p(features_df['total_rounds'])
        features_df['tournament_experience'] = np.log1p(features_df['total_tournaments'])

        # Improvement/decline features
        features_df['performance_trend'] = -features_df['scoring_trend']  # Negative trend = improvement
        features_df['trend_reliability'] = features_df['performance_trend'] * features_df['reliability_score']

        # Composite scores - FIXED MATHEMATICAL INSTABILITY
        # The original formula (1 / (1 + avg_score)) explodes when avg_score approaches -1
        # Use a stable sigmoid-like transformation instead
        features_df['course_mastery_score'] = (
            (1 / (1 + np.exp(features_df['avg_score'] / 2))) * 0.4 +  # Stable sigmoid transformation
            features_df['consistency_score'] * 0.3 +
            features_df['reliability_score'] * 0.3
        )

        features_df['course_comfort_score'] = (
            features_df['made_cut_rate'] * 0.4 +
            features_df['recency_factor'] * 0.3 +
            (features_df['course_experience'] / 5) * 0.3  # Normalize experience
        )

        return features_df
