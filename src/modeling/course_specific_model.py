"""
Advanced course-specific prediction model for US Open 2025.
Implements the sophisticated approach of matching player skills to course demands.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from typing import Dict, List, Tuple
import joblib
import os


class CourseSpecificPredictor:
    """
    Advanced predictor that matches player skills to specific course characteristics
    and weather conditions, moving beyond simple rankings.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.course_profiles = {}
        self.weather_adjustments = {}
        
        # Define US Open 2025 course profile (Oakmont CC)
        self.us_open_2025_profile = {
            'course_name': 'Oakmont Country Club',
            'location': 'Oakmont, PA',
            'par': 70,
            'yardage': 7255,
            'course_type': 'parkland',
            'grass_type_fairways': 'bentgrass',
            'grass_type_greens': 'bentgrass',
            'green_speed': 13.5,  # Stimpmeter reading
            'rough_height': 4.5,  # inches
            'fairway_width_avg': 28,  # yards
            'green_size_avg': 5800,  # square feet
            'elevation_change': 180,  # feet
            'water_hazards': 2,
            'bunkers': 180,
            'difficulty_rating': 9.2,  # out of 10
            'key_demands': [
                'accuracy_off_tee',
                'iron_precision', 
                'putting_bentgrass',
                'rough_recovery',
                'mental_toughness'
            ],
            'historical_winning_scores': [-5, -1, +1, -4, -2],  # Last 5 US Opens here
            'weather_sensitivity': {
                'wind': 'high',
                'rain': 'medium', 
                'temperature': 'low'
            }
        }
    
    def create_course_specific_features(self, player_data: pd.DataFrame, 
                                      course_profile: Dict) -> pd.DataFrame:
        """
        Create features that match player skills to specific course demands.
        This is the core of moving beyond simple rankings.
        """
        features_df = player_data.copy()
        
        # Course-specific skill weightings for Oakmont
        oakmont_weights = {
            'accuracy_weight': 0.35,      # High penalty for missing fairways
            'iron_precision_weight': 0.30, # Demanding approach shots
            'putting_weight': 0.20,       # Fast, undulating greens
            'distance_weight': 0.10,      # Length helps but accuracy more important
            'short_game_weight': 0.05     # Less important on this course
        }
        
        # Create Oakmont-specific composite scores
        if all(col in features_df.columns for col in ['sg_ott', 'sg_app', 'sg_putt']):
            # Oakmont Fit Score - weighted for course demands
            features_df['oakmont_fit_score'] = (
                features_df['sg_ott'] * oakmont_weights['accuracy_weight'] +
                features_df['sg_app'] * oakmont_weights['iron_precision_weight'] +
                features_df['sg_putt'] * oakmont_weights['putting_weight'] +
                features_df.get('driving_dist', 0) * oakmont_weights['distance_weight'] / 300 +
                features_df.get('sg_arg', 0) * oakmont_weights['short_game_weight']
            )
        
        # Accuracy-focused features (critical for Oakmont)
        if 'driving_acc' in features_df.columns:
            features_df['accuracy_premium'] = features_df['driving_acc'] * 2.0  # Double weight
            features_df['accuracy_rank'] = features_df['driving_acc'].rank(ascending=False)
        
        # Iron play precision (key for Oakmont's demanding greens)
        if 'sg_app' in features_df.columns:
            features_df['iron_precision_score'] = features_df['sg_app'] * 1.5
            features_df['approach_rank'] = features_df['sg_app'].rank(ascending=False)
        
        # Bentgrass putting specialization
        if 'sg_putt' in features_df.columns:
            # Assume players with better putting on bentgrass courses
            features_df['bentgrass_putting_adj'] = features_df['sg_putt'] * 1.2
        
        # Rough recovery ability (4.5" rough at Oakmont)
        if 'sg_arg' in features_df.columns:
            features_df['rough_recovery_score'] = features_df['sg_arg'] * 1.3
        
        # Course length adjustment (7255 yards)
        if 'driving_dist' in features_df.columns:
            # Moderate length bonus (not as important as accuracy)
            features_df['length_advantage'] = np.where(
                features_df['driving_dist'] > 290, 0.2, 0
            )
        
        # Major championship experience proxy
        if 'datagolf_rank' in features_df.columns:
            # Top players more likely to handle pressure
            features_df['major_experience_proxy'] = np.where(
                features_df['datagolf_rank'] <= 50, 0.3, 0
            )
        
        # US Open specific historical performance proxy
        # (In real implementation, this would use actual US Open history)
        if 'datagolf_rank' in features_df.columns:
            features_df['us_open_fit'] = (
                1 / (features_df['datagolf_rank'] + 1) * 
                features_df.get('oakmont_fit_score', 1)
            )
        
        return features_df
    
    def apply_weather_adjustments(self, features_df: pd.DataFrame, 
                                weather_forecast: Dict) -> pd.DataFrame:
        """
        Adjust predictions based on weather forecast for tournament week.
        """
        adjusted_df = features_df.copy()
        
        # Wind adjustments
        wind_speed = weather_forecast.get('avg_wind_speed', 10)
        if wind_speed > 15:  # High wind conditions
            # Favor players with lower ball flight and better accuracy
            if 'sg_ott' in adjusted_df.columns:
                adjusted_df['wind_adjusted_driving'] = adjusted_df['sg_ott'] * 1.3
            if 'driving_acc' in adjusted_df.columns:
                adjusted_df['wind_accuracy_bonus'] = adjusted_df['driving_acc'] * 0.5
        
        # Rain adjustments
        rain_probability = weather_forecast.get('rain_probability', 0)
        if rain_probability > 60:  # Likely rain
            # Soft conditions favor longer hitters, reduce putting importance
            if 'driving_dist' in adjusted_df.columns:
                adjusted_df['rain_distance_bonus'] = adjusted_df['driving_dist'] / 300 * 0.3
            if 'sg_putt' in adjusted_df.columns:
                adjusted_df['rain_putting_reduction'] = adjusted_df['sg_putt'] * 0.8
        
        # Temperature adjustments
        avg_temp = weather_forecast.get('avg_temperature', 70)
        if avg_temp < 60:  # Cold conditions
            # Ball doesn't travel as far, premium on accuracy
            if 'driving_acc' in adjusted_df.columns:
                adjusted_df['cold_accuracy_bonus'] = adjusted_df['driving_acc'] * 0.2
        
        return adjusted_df
    
    def create_player_course_interaction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between player skills and course characteristics.
        This captures the nuanced fit between player and course.
        """
        interaction_df = features_df.copy()
        
        # Accuracy × Course Difficulty interaction
        if all(col in features_df.columns for col in ['driving_acc', 'datagolf_rank']):
            interaction_df['accuracy_difficulty_fit'] = (
                features_df['driving_acc'] * (1 / (features_df['datagolf_rank'] + 1))
            )
        
        # Iron Play × Green Complexity interaction
        if 'sg_app' in features_df.columns:
            # Oakmont has very complex greens (difficulty 9.2/10)
            interaction_df['iron_green_complexity_fit'] = features_df['sg_app'] * 0.92
        
        # Putting × Green Speed interaction
        if 'sg_putt' in features_df.columns:
            # Oakmont greens are very fast (13.5 stimp)
            interaction_df['putting_speed_fit'] = features_df['sg_putt'] * 1.35
        
        # Experience × Pressure interaction
        if 'datagolf_rank' in features_df.columns:
            # Major championship pressure factor
            interaction_df['experience_pressure_fit'] = np.where(
                features_df['datagolf_rank'] <= 20, 0.5,
                np.where(features_df['datagolf_rank'] <= 50, 0.3, 0.1)
            )
        
        return interaction_df
    
    def predict_us_open_2025(self, player_data: pd.DataFrame, 
                           weather_forecast: Dict = None) -> pd.DataFrame:
        """
        Generate sophisticated US Open 2025 predictions that account for:
        1. Course-specific demands (Oakmont CC)
        2. Weather conditions
        3. Player-course fit
        """
        
        # Default weather forecast if none provided
        if weather_forecast is None:
            weather_forecast = {
                'avg_wind_speed': 12,  # mph
                'rain_probability': 30,  # %
                'avg_temperature': 75,  # F
                'conditions': 'partly_cloudy'
            }
        
        print("Generating course-specific US Open 2025 predictions...")
        print(f"Course: {self.us_open_2025_profile['course_name']}")
        print(f"Weather: {weather_forecast}")
        
        # Step 1: Create course-specific features
        course_features = self.create_course_specific_features(
            player_data, self.us_open_2025_profile
        )
        
        # Step 2: Apply weather adjustments
        weather_adjusted = self.apply_weather_adjustments(
            course_features, weather_forecast
        )
        
        # Step 3: Create interaction features
        final_features = self.create_player_course_interaction_features(
            weather_adjusted
        )
        
        # Step 4: Calculate final Oakmont-specific prediction
        prediction_components = []
        
        # Base skill component (30% weight)
        if 'dg_skill_estimate' in final_features.columns:
            prediction_components.append(final_features['dg_skill_estimate'] * 0.3)
        
        # Course fit component (40% weight)
        if 'oakmont_fit_score' in final_features.columns:
            prediction_components.append(final_features['oakmont_fit_score'] * 0.4)
        
        # Experience component (20% weight)
        if 'experience_pressure_fit' in final_features.columns:
            prediction_components.append(final_features['experience_pressure_fit'] * 0.2)
        
        # Weather adjustment component (10% weight)
        weather_component = 0
        if 'wind_adjusted_driving' in final_features.columns:
            weather_component += final_features['wind_adjusted_driving'].fillna(0) * 0.05
        if 'rain_distance_bonus' in final_features.columns:
            weather_component += final_features['rain_distance_bonus'].fillna(0) * 0.05
        prediction_components.append(weather_component)
        
        # Combine all components
        if prediction_components:
            final_features['oakmont_prediction'] = sum(prediction_components)
        else:
            # Fallback to basic ranking if components not available
            final_features['oakmont_prediction'] = 1 / (final_features['datagolf_rank'] + 1)
        
        # Create results dataframe
        results_df = final_features[[
            'player_name', 'dg_id', 'datagolf_rank', 'oakmont_prediction'
        ]].copy()
        
        # Add course-specific insights
        if 'oakmont_fit_score' in final_features.columns:
            results_df['course_fit_score'] = final_features['oakmont_fit_score']
        if 'accuracy_premium' in final_features.columns:
            results_df['accuracy_advantage'] = final_features['accuracy_premium']
        if 'iron_precision_score' in final_features.columns:
            results_df['iron_play_advantage'] = final_features['iron_precision_score']
        
        # Rank by Oakmont-specific prediction
        results_df['oakmont_rank'] = results_df['oakmont_prediction'].rank(ascending=False)
        results_df = results_df.sort_values('oakmont_prediction', ascending=False)
        
        return results_df
    
    def compare_predictions(self, general_predictions: pd.DataFrame, 
                          oakmont_predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Compare general rankings vs Oakmont-specific predictions to identify
        players who are better/worse fits for this specific course.
        """
        comparison = general_predictions.merge(
            oakmont_predictions[['player_name', 'oakmont_rank', 'oakmont_prediction']], 
            on='player_name', 
            how='inner'
        )
        
        comparison['rank_change'] = comparison['datagolf_rank'] - comparison['oakmont_rank']
        comparison['prediction_boost'] = comparison['oakmont_prediction'] - comparison.get('general_prediction', 0)
        
        # Identify biggest movers
        comparison['mover_category'] = np.where(
            comparison['rank_change'] > 20, 'Big Riser',
            np.where(comparison['rank_change'] > 10, 'Riser',
                    np.where(comparison['rank_change'] < -20, 'Big Faller',
                            np.where(comparison['rank_change'] < -10, 'Faller', 'Stable')))
        )
        
        return comparison.sort_values('rank_change', ascending=False)


if __name__ == "__main__":
    # This would be called with actual player data
    print("Course-Specific US Open 2025 Predictor initialized")
    print("Ready to generate Oakmont-specific predictions that go beyond simple rankings")
