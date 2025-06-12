"""
Test script to verify the US Open prediction system works.
"""

import sys
import os
sys.path.append('src')

from data_collection.datagolf_client import DataGolfClient
from preprocessing.data_cleaner import GolfDataCleaner
from preprocessing.feature_engineering import GolfFeatureEngineer
from modeling.tournament_predictor import TournamentPredictor

def test_api_connection():
    """Test DataGolf API connection."""
    print("Testing API connection...")
    
    try:
        client = DataGolfClient()
        
        # Test player list
        players = client.get_player_list()
        print(f"✓ Retrieved {len(players)} players")
        
        # Test current rankings
        rankings = client.get_dg_rankings()
        print(f"✓ Retrieved {len(rankings)} rankings")
        
        # Test current skills
        skills = client.get_skill_ratings()
        print(f"✓ Retrieved {len(skills)} skill ratings")
        
        return True
        
    except Exception as e:
        print(f"✗ API test failed: {e}")
        return False

def test_data_processing():
    """Test data processing components."""
    print("\nTesting data processing...")
    
    try:
        # Create sample data
        import pandas as pd
        import numpy as np
        
        sample_data = pd.DataFrame({
            'player_id': [1, 2, 3, 4, 5],
            'player_name': ['Player A', 'Player B', 'Player C', 'Player D', 'Player E'],
            'event_id': [26, 26, 26, 26, 26],
            'year': [2023, 2023, 2023, 2023, 2023],
            'finish_position': [1, 5, 10, 25, 50],
            'sg_total': [2.5, 1.2, 0.8, -0.5, -1.2],
            'sg_putt': [0.5, 0.2, 0.1, -0.1, -0.3],
            'sg_app': [1.0, 0.5, 0.3, -0.2, -0.4],
            'sg_ott': [0.8, 0.3, 0.2, -0.1, -0.3],
            'distance': [310, 305, 300, 295, 290],
            'accuracy': [0.65, 0.70, 0.68, 0.60, 0.55],
            'gir': [0.72, 0.68, 0.65, 0.60, 0.55]
        })
        
        # Test data cleaner
        cleaner = GolfDataCleaner()
        cleaned_data = cleaner.clean_tournament_data(sample_data)
        print(f"✓ Data cleaning: {sample_data.shape} -> {cleaned_data.shape}")
        
        # Test feature engineering
        engineer = GolfFeatureEngineer()
        form_features = engineer.create_player_form_features(cleaned_data)
        print(f"✓ Feature engineering: {len(form_features)} players with features")
        
        return True
        
    except Exception as e:
        print(f"✗ Data processing test failed: {e}")
        return False

def test_modeling():
    """Test modeling components."""
    print("\nTesting modeling...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample training data
        np.random.seed(42)
        n_samples = 1000
        
        sample_features = pd.DataFrame({
            'sg_total_avg_4t': np.random.normal(0, 1, n_samples),
            'sg_putt_avg_4t': np.random.normal(0, 0.5, n_samples),
            'sg_app_avg_4t': np.random.normal(0, 0.8, n_samples),
            'avg_finish_4t': np.random.uniform(10, 80, n_samples),
            'major_appearances': np.random.poisson(5, n_samples),
            'course_appearances': np.random.poisson(3, n_samples)
        })
        
        # Create target (top 10 finish)
        # Better players (lower avg finish, higher SG) more likely to finish top 10
        prob_top10 = 1 / (1 + np.exp(-(sample_features['sg_total_avg_4t'] * 2 - sample_features['avg_finish_4t'] / 20)))
        sample_features['top_10'] = np.random.binomial(1, prob_top10)
        
        # Test predictor
        predictor = TournamentPredictor()
        X, y = predictor.prepare_features(sample_features, 'top_10')
        
        print(f"✓ Feature preparation: {X.shape} features, {y.shape} targets")
        
        # Train a simple model
        model_info = predictor.train_model(X, y, 'random_forest', 'classification')
        print(f"✓ Model training: {model_info['metrics']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Modeling test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("US Open Prediction System - Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    if test_api_connection():
        tests_passed += 1
    
    if test_data_processing():
        tests_passed += 1
    
    if test_modeling():
        tests_passed += 1
    
    print(f"\nTest Results: {tests_passed}/{total_tests} passed")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! System is ready to use.")
        return True
    else:
        print("✗ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
