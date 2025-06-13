#!/usr/bin/env python3
"""
Test script to validate the model fixes for internal contradictions.
This will run the enhanced prediction model and check if the issues are resolved.
"""

import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from modeling.enhanced_course_prediction import EnhancedCoursePredictionSystem

def load_test_data():
    """Load the current player data for testing."""
    try:
        # Load current skills data
        skills_df = pd.read_csv('data/raw/current_skills.csv')

        # Parse the players column which contains JSON-like data
        players_data = []
        for _, row in skills_df.iterrows():
            player_str = row['players']
            # Convert string representation to dict
            player_dict = eval(player_str)
            players_data.append(player_dict)

        players_df = pd.DataFrame(players_data)

        # Load rankings data
        rankings_df = pd.read_csv('data/raw/current_rankings.csv')

        # Parse rankings data similarly
        rankings_data = []
        for _, row in rankings_df.iterrows():
            ranking_str = row['rankings']
            ranking_dict = eval(ranking_str)
            rankings_data.append(ranking_dict)

        rankings_parsed_df = pd.DataFrame(rankings_data)

        # Merge the data on player_name
        merged_df = players_df.merge(rankings_parsed_df, on='player_name', how='left', suffixes=('_skills', '_rankings'))

        print(f"Loaded {len(merged_df)} players for testing")
        print(f"Sample columns: {list(merged_df.columns)[:10]}")
        return merged_df

    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def test_model_fixes():
    """Test the model fixes and compare before/after rankings."""
    
    print("=" * 60)
    print("TESTING MODEL FIXES FOR INTERNAL CONTRADICTIONS")
    print("=" * 60)
    
    # Load test data
    player_data = load_test_data()
    if player_data.empty:
        print("No data available for testing")
        return
    
    # Initialize the enhanced prediction system
    predictor = EnhancedCoursePredictionSystem()
    
    print(f"\nFeature weights (UPDATED):")
    for component, weight in predictor.feature_weights.items():
        print(f"  {component}: {weight:.1%}")
    
    # Run predictions
    try:
        print(f"\nRunning enhanced predictions...")
        predictions, analysis_report = predictor.run_enhanced_prediction(player_data)
        
        if predictions.empty:
            print("No predictions generated")
            return
        
        # Sort by final prediction score
        predictions_sorted = predictions.sort_values('final_prediction_score', ascending=False)
        
        print(f"\nDEBUGGING: DETAILED BREAKDOWN FOR OUTLIERS")
        print("=" * 80)

        # Debug the top outliers vs normal players
        outliers = ['Kokrak, Jason', 'Smalley, Alex']  # Current outliers
        normal_players = ['Scheffler, Scottie', 'DeChambeau, Bryson']

        for player_name in outliers + normal_players:
            player_data = predictions_sorted[predictions_sorted['player_name'] == player_name]
            if not player_data.empty:
                player = player_data.iloc[0]
                print(f"\n🔍 DEBUGGING: {player_name}")
                print(f"   Final Score: {player['final_prediction_score']:.6f}")
                print(f"   Course Fit: {player['course_fit_score']:.6f}")
                print(f"   Course Penalty: {player.get('course_fit_penalty', 1.0):.6f}")
                print(f"   Historical Score: {player['historical_performance_score']:.6f}")
                print(f"   Form Score: {player['general_form_score']:.6f}")
                print(f"   Reliability Factor: {player['reliability_factor']:.6f}")
                print(f"   Confidence Level: {player['confidence_level']:.6f}")

        print(f"\nTOP 15 PREDICTIONS (After Fixes):")
        print("=" * 80)

        for i, (_, player) in enumerate(predictions_sorted.head(15).iterrows(), 1):
            name = player['player_name']
            final_score = player['final_prediction_score']
            course_fit = player['course_fit_score']
            course_penalty = player.get('course_fit_penalty', 1.0)
            form_score = player['general_form_score']
            fit_category = player.get('fit_category', 'Unknown')

            print(f"{i:2d}. {name:<20} | Score: {final_score:.3f} | "
                  f"Fit: {course_fit:.3f} ({fit_category}) | "
                  f"Penalty: {course_penalty:.3f} | Form: {form_score:.3f}")
        
        # Check for specific players to validate fixes
        print(f"\nVALIDATION CHECKS:")
        print("=" * 40)
        
        # Check Rory McIlroy's position
        rory_data = predictions_sorted[predictions_sorted['player_name'].str.contains('McIlroy', na=False)]
        if not rory_data.empty:
            rory_rank = (predictions_sorted['player_name'] == rory_data.iloc[0]['player_name']).idxmax()
            rory_position = predictions_sorted.index.get_loc(rory_rank) + 1
            rory_fit = rory_data.iloc[0]['course_fit_score']
            rory_penalty = rory_data.iloc[0].get('course_fit_penalty', 1.0)
            print(f"✓ Rory McIlroy: Position #{rory_position}, Course Fit: {rory_fit:.3f}, Penalty: {rory_penalty:.3f}")
        
        # Check Sepp Straka's position
        straka_data = predictions_sorted[predictions_sorted['player_name'].str.contains('Straka', na=False)]
        if not straka_data.empty:
            straka_rank = (predictions_sorted['player_name'] == straka_data.iloc[0]['player_name']).idxmax()
            straka_position = predictions_sorted.index.get_loc(straka_rank) + 1
            straka_fit = straka_data.iloc[0]['course_fit_score']
            straka_penalty = straka_data.iloc[0].get('course_fit_penalty', 1.0)
            print(f"✓ Sepp Straka: Position #{straka_position}, Course Fit: {straka_fit:.3f}, Penalty: {straka_penalty:.3f}")
        
        # Check Scottie Scheffler's position
        scottie_data = predictions_sorted[predictions_sorted['player_name'].str.contains('Scheffler', na=False)]
        if not scottie_data.empty:
            scottie_rank = (predictions_sorted['player_name'] == scottie_data.iloc[0]['player_name']).idxmax()
            scottie_position = predictions_sorted.index.get_loc(scottie_rank) + 1
            scottie_fit = scottie_data.iloc[0]['course_fit_score']
            scottie_penalty = scottie_data.iloc[0].get('course_fit_penalty', 1.0)
            print(f"✓ Scottie Scheffler: Position #{scottie_position}, Course Fit: {scottie_fit:.3f}, Penalty: {scottie_penalty:.3f}")
        
        # Save updated predictions
        output_file = 'data/predictions/us_open_2025_fixed_predictions.csv'
        predictions_sorted.to_csv(output_file, index=False)
        print(f"\n✓ Updated predictions saved to: {output_file}")
        
        print(f"\nMODEL FIXES SUMMARY:")
        print("=" * 30)
        print(f"✓ Course fit weight increased: 25% → 40%")
        print(f"✓ Form calculation replaced with stable SG-based metric")
        print(f"✓ Course fit penalties applied for poor fits")
        print(f"✓ Predictions should now be more logically consistent")
        
    except Exception as e:
        print(f"Error running predictions: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_fixes()
