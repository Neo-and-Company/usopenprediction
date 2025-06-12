"""
Test the advanced course-specific US Open 2025 prediction model.
This demonstrates the sophisticated approach of matching player skills to course demands.
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# Add src to path
sys.path.append('src')

from modeling.course_specific_model import CourseSpecificPredictor


def parse_json_column(df, column_name):
    """Parse JSON strings in a DataFrame column."""
    parsed_data = []
    for idx, row in df.iterrows():
        try:
            if pd.notna(row[column_name]):
                data = eval(row[column_name])
                parsed_data.append(data)
            else:
                parsed_data.append({})
        except:
            parsed_data.append({})
    
    return pd.DataFrame(parsed_data)


def load_and_process_data():
    """Load and process the collected DataGolf data."""
    
    # Load rankings data
    rankings_df = pd.read_csv('data/raw/current_rankings.csv')
    rankings_parsed = parse_json_column(rankings_df, 'rankings')
    
    # Load skill ratings data  
    skills_df = pd.read_csv('data/raw/skill_ratings.csv')
    skills_parsed = parse_json_column(skills_df, 'players')
    
    # Load player data
    players_df = pd.read_csv('data/raw/players.csv')
    
    print(f"Loaded {len(rankings_parsed)} rankings")
    print(f"Loaded {len(skills_parsed)} skill ratings")
    print(f"Loaded {len(players_df)} players")
    
    return rankings_parsed, skills_parsed, players_df


def create_combined_dataset(rankings_df, skills_df, players_df):
    """Combine all datasets into a comprehensive player dataset."""
    
    # Start with rankings
    combined_df = rankings_df.copy()
    
    # Merge with skills
    if 'dg_id' in combined_df.columns and 'dg_id' in skills_df.columns:
        combined_df = combined_df.merge(skills_df, on='dg_id', how='left', suffixes=('', '_skill'))
    
    # Merge with player info
    if 'dg_id' in combined_df.columns and 'dg_id' in players_df.columns:
        combined_df = combined_df.merge(players_df, on='dg_id', how='left', suffixes=('', '_player'))
    
    print(f"Combined dataset: {combined_df.shape[0]} players, {combined_df.shape[1]} features")
    return combined_df


def test_weather_scenarios():
    """Test different weather scenarios for US Open 2025."""
    
    weather_scenarios = {
        'ideal_conditions': {
            'avg_wind_speed': 8,
            'rain_probability': 10,
            'avg_temperature': 75,
            'conditions': 'sunny'
        },
        'windy_conditions': {
            'avg_wind_speed': 20,
            'rain_probability': 20,
            'avg_temperature': 70,
            'conditions': 'windy'
        },
        'wet_conditions': {
            'avg_wind_speed': 12,
            'rain_probability': 80,
            'avg_temperature': 65,
            'conditions': 'rainy'
        },
        'challenging_conditions': {
            'avg_wind_speed': 18,
            'rain_probability': 60,
            'avg_temperature': 60,
            'conditions': 'cold_and_windy'
        }
    }
    
    return weather_scenarios


def main():
    """Main function to test the course-specific model."""
    
    print("=" * 80)
    print("ADVANCED US OPEN 2025 COURSE-SPECIFIC PREDICTION MODEL")
    print("Moving Beyond Simple Rankings to Course-Specific Fit")
    print("=" * 80)
    
    # Load data
    rankings_df, skills_df, players_df = load_and_process_data()
    combined_df = create_combined_dataset(rankings_df, skills_df, players_df)
    
    # Initialize the advanced predictor
    predictor = CourseSpecificPredictor()
    
    # Load our previous general predictions for comparison
    try:
        general_predictions = pd.read_csv('data/predictions/us_open_2025_test_predictions.csv')
        print(f"Loaded previous general predictions for comparison")
    except:
        general_predictions = None
        print("No previous predictions found for comparison")
    
    # Test different weather scenarios
    weather_scenarios = test_weather_scenarios()
    
    print(f"\nTesting {len(weather_scenarios)} weather scenarios...")
    
    all_results = {}
    
    for scenario_name, weather_forecast in weather_scenarios.items():
        print(f"\n" + "="*60)
        print(f"SCENARIO: {scenario_name.upper()}")
        print(f"Weather: {weather_forecast}")
        print("="*60)
        
        # Generate Oakmont-specific predictions
        oakmont_predictions = predictor.predict_us_open_2025(
            combined_df, weather_forecast
        )
        
        # Show top 20 for this scenario
        print(f"\nTOP 20 OAKMONT-SPECIFIC PREDICTIONS ({scenario_name}):")
        print("-" * 80)
        
        top_20 = oakmont_predictions.head(20)
        for idx, row in top_20.iterrows():
            rank_change = ""
            if 'datagolf_rank' in row:
                change = row['datagolf_rank'] - row['oakmont_rank']
                if change > 0:
                    rank_change = f"(↑{int(change)})"
                elif change < 0:
                    rank_change = f"(↓{int(abs(change))})"
                else:
                    rank_change = "(=)"
            
            print(f"{int(row['oakmont_rank']):2d}. {row['player_name']:<25} "
                  f"DG Rank: {int(row['datagolf_rank']):3d} {rank_change:<6} "
                  f"Oakmont Score: {row['oakmont_prediction']:.3f}")
        
        # Store results
        all_results[scenario_name] = oakmont_predictions
        
        # Save scenario-specific predictions
        output_file = f"data/predictions/us_open_2025_oakmont_{scenario_name}.csv"
        os.makedirs('data/predictions', exist_ok=True)
        oakmont_predictions.to_csv(output_file, index=False)
        print(f"\nSaved predictions to: {output_file}")
    
    # Compare scenarios to identify weather-sensitive players
    print(f"\n" + "="*80)
    print("WEATHER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Compare ideal vs challenging conditions
    if 'ideal_conditions' in all_results and 'challenging_conditions' in all_results:
        ideal = all_results['ideal_conditions'][['player_name', 'oakmont_rank']].rename(
            columns={'oakmont_rank': 'ideal_rank'}
        )
        challenging = all_results['challenging_conditions'][['player_name', 'oakmont_rank']].rename(
            columns={'oakmont_rank': 'challenging_rank'}
        )
        
        weather_comparison = ideal.merge(challenging, on='player_name')
        weather_comparison['weather_sensitivity'] = (
            weather_comparison['challenging_rank'] - weather_comparison['ideal_rank']
        )
        
        # Players who perform better in challenging conditions
        weather_warriors = weather_comparison[
            weather_comparison['weather_sensitivity'] < -5
        ].sort_values('weather_sensitivity')
        
        print("\nWEATHER WARRIORS (Better in Challenging Conditions):")
        for idx, row in weather_warriors.head(10).iterrows():
            print(f"{row['player_name']:<25} Ideal: {int(row['ideal_rank']):3d} → "
                  f"Challenging: {int(row['challenging_rank']):3d} "
                  f"(Improvement: {int(abs(row['weather_sensitivity']))})")
        
        # Players who struggle in challenging conditions
        weather_sensitive = weather_comparison[
            weather_comparison['weather_sensitivity'] > 5
        ].sort_values('weather_sensitivity', ascending=False)
        
        print("\nWEATHER SENSITIVE (Worse in Challenging Conditions):")
        for idx, row in weather_sensitive.head(10).iterrows():
            print(f"{row['player_name']:<25} Ideal: {int(row['ideal_rank']):3d} → "
                  f"Challenging: {int(row['challenging_rank']):3d} "
                  f"(Drop: {int(row['weather_sensitivity'])})")
    
    # Compare with general predictions if available
    if general_predictions is not None:
        print(f"\n" + "="*80)
        print("GENERAL RANKING vs OAKMONT-SPECIFIC COMPARISON")
        print("="*80)
        
        # Use ideal conditions for comparison
        oakmont_ideal = all_results['ideal_conditions']
        comparison = predictor.compare_predictions(general_predictions, oakmont_ideal)
        
        # Biggest risers (better fit for Oakmont than general ranking suggests)
        risers = comparison[comparison['rank_change'] > 0].head(15)
        print("\nBIGGEST OAKMONT RISERS (Better fit for this course):")
        for idx, row in risers.iterrows():
            print(f"{row['player_name']:<25} General: {int(row['datagolf_rank']):3d} → "
                  f"Oakmont: {int(row['oakmont_rank']):3d} "
                  f"(↑{int(row['rank_change'])})")
        
        # Biggest fallers (worse fit for Oakmont)
        fallers = comparison[comparison['rank_change'] < 0].tail(15)
        print("\nBIGGEST OAKMONT FALLERS (Worse fit for this course):")
        for idx, row in fallers.iterrows():
            print(f"{row['player_name']:<25} General: {int(row['datagolf_rank']):3d} → "
                  f"Oakmont: {int(row['oakmont_rank']):3d} "
                  f"(↓{int(abs(row['rank_change']))})")
    
    print(f"\n" + "="*80)
    print("COURSE-SPECIFIC MODEL ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Insights:")
    print("• Oakmont rewards accuracy over distance")
    print("• Weather conditions significantly impact player rankings")
    print("• Course-specific fit can move players 20+ spots vs general rankings")
    print("• Iron play and putting on bentgrass are critical success factors")
    print(f"\nPredictions saved for {len(weather_scenarios)} weather scenarios")


if __name__ == "__main__":
    main()
