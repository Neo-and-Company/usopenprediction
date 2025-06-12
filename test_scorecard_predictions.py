"""
Test the scorecard prediction system - generate detailed scorecards for US Open 2025.
"""

import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from modeling.scorecard_predictor import ScorecardPredictor


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


def load_player_data():
    """Load and process player data."""
    
    # Load rankings data
    rankings_df = pd.read_csv('data/raw/current_rankings.csv')
    rankings_parsed = parse_json_column(rankings_df, 'rankings')
    
    # Load skill ratings data  
    skills_df = pd.read_csv('data/raw/skill_ratings.csv')
    skills_parsed = parse_json_column(skills_df, 'players')
    
    # Merge datasets
    combined_df = rankings_parsed.copy()
    if 'dg_id' in combined_df.columns and 'dg_id' in skills_parsed.columns:
        combined_df = combined_df.merge(skills_parsed, on='dg_id', how='left', suffixes=('', '_skill'))
    
    return combined_df


def format_scorecard_display(scorecard_data):
    """Format scorecard for nice display."""
    
    player_name = scorecard_data['player_name']
    total_relative = scorecard_data['relative_to_par']
    total_score = scorecard_data['tournament_total']
    
    print(f"\n{'='*80}")
    print(f"US OPEN 2025 SCORECARD - {player_name.upper()}")
    print(f"Total: {total_score} ({total_relative:+d}) - {scorecard_data['projected_finish']}")
    print(f"{'='*80}")
    
    # Round by round summary
    print(f"\nROUND-BY-ROUND SUMMARY:")
    print(f"{'Round':<8} {'Score':<8} {'To Par':<8} {'Conditions':<15}")
    print(f"{'-'*45}")
    
    for i, round_data in enumerate(scorecard_data['rounds']):
        conditions = f"{round_data['conditions'].wind_speed:.0f}mph wind, {round_data['conditions'].pin_positions} pins"
        relative_str = f"{round_data['relative_to_par']:+d}"
        print(f"R{round_data['round_number']:<7} {round_data['total_score']:<8} {relative_str:<7} {conditions}")
    
    # Tournament statistics
    summary = scorecard_data['tournament_summary']
    print(f"\nTOURNAMENT STATISTICS:")
    print(f"Eagles: {summary['eagles']}, Birdies: {summary['birdies']}, Pars: {summary['pars']}")
    print(f"Bogeys: {summary['bogeys']}, Doubles+: {summary['doubles_plus']}")
    
    # Show detailed scorecard for final round
    final_round = scorecard_data['rounds'][-1]
    print(f"\nFINAL ROUND HOLE-BY-HOLE:")
    print(f"{'Hole':<5} {'Par':<4} {'Score':<6} {'To Par':<7} {'Yardage':<8}")
    print(f"{'-'*35}")
    
    for hole in final_round['hole_by_hole']:
        relative_str = f"{hole['relative']:+d}" if hole['relative'] != 0 else "E"
        print(f"{hole['hole']:<5} {hole['par']:<4} {hole['score']:<6} {relative_str:<7} {hole['yardage']}")
    
    print(f"\nFinal Round: {final_round['front_nine']}-{final_round['back_nine']} = {final_round['total_score']}")


def generate_leaderboard(all_scorecards):
    """Generate tournament leaderboard from all scorecards."""
    
    leaderboard = []
    for scorecard in all_scorecards:
        leaderboard.append({
            'player_name': scorecard['player_name'],
            'total_score': scorecard['tournament_total'],
            'relative_to_par': scorecard['relative_to_par'],
            'r1': scorecard['rounds'][0]['relative_to_par'],
            'r2': scorecard['rounds'][1]['relative_to_par'],
            'r3': scorecard['rounds'][2]['relative_to_par'],
            'r4': scorecard['rounds'][3]['relative_to_par'],
            'made_cut': scorecard['made_cut'],
            'projected_finish': scorecard['projected_finish']
        })
    
    # Sort by total score
    leaderboard.sort(key=lambda x: x['relative_to_par'])
    
    return leaderboard


def main():
    """Generate detailed scorecards for US Open 2025."""
    
    print("="*80)
    print("US OPEN 2025 DETAILED SCORECARD PREDICTIONS")
    print("Oakmont Country Club - June 12-15, 2025")
    print("="*80)
    
    # Load player data
    player_data = load_player_data()
    
    # Initialize scorecard predictor
    predictor = ScorecardPredictor()
    
    # Define weather scenario for the tournament
    tournament_weather = ['ideal', 'windy', 'challenging', 'windy']  # Thu, Fri, Sat, Sun
    
    print(f"\nTournament Weather Forecast:")
    weather_names = ['Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, weather in enumerate(tournament_weather):
        print(f"{weather_names[i]}: {weather.title()} conditions")
    
    # Generate scorecards for top 20 players
    top_players = player_data.head(20)
    all_scorecards = []
    
    print(f"\nGenerating detailed scorecards for top 20 players...")
    
    for idx, player in top_players.iterrows():
        player_dict = player.to_dict()
        
        # Generate complete tournament scorecard
        scorecard = predictor.predict_tournament_scorecard(
            player_dict, tournament_weather
        )
        
        all_scorecards.append(scorecard)
        
        # Show progress
        print(f"Generated scorecard for {player_dict.get('player_name', 'Unknown')} "
              f"({scorecard['relative_to_par']:+d})")
    
    # Generate leaderboard
    leaderboard = generate_leaderboard(all_scorecards)
    
    # Display leaderboard
    print(f"\n{'='*80}")
    print("US OPEN 2025 PREDICTED LEADERBOARD")
    print(f"{'='*80}")
    print(f"{'Pos':<4} {'Player':<25} {'Total':<6} {'R1':<4} {'R2':<4} {'R3':<4} {'R4':<4} {'Status'}")
    print(f"{'-'*75}")
    
    for i, player in enumerate(leaderboard[:20]):
        pos = i + 1
        total_str = f"{player['relative_to_par']:+d}" if player['relative_to_par'] != 0 else "E"
        
        r1_str = f"{player['r1']:+d}"
        r2_str = f"{player['r2']:+d}"
        r3_str = f"{player['r3']:+d}"
        r4_str = f"{player['r4']:+d}"

        print(f"{pos:<4} {player['player_name']:<25} {total_str:<6} "
              f"{r1_str:<4} {r2_str:<4} {r3_str:<4} {r4_str:<4} "
              f"{player['projected_finish']}")
    
    # Show detailed scorecards for top 3
    print(f"\n{'='*80}")
    print("DETAILED SCORECARDS - TOP 3 FINISHERS")
    print(f"{'='*80}")
    
    for i in range(min(3, len(all_scorecards))):
        # Find scorecard for leaderboard position
        winner_name = leaderboard[i]['player_name']
        winner_scorecard = next(s for s in all_scorecards if s['player_name'] == winner_name)
        format_scorecard_display(winner_scorecard)
    
    # Save detailed results
    os.makedirs('data/predictions', exist_ok=True)
    
    # Save leaderboard
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df.to_csv('data/predictions/us_open_2025_leaderboard.csv', index=False)
    
    # Save detailed scorecards
    scorecards_data = []
    for scorecard in all_scorecards:
        # Flatten scorecard data for CSV
        base_data = {
            'player_name': scorecard['player_name'],
            'tournament_total': scorecard['tournament_total'],
            'relative_to_par': scorecard['relative_to_par'],
            'made_cut': scorecard['made_cut'],
            'projected_finish': scorecard['projected_finish']
        }
        
        # Add round scores
        for i, round_data in enumerate(scorecard['rounds']):
            base_data[f'round_{i+1}_score'] = round_data['total_score']
            base_data[f'round_{i+1}_relative'] = round_data['relative_to_par']
            base_data[f'round_{i+1}_front'] = round_data['front_nine']
            base_data[f'round_{i+1}_back'] = round_data['back_nine']
        
        # Add tournament stats
        stats = scorecard['tournament_summary']
        base_data.update({
            'total_eagles': stats['eagles'],
            'total_birdies': stats['birdies'],
            'total_pars': stats['pars'],
            'total_bogeys': stats['bogeys'],
            'total_doubles_plus': stats['doubles_plus']
        })
        
        scorecards_data.append(base_data)
    
    scorecards_df = pd.DataFrame(scorecards_data)
    scorecards_df.to_csv('data/predictions/us_open_2025_detailed_scorecards.csv', index=False)
    
    print(f"\n{'='*80}")
    print("SCORECARD PREDICTION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated detailed scorecards for {len(all_scorecards)} players")
    print(f"Leaderboard saved to: data/predictions/us_open_2025_leaderboard.csv")
    print(f"Detailed scorecards saved to: data/predictions/us_open_2025_detailed_scorecards.csv")
    
    # Tournament insights
    winner = leaderboard[0]
    print(f"\nPREDICTED WINNER: {winner['player_name']}")
    print(f"Winning Score: {winner['relative_to_par']:+d} ({winner['total_score']})")
    print(f"Round Scores: {winner['r1']:+d}, {winner['r2']:+d}, {winner['r3']:+d}, {winner['r4']:+d}")
    
    # Cut line analysis
    made_cut = [p for p in leaderboard if p['made_cut']]
    cut_line = max(p['relative_to_par'] for p in made_cut)
    print(f"\nPredicted Cut Line: {cut_line:+d}")
    print(f"Players Making Cut: {len(made_cut)}")


if __name__ == "__main__":
    main()
