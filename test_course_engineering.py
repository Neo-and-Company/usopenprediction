"""
Test the advanced course engineering system.
Demonstrates how vague course descriptions become precise model inputs.
"""

import pandas as pd
import numpy as np
import json
import sys
import os

# Add src to path
sys.path.append('src')

from modeling.course_engineering import CourseEngineeringSystem


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


def display_course_engineering_analysis(engineering_system, player_data):
    """Display detailed course engineering analysis."""
    
    print("="*80)
    print("ADVANCED COURSE ENGINEERING ANALYSIS")
    print("Converting Course Conditions to Precise Model Inputs")
    print("="*80)
    
    # Get Oakmont setup
    oakmont_setup = engineering_system.course_database["oakmont_2025"]
    
    # Display course report
    print(engineering_system.generate_course_report(oakmont_setup))
    
    print("\n" + "="*80)
    print("PLAYER-COURSE FIT ANALYSIS")
    print("="*80)
    
    # Analyze top 15 players
    top_players = player_data.head(15)
    fit_results = []
    
    for idx, player in top_players.iterrows():
        player_dict = player.to_dict()
        
        # Calculate course fit
        fit_analysis = engineering_system.calculate_player_course_fit(
            player_dict, oakmont_setup
        )
        
        fit_results.append(fit_analysis)
    
    # Sort by overall fit score
    fit_results.sort(key=lambda x: x["overall_fit_score"], reverse=True)
    
    # Display results
    print(f"{'Rank':<4} {'Player':<25} {'Fit Score':<10} {'Category':<15} {'Key Advantages'}")
    print("-" * 90)
    
    for i, result in enumerate(fit_results):
        advantages = ", ".join(result["key_advantages"][:2]) if result["key_advantages"] else "None"
        print(f"{i+1:<4} {result['player_name']:<25} {result['overall_fit_score']:.3f}     "
              f"{result['fit_category']:<15} {advantages}")
    
    # Detailed analysis for top 3 fits
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS - TOP 3 COURSE FITS")
    print(f"{'='*80}")
    
    for i in range(min(3, len(fit_results))):
        result = fit_results[i]
        
        print(f"\n{i+1}. {result['player_name'].upper()}")
        print(f"Overall Fit Score: {result['overall_fit_score']:.3f} ({result['fit_category']})")
        
        print(f"\nCondition Breakdown:")
        for condition, scores in result["condition_breakdown"].items():
            condition_display = condition.replace('_', ' ').title()
            print(f"  {condition_display:<20}: {scores['fit_score']:.3f} "
                  f"(Weight: {scores['weight']:.1%}, Raw Skill: {scores['raw_skill']:.1f})")
        
        if result["key_advantages"]:
            print(f"\nKey Advantages: {', '.join(result['key_advantages'])}")
        
        if result["key_vulnerabilities"]:
            print(f"Vulnerabilities: {', '.join(result['key_vulnerabilities'])}")
    
    # Course condition impact analysis
    print(f"\n{'='*80}")
    print("COURSE CONDITION IMPACT ANALYSIS")
    print(f"{'='*80}")
    
    condition_impacts = {}
    for result in fit_results:
        for condition, scores in result["condition_breakdown"].items():
            if condition not in condition_impacts:
                condition_impacts[condition] = []
            condition_impacts[condition].append(scores["fit_score"])
    
    print(f"{'Condition':<20} {'Avg Fit':<10} {'Std Dev':<10} {'Impact'}")
    print("-" * 55)
    
    for condition, scores in condition_impacts.items():
        avg_fit = np.mean(scores)
        std_dev = np.std(scores)
        impact = "High" if std_dev > 0.3 else "Medium" if std_dev > 0.15 else "Low"
        
        condition_display = condition.replace('_', ' ').title()
        print(f"{condition_display:<20} {avg_fit:.3f}     {std_dev:.3f}     {impact}")
    
    return fit_results


def compare_engineering_vs_simple_ranking(fit_results, player_data):
    """Compare engineered course fit vs simple DataGolf rankings."""
    
    print(f"\n{'='*80}")
    print("ENGINEERED FIT vs SIMPLE RANKING COMPARISON")
    print(f"{'='*80}")
    
    # Create comparison data
    comparison_data = []
    
    for result in fit_results:
        player_name = result["player_name"]
        
        # Find original ranking
        player_row = player_data[player_data["player_name"] == player_name]
        if not player_row.empty:
            original_rank = player_row.iloc[0]["datagolf_rank"]
            
            comparison_data.append({
                "player_name": player_name,
                "original_rank": original_rank,
                "fit_score": result["overall_fit_score"],
                "fit_category": result["fit_category"]
            })
    
    # Sort by fit score to get engineered ranking
    comparison_data.sort(key=lambda x: x["fit_score"], reverse=True)
    
    # Add engineered ranks
    for i, data in enumerate(comparison_data):
        data["engineered_rank"] = i + 1
        data["rank_change"] = data["original_rank"] - data["engineered_rank"]
    
    print(f"{'Player':<25} {'Original':<8} {'Engineered':<10} {'Change':<8} {'Fit Category'}")
    print("-" * 75)
    
    for data in comparison_data:
        change_str = f"{data['rank_change']:+d}" if data['rank_change'] != 0 else "="
        print(f"{data['player_name']:<25} {data['original_rank']:<8} "
              f"{data['engineered_rank']:<10} {change_str:<8} {data['fit_category']}")
    
    # Identify biggest movers
    biggest_risers = sorted([d for d in comparison_data if d['rank_change'] > 0], 
                           key=lambda x: x['rank_change'], reverse=True)[:3]
    biggest_fallers = sorted([d for d in comparison_data if d['rank_change'] < 0], 
                            key=lambda x: x['rank_change'])[:3]
    
    if biggest_risers:
        print(f"\nBIGGEST RISERS (Better Course Fit):")
        for riser in biggest_risers:
            print(f"  {riser['player_name']}: {riser['original_rank']} → {riser['engineered_rank']} "
                  f"(+{riser['rank_change']})")
    
    if biggest_fallers:
        print(f"\nBIGGEST FALLERS (Worse Course Fit):")
        for faller in biggest_fallers:
            print(f"  {faller['player_name']}: {faller['original_rank']} → {faller['engineered_rank']} "
                  f"({faller['rank_change']})")


def main():
    """Main function to test course engineering system."""
    
    print("COURSE ENGINEERING SYSTEM TEST")
    print("Transforming 'Fast Greens' into 'Stimpmeter 14.5'")
    print("="*60)
    
    # Load player data
    player_data = load_player_data()
    
    # Initialize engineering system
    engineering_system = CourseEngineeringSystem()
    
    # Run comprehensive analysis
    fit_results = display_course_engineering_analysis(engineering_system, player_data)
    
    # Compare with simple rankings
    compare_engineering_vs_simple_ranking(fit_results, player_data)
    
    # Save results
    os.makedirs('data/predictions', exist_ok=True)
    
    # Convert fit results to DataFrame and save
    fit_data = []
    for result in fit_results:
        base_data = {
            'player_name': result['player_name'],
            'overall_fit_score': result['overall_fit_score'],
            'fit_category': result['fit_category'],
            'key_advantages': '; '.join(result['key_advantages']),
            'key_vulnerabilities': '; '.join(result['key_vulnerabilities'])
        }
        
        # Add condition scores
        for condition, scores in result['condition_breakdown'].items():
            base_data[f'{condition}_fit'] = scores['fit_score']
            base_data[f'{condition}_raw_skill'] = scores['raw_skill']
        
        fit_data.append(base_data)
    
    fit_df = pd.DataFrame(fit_data)
    fit_df.to_csv('data/predictions/us_open_2025_course_fit_analysis.csv', index=False)
    
    print(f"\n{'='*80}")
    print("COURSE ENGINEERING ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print("Key Insights:")
    print("• Course conditions converted to precise measurements")
    print("• Player skills matched to specific course demands")
    print("• Fit scores reveal who benefits from Oakmont's setup")
    print("• Engineering approach shows different rankings than simple DG rank")
    print(f"\nDetailed analysis saved to: data/predictions/us_open_2025_course_fit_analysis.csv")


if __name__ == "__main__":
    main()
