"""
Test script for Enhanced Course-Specific Prediction System.
Demonstrates integration of course fit, historical performance, and general form.
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')
from modeling.enhanced_course_prediction import EnhancedCoursePredictionSystem


def load_test_data():
    """Load test data for the enhanced prediction system."""
    try:
        # Try to load actual data
        player_data = pd.read_csv('data/processed/processed_current_rankings.csv')
        skills_data = pd.read_csv('data/processed/processed_current_skills.csv')

        # Check if both files have the required columns
        if 'dg_id' in player_data.columns and 'dg_id' in skills_data.columns:
            # Merge player data with skills
            merged_data = pd.merge(player_data, skills_data, on=['dg_id', 'player_name'], how='inner')
            print(f"Loaded {len(merged_data)} players from actual data files")
            return merged_data.head(50)  # Use top 50 for testing
        else:
            print("Required columns not found in data files, using simulated data...")
            return create_simulated_data()

    except (FileNotFoundError, KeyError) as e:
        print(f"Data loading issue ({e}), creating simulated test data...")
        return create_simulated_data()


def create_simulated_data():
    """Create simulated player data for testing."""
    np.random.seed(42)
    
    # Top golfers for realistic testing
    players = [
        "Scheffler, Scottie", "DeChambeau, Bryson", "Rahm, Jon", "McIlroy, Rory",
        "Niemann, Joaquin", "Thomas, Justin", "Fleetwood, Tommy", "Morikawa, Collin",
        "Schauffele, Xander", "Straka, Sepp", "Cantlay, Patrick", "Lowry, Shane",
        "Conners, Corey", "Henley, Russell", "Burns, Sam", "Griffin, Ben",
        "Bradley, Keegan", "Hatton, Tyrrell", "Matsuyama, Hideki", "Aberg, Ludvig",
        "Kim, Si Woo", "Spieth, Jordan", "Hovland, Viktor", "English, Harris",
        "Spaun, J.J.", "Berger, Daniel", "McNealy, Maverick", "MacIntyre, Robert",
        "McCarthy, Denny", "Hall, Harry", "Bhatia, Akshay", "Finau, Tony",
        "Pendrith, Taylor", "Im, Sungjae", "Poston, J.T.", "Rai, Aaron",
        "Day, Jason", "Reed, Patrick", "Noren, Alex", "Mitchell, Keith"
    ]
    
    simulated_data = []
    
    for i, player_name in enumerate(players):
        # Simulate realistic player data
        rank = i + 1
        skill_estimate = max(0, 3.5 - (rank * 0.08) + np.random.normal(0, 0.3))
        
        player_data = {
            'player_name': player_name,
            'dg_id': 18000 + i,
            'datagolf_rank': rank,
            'dg_skill_estimate': skill_estimate,
            'owgr_rank': rank + np.random.randint(-5, 15),
            'sg_total': skill_estimate + np.random.normal(0, 0.2),
            'sg_ott': np.random.normal(0.5, 0.4),
            'sg_app': np.random.normal(0.4, 0.3),
            'sg_arg': np.random.normal(0.2, 0.3),
            'sg_putt': np.random.normal(0.2, 0.3),
            'driving_dist': np.random.normal(0, 10),
            'driving_acc': np.random.normal(0, 0.05),
            'country': 'USA',
            'primary_tour': 'PGA'
        }
        
        simulated_data.append(player_data)
    
    return pd.DataFrame(simulated_data)


def display_enhanced_predictions(predictions_df, analysis_report):
    """Display the enhanced prediction results."""
    
    print(f"\n{'='*100}")
    print("ENHANCED COURSE-SPECIFIC PREDICTION RESULTS")
    print(f"Course: Oakmont Country Club - US Open 2025")
    print(f"{'='*100}")
    
    # Summary statistics
    summary = analysis_report['summary']
    print(f"\nANALYSIS SUMMARY:")
    print(f"• Total Players Analyzed: {summary['total_players_analyzed']}")
    print(f"• Players with Course History: {summary['players_with_course_history']}")
    print(f"• Players without History: {summary['players_without_history']}")
    print(f"• Average Prediction Confidence: {summary['average_prediction_confidence']:.3f}")
    
    # Top contenders
    print(f"\n{'='*100}")
    print("TOP 15 CONTENDERS - ENHANCED PREDICTIONS")
    print(f"{'='*100}")
    
    print(f"{'Rank':<4} {'Player':<25} {'Pred':<6} {'Fit':<6} {'Hist':<6} {'Form':<6} {'Conf':<6} {'Exp':<4}")
    print(f"{'':4} {'':25} {'Score':<6} {'Score':<6} {'Score':<6} {'Score':<6} {'Level':<6} {'Rnds':<4}")
    print("-" * 100)
    
    for i, (_, player) in enumerate(predictions_df.head(15).iterrows()):
        rank = i + 1
        pred_score = f"{player['final_prediction_score']:.3f}"
        fit_score = f"{player['course_fit_score']:.3f}"
        hist_score = f"{player['historical_performance_score']:.3f}"
        form_score = f"{player['general_form_score']:.3f}"
        confidence = f"{player['confidence_level']:.3f}"
        experience = f"{int(player['course_experience_rounds'])}"
        
        print(f"{rank:<4} {player['player_name']:<25} {pred_score:<6} {fit_score:<6} "
              f"{hist_score:<6} {form_score:<6} {confidence:<6} {experience:<4}")
    
    # Detailed analysis of top 5
    print(f"\n{'='*100}")
    print("DETAILED ANALYSIS - TOP 5 CONTENDERS")
    print(f"{'='*100}")
    
    for contender in analysis_report['top_contenders'][:5]:
        print(f"\n{contender['rank']}. {contender['player_name'].upper()}")
        print(f"   Prediction Score: {contender['prediction_score']:.3f}")
        print(f"   Course Experience: {contender['course_experience']} rounds at Oakmont")
        print(f"   Confidence Level: {contender['confidence']:.3f}")
        
        print(f"   Component Breakdown:")
        print(f"     • Course Fit: {contender['course_fit']:.3f}")
        print(f"     • Historical Performance: {contender['historical_performance']:.3f}")
        print(f"     • General Form: {contender['general_form']:.3f}")
        
        if contender['key_strengths']:
            print(f"   Key Strengths: {', '.join(contender['key_strengths'][:3])}")
        if contender['potential_concerns']:
            print(f"   Potential Concerns: {', '.join(contender['potential_concerns'][:2])}")
    
    # Historical insights
    if 'historical_insights' in analysis_report and analysis_report['historical_insights']:
        hist_insights = analysis_report['historical_insights']
        print(f"\n{'='*100}")
        print("HISTORICAL PERFORMANCE INSIGHTS")
        print(f"{'='*100}")
        
        if 'most_experienced_player' in hist_insights:
            exp_player = hist_insights['most_experienced_player']
            print(f"Most Experienced at Oakmont: {exp_player['name']} ({exp_player['rounds']} rounds)")
        
        if 'best_historical_performer' in hist_insights:
            best_player = hist_insights['best_historical_performer']
            print(f"Best Historical Performer: {best_player['name']} (Mastery Score: {best_player['mastery_score']:.3f})")
        
        if 'average_historical_score' in hist_insights:
            print(f"Average Historical Score at Oakmont: {hist_insights['average_historical_score']:+.1f}")
        
        if 'players_with_winning_history' in hist_insights:
            print(f"Players with Winning History: {hist_insights['players_with_winning_history']}")
    
    # Course fit insights
    if 'course_fit_insights' in analysis_report and analysis_report['course_fit_insights']:
        fit_insights = analysis_report['course_fit_insights']
        print(f"\n{'='*100}")
        print("COURSE FIT INSIGHTS")
        print(f"{'='*100}")
        
        if 'best_course_fit' in fit_insights:
            best_fit = fit_insights['best_course_fit']
            print(f"Best Course Fit: {best_fit['name']} (Fit Score: {best_fit['fit_score']:.3f})")
        
        if 'average_fit_score' in fit_insights:
            print(f"Average Course Fit Score: {fit_insights['average_fit_score']:.3f}")
        
        if 'fit_categories_distribution' in fit_insights:
            print(f"Fit Categories Distribution:")
            for category, count in fit_insights['fit_categories_distribution'].items():
                print(f"  • {category}: {count} players")
    
    # Methodology explanation
    methodology = analysis_report['methodology']
    print(f"\n{'='*100}")
    print("PREDICTION METHODOLOGY")
    print(f"{'='*100}")
    
    print(f"Feature Weights:")
    for feature, weight in methodology['feature_weights'].items():
        description = methodology['description'].get(feature, '')
        print(f"  • {feature.replace('_', ' ').title()}: {weight:.1%} - {description}")
    
    print(f"\nKey Parameters:")
    print(f"  • Recency Decay Factor: {methodology['recency_decay_factor']}")
    print(f"  • Min Rounds for Reliability: {methodology['min_rounds_for_reliability']}")
    
    # Value picks analysis
    print(f"\n{'='*100}")
    print("VALUE PICKS ANALYSIS")
    print(f"{'='*100}")
    
    # Find players with high prediction scores but lower general rankings
    value_candidates = predictions_df[
        (predictions_df['datagolf_rank'] > 15) & 
        (predictions_df['final_prediction_score'] > predictions_df['final_prediction_score'].quantile(0.7))
    ].head(5)
    
    if not value_candidates.empty:
        print("Players with Strong Course-Specific Potential:")
        for _, player in value_candidates.iterrows():
            print(f"  • {player['player_name']} (Rank {player['datagolf_rank']}) - "
                  f"Prediction Score: {player['final_prediction_score']:.3f}")
            if player['course_experience_rounds'] > 0:
                print(f"    Has {int(player['course_experience_rounds'])} rounds of Oakmont experience")
            else:
                print(f"    Strong course fit despite no Oakmont history")
    
    print(f"\n{'='*100}")
    print("ENHANCED PREDICTION ANALYSIS COMPLETE")
    print(f"{'='*100}")


def main():
    """Run the enhanced course prediction system test."""
    print("Enhanced Course-Specific Prediction System Test")
    print("=" * 60)
    
    # Load data
    player_data = load_test_data()
    
    # Initialize enhanced prediction system
    enhanced_system = EnhancedCoursePredictionSystem("Oakmont Country Club")
    
    # Run enhanced predictions
    predictions_df, analysis_report = enhanced_system.run_enhanced_prediction(player_data)
    
    # Display results
    display_enhanced_predictions(predictions_df, analysis_report)
    
    # Save results
    output_dir = 'data/predictions'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    predictions_file = f"{output_dir}/enhanced_oakmont_predictions_{timestamp}.csv"
    predictions_df.to_csv(predictions_file, index=False)
    
    print(f"\nResults saved to: {predictions_file}")


if __name__ == "__main__":
    main()
