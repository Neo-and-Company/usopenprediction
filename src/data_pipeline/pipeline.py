"""
Complete Data Pipeline for Golf Prediction System
Integrates field filtering, database management, and prediction generation.
"""

import pandas as pd
import ast
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_pipeline.field_filter import USOpenFieldFilter
from data_pipeline.database_manager import GolfPredictionDB
from modeling.enhanced_course_prediction import EnhancedCoursePredictionSystem


class GolfPredictionPipeline:
    """Complete data pipeline for golf predictions."""
    
    def __init__(self, db_path: str = "data/golf_predictions.db"):
        """Initialize the pipeline.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.field_filter = USOpenFieldFilter()
        self.predictor = EnhancedCoursePredictionSystem()
        
    def run_complete_pipeline(self) -> dict:
        """Run the complete data pipeline.
        
        Returns:
            Pipeline execution report
        """
        print("=" * 80)
        print("GOLF PREDICTION DATA PIPELINE")
        print("=" * 80)
        
        report = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'errors': [],
            'final_stats': {}
        }
        
        try:
            # Step 1: Load and parse raw data
            print("\n1. Loading Raw Data...")
            skills_df, rankings_df = self._load_raw_data()
            report['steps_completed'].append('load_raw_data')
            
            # Step 2: Extract tournament field
            print("\n2. Extracting Tournament Field...")
            tournament_field = self.field_filter.extract_tournament_field()
            field_stats = self.field_filter.get_field_statistics()
            report['field_stats'] = field_stats
            report['steps_completed'].append('extract_field')
            
            # Step 3: Filter data to field participants
            print("\n3. Filtering Data to Field Participants...")
            filtered_skills = self.field_filter.filter_player_data(skills_df)
            validation = self.field_filter.validate_field_data(skills_df)
            report['validation'] = validation
            report['steps_completed'].append('filter_data')
            
            # Step 4: Initialize database
            print("\n4. Setting Up Database...")
            with GolfPredictionDB(self.db_path) as db:
                db.create_schema()
                
                # Load players
                players_loaded = db.load_players_data(tournament_field)
                
                # Load skills for field participants
                skills_loaded = db.load_player_skills(filtered_skills)
                
                # Create tournament
                tournament_id = db.create_tournament(
                    tournament_name="U.S. Open 2025",
                    course_name="Oakmont Country Club",
                    start_date="2025-06-12",
                    field_size=len(tournament_field)
                )
                
                # Load tournament field
                field_loaded = db.load_tournament_field(tournament_id, tournament_field)
                
                report['database_loads'] = {
                    'players': players_loaded,
                    'skills': skills_loaded,
                    'field_entries': field_loaded,
                    'tournament_id': tournament_id
                }
                report['steps_completed'].append('setup_database')
                
                # Step 5: Generate predictions
                print("\n5. Generating Predictions...")
                if len(filtered_skills) > 0:
                    predictions_df, analysis_report = self.predictor.run_enhanced_prediction(filtered_skills)
                    
                    # Save predictions to database
                    predictions_saved = db.save_predictions(tournament_id, predictions_df, "v1.0")
                    
                    # Save predictions to CSV for backup
                    predictions_file = f"data/predictions/us_open_2025_pipeline_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    predictions_df.to_csv(predictions_file, index=False)
                    
                    report['predictions'] = {
                        'predictions_generated': len(predictions_df),
                        'predictions_saved': predictions_saved,
                        'predictions_file': predictions_file,
                        'analysis_report': analysis_report
                    }
                    report['steps_completed'].append('generate_predictions')
                    
                    # Display top predictions
                    self._display_top_predictions(predictions_df)
                    
                else:
                    print("⚠️  No field participants found in skills data")
                    report['errors'].append('no_field_participants_in_skills')
                
                # Step 6: Generate final statistics
                db_stats = db.get_database_stats()
                report['final_stats'] = db_stats
                report['steps_completed'].append('generate_stats')
                
        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            report['errors'].append(str(e))
            import traceback
            traceback.print_exc()
        
        report['end_time'] = datetime.now()
        report['duration'] = report['end_time'] - report['start_time']
        
        # Print final report
        self._print_pipeline_report(report)
        
        return report
    
    def _load_raw_data(self) -> tuple:
        """Load and parse raw data files.
        
        Returns:
            Tuple of (skills_df, rankings_df)
        """
        # Load skills data
        skills_df = pd.read_csv('data/raw/current_skills.csv')
        skills_data = []
        for _, row in skills_df.iterrows():
            player_str = row['players']
            player_dict = ast.literal_eval(player_str)
            skills_data.append(player_dict)
        skills_df = pd.DataFrame(skills_data)
        
        # Load rankings data
        rankings_df = pd.read_csv('data/raw/current_rankings.csv')
        rankings_data = []
        for _, row in rankings_df.iterrows():
            ranking_str = row['rankings']
            ranking_dict = ast.literal_eval(ranking_str)
            rankings_data.append(ranking_dict)
        rankings_df = pd.DataFrame(rankings_data)
        
        # Merge skills and rankings
        merged_df = skills_df.merge(rankings_df, on='player_name', how='left', suffixes=('_skills', '_rankings'))
        
        print(f"✓ Loaded {len(skills_df)} skills records and {len(rankings_df)} ranking records")
        print(f"✓ Merged to {len(merged_df)} player records")
        
        return merged_df, rankings_df
    
    def _display_top_predictions(self, predictions_df: pd.DataFrame):
        """Display top predictions.
        
        Args:
            predictions_df: Predictions DataFrame
        """
        print("\n" + "=" * 80)
        print("TOP 15 PREDICTIONS - FILTERED TO ACTUAL US OPEN FIELD")
        print("=" * 80)
        
        # Sort by prediction score
        top_predictions = predictions_df.sort_values('final_prediction_score', ascending=False).head(15)
        
        for i, (_, player) in enumerate(top_predictions.iterrows(), 1):
            name = player['player_name']
            score = player['final_prediction_score']
            fit = player['course_fit_score']
            penalty = player.get('course_fit_penalty', 1.0)
            form = player['general_form_score']
            category = player.get('fit_category', 'Unknown')
            
            print(f"{i:2d}. {name:<25} | Score: {score:.3f} | "
                  f"Fit: {fit:.3f} ({category}) | "
                  f"Penalty: {penalty:.3f} | Form: {form:.3f}")
    
    def _print_pipeline_report(self, report: dict):
        """Print pipeline execution report.
        
        Args:
            report: Pipeline execution report
        """
        print("\n" + "=" * 80)
        print("PIPELINE EXECUTION REPORT")
        print("=" * 80)
        
        print(f"Duration: {report['duration']}")
        print(f"Steps Completed: {len(report['steps_completed'])}")
        
        if report['errors']:
            print(f"Errors: {len(report['errors'])}")
            for error in report['errors']:
                print(f"  - {error}")
        
        if 'field_stats' in report:
            stats = report['field_stats']
            print(f"\nField Statistics:")
            print(f"  Total Players: {stats['total_players']}")
            print(f"  Professionals: {stats['professionals']}")
            print(f"  Amateurs: {stats['amateurs']}")
            print(f"  Countries: {stats['countries_represented']}")
        
        if 'validation' in report:
            val = report['validation']
            print(f"\nData Validation:")
            print(f"  Match Rate: {val['match_percentage']:.1f}%")
            print(f"  Matched Players: {val['matched_players']}/{val['field_players']}")
        
        if 'database_loads' in report:
            loads = report['database_loads']
            print(f"\nDatabase Loads:")
            print(f"  Players: {loads['players']}")
            print(f"  Skills: {loads['skills']}")
            print(f"  Field Entries: {loads['field_entries']}")
            print(f"  Tournament ID: {loads['tournament_id']}")
        
        if 'predictions' in report:
            pred = report['predictions']
            print(f"\nPredictions:")
            print(f"  Generated: {pred['predictions_generated']}")
            print(f"  Saved to DB: {pred['predictions_saved']}")
            print(f"  File: {pred['predictions_file']}")
        
        if 'final_stats' in report:
            stats = report['final_stats']
            print(f"\nFinal Database Statistics:")
            for table, count in stats.items():
                print(f"  {table}: {count} records")
        
        print(f"\n✓ Pipeline completed successfully!")


def main():
    """Run the complete pipeline."""
    pipeline = GolfPredictionPipeline()
    report = pipeline.run_complete_pipeline()
    
    # Return report for potential further processing
    return report


if __name__ == "__main__":
    main()
