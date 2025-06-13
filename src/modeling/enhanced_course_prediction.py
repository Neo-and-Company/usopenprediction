"""
Enhanced Course Prediction System integrating course-specific historical performance.
Combines course engineering, historical performance, and general form metrics.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')
from modeling.course_engineering import CourseEngineeringSystem
from modeling.course_specific_performance import CourseSpecificPerformanceAnalyzer
from modeling.scorecard_predictor import ScorecardPredictor


class EnhancedCoursePredictionSystem:
    """Enhanced prediction system with course-specific historical performance."""
    
    def __init__(self, course_name: str = "Oakmont Country Club"):
        """Initialize the enhanced prediction system.
        
        Args:
            course_name: Name of the course for predictions
        """
        self.course_name = course_name
        self.course_engineering = CourseEngineeringSystem()
        self.performance_analyzer = CourseSpecificPerformanceAnalyzer(course_name)
        self.scorecard_predictor = ScorecardPredictor()
        
        # Feature weights for final prediction - REBALANCED FOR BETTER COURSE FIT IMPACT
        self.feature_weights = {
            'course_fit': 0.40,           # Course engineering fit (increased from 0.25)
            'historical_performance': 0.30,  # Course-specific history (decreased from 0.35)
            'general_form': 0.20,         # Recent general form (decreased from 0.25)
            'scorecard_prediction': 0.10  # Expected scoring (decreased from 0.15)
        }
    
    def run_enhanced_prediction(self, player_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Run the complete enhanced prediction analysis.
        
        Args:
            player_data: DataFrame with player information
            
        Returns:
            Tuple of (predictions_df, analysis_report)
        """
        print(f"\n{'='*80}")
        print("ENHANCED COURSE PREDICTION SYSTEM")
        print(f"Course: {self.course_name}")
        print(f"{'='*80}")
        
        # Step 1: Course Engineering Analysis
        print("\n1. Running Course Engineering Analysis...")
        oakmont_setup = self.course_engineering.course_database["oakmont_2025"]
        course_fit_results = []

        for _, player in player_data.iterrows():
            player_dict = player.to_dict()
            fit_analysis = self.course_engineering.calculate_player_course_fit(
                player_dict, oakmont_setup
            )
            course_fit_results.append(fit_analysis)
        
        # Step 2: Course-Specific Historical Performance
        print("\n2. Analyzing Course-Specific Historical Performance...")
        course_history = self.performance_analyzer.collect_course_historical_data(player_data)
        course_performance_metrics = self.performance_analyzer.calculate_course_performance_metrics(course_history)
        
        # Handle missing historical data
        complete_performance_metrics = self.performance_analyzer.handle_missing_course_data(
            player_data, course_performance_metrics
        )
        
        # Create performance features
        performance_features = self.performance_analyzer.create_course_performance_features(
            complete_performance_metrics
        )
        
        # Step 3: General Form Analysis (using existing scorecard predictor)
        print("\n3. Analyzing General Form and Scorecard Predictions...")
        general_form_metrics = self._analyze_general_form(player_data)
        
        # Step 4: Integrate All Components
        print("\n4. Integrating All Prediction Components...")
        integrated_predictions = self._integrate_prediction_components(
            player_data, course_fit_results, performance_features, general_form_metrics
        )
        
        # Step 5: Generate Analysis Report
        analysis_report = self._generate_analysis_report(
            integrated_predictions, course_fit_results, performance_features
        )
        
        return integrated_predictions, analysis_report
    
    def _analyze_general_form(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Analyze general form metrics for players using stable Strokes Gained data.

        Args:
            player_data: Player data

        Returns:
            DataFrame with general form metrics
        """
        form_metrics = []

        for _, player in player_data.iterrows():
            player_dict = player.to_dict()

            # Calculate stable form score based on Strokes Gained Total
            # This is more reliable than using predicted scores
            sg_total = player.get('sg_total', 0.0)
            datagolf_rank = player.get('datagolf_rank', 100)

            # Create a form score that combines SG Total with ranking
            # Higher SG Total = better form, Lower ranking = better form
            form_score = self._calculate_stable_form_score(sg_total, datagolf_rank)

            # Use scorecard predictor only for reference, not as primary form indicator
            scorecard = self.scorecard_predictor.predict_tournament_scorecard(
                player_dict, ['ideal', 'windy', 'challenging', 'windy']
            )

            # Extract form indicators from player data
            form_metric = {
                'player_name': player['player_name'],
                'dg_id': player.get('dg_id', 0),
                'datagolf_rank': datagolf_rank,
                'sg_total': sg_total,
                'stable_form_score': form_score,
                'predicted_total_score': scorecard['tournament_total'],
                'predicted_relative_score': scorecard['relative_to_par'],
                'expected_finish': scorecard['projected_finish'],
                'make_cut_probability': 1.0 if scorecard['made_cut'] else 0.0,
                'general_skill_rating': player.get('dg_skill_estimate', 0.0),
                'recent_form_indicator': self._calculate_recent_form_indicator(player)
            }

            form_metrics.append(form_metric)

        return pd.DataFrame(form_metrics)

    def _calculate_stable_form_score(self, sg_total: float, datagolf_rank: int) -> float:
        """Calculate a stable form score based on Strokes Gained Total and ranking.

        This replaces the previous circular form calculation that used predicted scores.

        Args:
            sg_total: Player's Strokes Gained Total
            datagolf_rank: Player's DataGolf ranking

        Returns:
            Normalized form score between 0 and 1 (higher = better form)
        """
        # Normalize SG Total (typical range is -2 to +4, elite players are 2+)
        # Scale to 0-1 where 0 = -2 SG, 1 = +4 SG
        sg_component = max(0, min(1, (sg_total + 2) / 6))

        # Normalize ranking (1 = best, 200+ = worst)
        # Scale to 0-1 where 1 = rank 1, 0 = rank 200+
        rank_component = max(0, min(1, (200 - datagolf_rank) / 199))

        # Combine with more weight on SG Total (70%) than ranking (30%)
        # SG Total is more stable and predictive than rankings
        form_score = (sg_component * 0.7) + (rank_component * 0.3)

        return form_score

    def _calculate_recent_form_indicator(self, player: pd.Series) -> float:
        """Calculate recent form indicator from player data.
        
        Args:
            player: Player data series
            
        Returns:
            Recent form indicator score
        """
        # Use available strokes gained data as form indicator
        sg_total = player.get('sg_total', 0.0)
        sg_components = [
            player.get('sg_ott', 0.0),
            player.get('sg_app', 0.0),
            player.get('sg_arg', 0.0),
            player.get('sg_putt', 0.0)
        ]
        
        # Calculate form score (higher = better recent form)
        form_score = sg_total + sum(sg_components) / 4
        return form_score

    def _calculate_course_fit_penalty(self, course_fit_score: float, course_fit_data: dict) -> float:
        """Calculate penalty multiplier based on course fit analysis.

        This creates negative interaction effects for players with poor course fits,
        ensuring that course fit analysis actually impacts final predictions.

        Args:
            course_fit_score: Overall course fit score (0-1)
            course_fit_data: Course fit analysis data

        Returns:
            Penalty multiplier (0.5-1.0, where 1.0 = no penalty, 0.5 = maximum penalty)
        """
        # Base penalty starts at 1.0 (no penalty)
        penalty_multiplier = 1.0

        # Apply penalties based on course fit score
        if course_fit_score < 0.6:  # Poor fit threshold
            # Scale penalty: 0.6 fit = 0.9 multiplier, 0.4 fit = 0.7 multiplier, etc.
            penalty_multiplier = 0.5 + (course_fit_score * 0.8)
        elif course_fit_score < 0.7:  # Average fit threshold
            # Lighter penalty for average fits
            penalty_multiplier = 0.8 + (course_fit_score * 0.3)

        # Additional penalties for specific weaknesses
        vulnerabilities = course_fit_data.get('key_vulnerabilities', [])
        if vulnerabilities is not None and len(vulnerabilities) > 0:
            # Handle both list and string representations
            if isinstance(vulnerabilities, str):
                vulnerabilities = [vulnerabilities]
            elif hasattr(vulnerabilities, '__iter__') and not isinstance(vulnerabilities, str):
                vulnerabilities = list(vulnerabilities)
            else:
                vulnerabilities = []

            # Count critical weaknesses
            critical_weaknesses = len([v for v in vulnerabilities if 'Weak' in str(v)])
            if critical_weaknesses >= 3:
                penalty_multiplier *= 0.85  # 15% additional penalty for 3+ weaknesses
            elif critical_weaknesses >= 2:
                penalty_multiplier *= 0.92  # 8% additional penalty for 2+ weaknesses

        # Ensure penalty stays within reasonable bounds
        penalty_multiplier = max(0.5, min(1.0, penalty_multiplier))

        return penalty_multiplier

    def _integrate_prediction_components(self, player_data: pd.DataFrame,
                                       course_fit_results: List[Dict],
                                       performance_features: pd.DataFrame,
                                       general_form_metrics: pd.DataFrame) -> pd.DataFrame:
        """Integrate all prediction components into final predictions.
        
        Args:
            player_data: Original player data
            course_fit_results: Course engineering results
            performance_features: Course-specific performance features
            general_form_metrics: General form metrics
            
        Returns:
            DataFrame with integrated predictions
        """
        integrated_predictions = []
        
        # Convert course fit results to DataFrame for easier merging
        course_fit_df = pd.DataFrame(course_fit_results)
        
        for _, player in player_data.iterrows():
            player_name = player['player_name']
            
            # Get course fit data
            course_fit_data = course_fit_df[course_fit_df['player_name'] == player_name]
            if course_fit_data.empty:
                continue
            course_fit_score = course_fit_data.iloc[0]['overall_fit_score']
            
            # Get historical performance data
            perf_data = performance_features[performance_features['player_name'] == player_name]
            if not perf_data.empty:
                perf_data = perf_data.iloc[0]
                historical_score = perf_data['course_mastery_score']
                reliability = perf_data['reliability_score']
                course_experience = perf_data['total_rounds']
            else:
                historical_score = 0.5  # Default for missing data
                reliability = 0.1
                course_experience = 0
            
            # Get general form data - USE NEW STABLE FORM SCORE
            form_data = general_form_metrics[general_form_metrics['player_name'] == player_name]
            if not form_data.empty:
                form_data = form_data.iloc[0]
                # Use the new stable form score instead of predicted relative score
                general_form_score = form_data['stable_form_score']
                predicted_score = form_data['predicted_relative_score']
                # Create independent scorecard prediction score based on expected performance
                scorecard_prediction_score = max(0, (10 - abs(predicted_score)) / 10)
            else:
                general_form_score = 0.5
                predicted_score = 5
                scorecard_prediction_score = 0.5
            
            # Calculate weighted prediction score
            weighted_score = (
                course_fit_score * self.feature_weights['course_fit'] +
                historical_score * self.feature_weights['historical_performance'] +
                general_form_score * self.feature_weights['general_form'] +
                scorecard_prediction_score * self.feature_weights['scorecard_prediction']  # Now using independent score
            )

            # APPLY COURSE FIT PENALTIES - New penalty system for poor course fits
            course_fit_penalty = self._calculate_course_fit_penalty(course_fit_score, course_fit_data)
            weighted_score_with_penalty = weighted_score * course_fit_penalty

            # Adjust for reliability of historical data
            confidence_adjustment = 0.8 + (reliability * 0.2)
            final_prediction_score = weighted_score_with_penalty * confidence_adjustment
            
            # Create integrated prediction
            prediction = {
                'player_name': player_name,
                'dg_id': player.get('dg_id', 0),
                'datagolf_rank': player.get('datagolf_rank', 100),
                'final_prediction_score': final_prediction_score,
                'course_fit_score': course_fit_score,
                'course_fit_penalty': course_fit_penalty,  # NEW: Show penalty applied
                'historical_performance_score': historical_score,
                'general_form_score': general_form_score,
                'predicted_tournament_score': predicted_score,
                'reliability_factor': reliability,
                'course_experience_rounds': course_experience,
                'confidence_level': confidence_adjustment,
                'prediction_components': {
                    'course_fit_weight': self.feature_weights['course_fit'],
                    'historical_weight': self.feature_weights['historical_performance'],
                    'form_weight': self.feature_weights['general_form'],
                    'scorecard_weight': self.feature_weights['scorecard_prediction']
                }
            }
            
            # Add course fit details
            if not course_fit_data.empty:
                fit_details = course_fit_data.iloc[0]
                prediction.update({
                    'fit_category': fit_details.get('fit_category', 'Unknown'),
                    'key_advantages': fit_details.get('key_advantages', []),
                    'key_vulnerabilities': fit_details.get('key_vulnerabilities', [])
                })
            
            # Add historical performance details
            if not perf_data.empty:
                prediction.update({
                    'historical_avg_score': perf_data.get('avg_score', 0),
                    'historical_tournaments': perf_data.get('total_tournaments', 0),
                    'years_since_last_played': perf_data.get('years_since_last_played', 999),
                    'course_consistency': perf_data.get('consistency_score', 0),
                    'made_cut_rate_at_course': perf_data.get('made_cut_rate', 0)
                })
            
            integrated_predictions.append(prediction)
        
        # Sort by final prediction score
        predictions_df = pd.DataFrame(integrated_predictions)
        predictions_df = predictions_df.sort_values('final_prediction_score', ascending=False)
        
        return predictions_df

    def _generate_analysis_report(self, predictions: pd.DataFrame,
                                course_fit_results: List[Dict],
                                performance_features: pd.DataFrame) -> Dict:
        """Generate comprehensive analysis report.

        Args:
            predictions: Integrated predictions
            course_fit_results: Course fit results
            performance_features: Performance features

        Returns:
            Analysis report dictionary
        """
        report = {
            'summary': {},
            'top_contenders': [],
            'historical_insights': {},
            'course_fit_insights': {},
            'methodology': {}
        }

        # Summary statistics
        report['summary'] = {
            'total_players_analyzed': len(predictions),
            'players_with_course_history': len(performance_features[performance_features['total_rounds'] > 0]),
            'players_without_history': len(performance_features[performance_features['total_rounds'] == 0]),
            'average_prediction_confidence': predictions['confidence_level'].mean(),
            'prediction_score_range': {
                'min': predictions['final_prediction_score'].min(),
                'max': predictions['final_prediction_score'].max(),
                'std': predictions['final_prediction_score'].std()
            }
        }

        # Top contenders analysis
        top_10 = predictions.head(10)
        for _, player in top_10.iterrows():
            contender = {
                'rank': len(report['top_contenders']) + 1,
                'player_name': player['player_name'],
                'prediction_score': player['final_prediction_score'],
                'course_fit': player['course_fit_score'],
                'historical_performance': player['historical_performance_score'],
                'general_form': player['general_form_score'],
                'confidence': player['confidence_level'],
                'course_experience': player['course_experience_rounds'],
                'key_strengths': player.get('key_advantages', []),
                'potential_concerns': player.get('key_vulnerabilities', [])
            }
            report['top_contenders'].append(contender)

        # Historical insights
        if not performance_features.empty:
            experienced_players = performance_features[performance_features['total_rounds'] >= 8]
            report['historical_insights'] = {
                'most_experienced_player': {
                    'name': performance_features.loc[performance_features['total_rounds'].idxmax(), 'player_name'],
                    'rounds': performance_features['total_rounds'].max()
                },
                'best_historical_performer': {
                    'name': performance_features.loc[performance_features['course_mastery_score'].idxmax(), 'player_name'],
                    'mastery_score': performance_features['course_mastery_score'].max()
                },
                'average_historical_score': performance_features['avg_score'].mean(),
                'players_with_winning_history': len(performance_features[performance_features['best_finish'] == 1])
            }

        # Course fit insights
        course_fit_df = pd.DataFrame(course_fit_results)
        if not course_fit_df.empty:
            report['course_fit_insights'] = {
                'best_course_fit': {
                    'name': course_fit_df.loc[course_fit_df['overall_fit_score'].idxmax(), 'player_name'],
                    'fit_score': course_fit_df['overall_fit_score'].max()
                },
                'average_fit_score': course_fit_df['overall_fit_score'].mean(),
                'fit_categories_distribution': course_fit_df['fit_category'].value_counts().to_dict()
            }

        # Methodology explanation
        report['methodology'] = {
            'feature_weights': self.feature_weights,
            'recency_decay_factor': self.performance_analyzer.recency_decay_factor,
            'min_rounds_for_reliability': self.performance_analyzer.min_rounds_for_reliability,
            'description': {
                'course_fit': 'Analyzes player skills vs course demands (greens, rough, etc.)',
                'historical_performance': 'Player\'s actual performance history at this specific course',
                'general_form': 'Recent overall form and skill level',
                'scorecard_prediction': 'Expected scoring based on current form'
            }
        }

        return report
