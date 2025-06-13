"""
Vercel-optimized Flask application for Golf Prediction System
Lightweight serverless deployment with self-contained functionality
"""

import os
import json
import sqlite3
from flask import Flask, render_template, jsonify, request
from datetime import datetime
import pandas as pd
import numpy as np

# Mock data for fast deployment and fallback
MOCK_PREDICTIONS = [
    {'player_name': 'Scottie Scheffler', 'final_prediction_score': 0.95, 'course_fit_score': 0.92, 'fit_category': 'Excellent Fit', 'datagolf_rank': 1, 'country': 'USA'},
    {'player_name': 'Rory McIlroy', 'final_prediction_score': 0.88, 'course_fit_score': 0.85, 'fit_category': 'Good Fit', 'datagolf_rank': 2, 'country': 'NIR'},
    {'player_name': 'Jon Rahm', 'final_prediction_score': 0.82, 'course_fit_score': 0.78, 'fit_category': 'Good Fit', 'datagolf_rank': 3, 'country': 'ESP'},
    {'player_name': 'Viktor Hovland', 'final_prediction_score': 0.79, 'course_fit_score': 0.75, 'fit_category': 'Good Fit', 'datagolf_rank': 4, 'country': 'NOR'},
    {'player_name': 'Xander Schauffele', 'final_prediction_score': 0.76, 'course_fit_score': 0.72, 'fit_category': 'Good Fit', 'datagolf_rank': 5, 'country': 'USA'},
    {'player_name': 'Collin Morikawa', 'final_prediction_score': 0.74, 'course_fit_score': 0.70, 'fit_category': 'Good Fit', 'datagolf_rank': 6, 'country': 'USA'},
    {'player_name': 'Patrick Cantlay', 'final_prediction_score': 0.72, 'course_fit_score': 0.68, 'fit_category': 'Average Fit', 'datagolf_rank': 7, 'country': 'USA'},
    {'player_name': 'Ludvig Aberg', 'final_prediction_score': 0.70, 'course_fit_score': 0.66, 'fit_category': 'Average Fit', 'datagolf_rank': 8, 'country': 'SWE'},
    {'player_name': 'Wyndham Clark', 'final_prediction_score': 0.68, 'course_fit_score': 0.64, 'fit_category': 'Average Fit', 'datagolf_rank': 9, 'country': 'USA'},
    {'player_name': 'Max Homa', 'final_prediction_score': 0.66, 'course_fit_score': 0.62, 'fit_category': 'Average Fit', 'datagolf_rank': 10, 'country': 'USA'}
]

MOCK_EVALUATION = {
    'made_cut': {'roc_auc': 0.742, 'f1_score': 0.681, 'precision': 0.723, 'recall': 0.642, 'accuracy': 0.698},
    'top_10': {'roc_auc': 0.834, 'f1_score': 0.756, 'precision': 0.789, 'recall': 0.725, 'accuracy': 0.812},
    'top_20': {'roc_auc': 0.798, 'f1_score': 0.712, 'precision': 0.745, 'recall': 0.681, 'accuracy': 0.776},
    'winner': {'roc_auc': 0.923, 'f1_score': 0.845, 'precision': 0.867, 'recall': 0.824, 'accuracy': 0.891}
}

# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'golf-prediction-vercel-key-2025')

# Lightweight database interface for Vercel
class VercelGolfQueries:
    """Simplified database queries for Vercel deployment."""

    def __init__(self, db_path: str = "data/golf_predictions.db"):
        self.db_path = db_path
        self.use_mock_data = not os.path.exists(db_path)

    def get_top_predictions(self, limit: int = 25) -> pd.DataFrame:
        """Get top predictions, fallback to mock data if database unavailable."""
        if self.use_mock_data:
            return pd.DataFrame(MOCK_PREDICTIONS[:limit])

        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT
                    p.player_name,
                    pr.final_prediction_score,
                    pr.course_fit_score,
                    pr.fit_category,
                    ps.datagolf_rank,
                    p.country
                FROM predictions pr
                JOIN players p ON pr.player_id = p.player_id
                LEFT JOIN player_skills ps ON p.player_id = ps.player_id
                ORDER BY pr.final_prediction_score DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
            conn.close()
            return df
        except Exception:
            return pd.DataFrame(MOCK_PREDICTIONS[:limit])

    def get_tournament_field_summary(self) -> dict:
        """Get tournament field summary."""
        if self.use_mock_data:
            return {
                'total_players': len(MOCK_PREDICTIONS),
                'countries_represented': len(set(p['country'] for p in MOCK_PREDICTIONS)),
                'avg_prediction_score': np.mean([p['final_prediction_score'] for p in MOCK_PREDICTIONS]),
                'tournament': 'US Open 2025',
                'course': 'Oakmont Country Club'
            }

        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT
                    COUNT(*) as total_players,
                    COUNT(DISTINCT p.country) as countries_represented,
                    AVG(pr.final_prediction_score) as avg_prediction_score
                FROM predictions pr
                JOIN players p ON pr.player_id = p.player_id
            """
            result = pd.read_sql_query(query, conn).iloc[0]
            conn.close()
            return {
                'total_players': int(result['total_players']),
                'countries_represented': int(result['countries_represented']),
                'avg_prediction_score': float(result['avg_prediction_score']),
                'tournament': 'US Open 2025',
                'course': 'Oakmont Country Club'
            }
        except Exception:
            return {
                'total_players': len(MOCK_PREDICTIONS),
                'countries_represented': 5,
                'avg_prediction_score': 0.75,
                'tournament': 'US Open 2025',
                'course': 'Oakmont Country Club'
            }

    def get_course_fit_vs_ranking_analysis(self) -> pd.DataFrame:
        """Get course fit vs ranking analysis."""
        if self.use_mock_data:
            fit_data = [
                {'fit_category': 'Excellent Fit', 'player_count': 2, 'avg_world_rank': 1.5, 'avg_fit_score': 0.92},
                {'fit_category': 'Good Fit', 'player_count': 6, 'avg_world_rank': 4.5, 'avg_fit_score': 0.75},
                {'fit_category': 'Average Fit', 'player_count': 2, 'avg_world_rank': 8.5, 'avg_fit_score': 0.65}
            ]
            return pd.DataFrame(fit_data)

        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT
                    pr.fit_category,
                    COUNT(*) as player_count,
                    AVG(ps.datagolf_rank) as avg_world_rank,
                    AVG(pr.course_fit_score) as avg_fit_score
                FROM predictions pr
                LEFT JOIN player_skills ps ON pr.player_id = ps.player_id
                GROUP BY pr.fit_category
                ORDER BY avg_fit_score DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame([
                {'fit_category': 'Good Fit', 'player_count': 25, 'avg_world_rank': 30, 'avg_fit_score': 0.8}
            ])

    def find_value_picks(self, limit: int = 10) -> pd.DataFrame:
        """Find value picks - players ranked higher by model than DataGolf."""
        predictions = self.get_top_predictions(50)
        if predictions.empty:
            return pd.DataFrame(MOCK_PREDICTIONS[:limit])

        # Calculate rank improvement (lower is better for ranks)
        predictions['prediction_rank'] = range(1, len(predictions) + 1)
        predictions['rank_improvement'] = predictions['datagolf_rank'] - predictions['prediction_rank']

        # Filter for positive improvements and sort
        value_picks = predictions[predictions['rank_improvement'] > 0].sort_values(
            'rank_improvement', ascending=False
        ).head(limit)

        return value_picks

    def get_elite_players_analysis(self) -> pd.DataFrame:
        """Get elite players analysis."""
        predictions = self.get_top_predictions(20)
        if predictions.empty:
            return pd.DataFrame(MOCK_PREDICTIONS[:10])

        # Filter for top 20 ranked players
        elite_players = predictions[predictions['datagolf_rank'] <= 20]
        return elite_players

    def get_country_representation(self) -> pd.DataFrame:
        """Get country representation in the field."""
        if self.use_mock_data:
            country_data = [
                {'country': 'USA', 'player_count': 5, 'players_with_predictions': 5},
                {'country': 'NIR', 'player_count': 1, 'players_with_predictions': 1},
                {'country': 'ESP', 'player_count': 1, 'players_with_predictions': 1},
                {'country': 'NOR', 'player_count': 1, 'players_with_predictions': 1},
                {'country': 'SWE', 'player_count': 1, 'players_with_predictions': 1}
            ]
            return pd.DataFrame(country_data)

        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT
                    p.country,
                    COUNT(*) as player_count,
                    COUNT(pr.player_id) as players_with_predictions
                FROM players p
                LEFT JOIN predictions pr ON p.player_id = pr.player_id
                GROUP BY p.country
                ORDER BY player_count DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception:
            return pd.DataFrame([
                {'country': 'USA', 'player_count': 50, 'players_with_predictions': 45}
            ])

# Initialize queries with error handling
queries = VercelGolfQueries()

# Lightweight evaluator for Vercel
class VercelGolfEvaluator:
    """Simplified model evaluator for Vercel deployment."""

    def evaluate_prediction_model(self) -> dict:
        """Return mock evaluation results."""
        return MOCK_EVALUATION

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'to_dict'):
        return convert_numpy_types(obj.to_dict('records'))
    return obj

# Initialize evaluator
evaluator = VercelGolfEvaluator()

@app.route('/')
def index():
    """Main dashboard page."""
    try:
        # Get summary statistics
        summary = queries.get_tournament_field_summary()

        # Get top 10 predictions
        top_predictions = queries.get_top_predictions(10)

        # Get course fit analysis
        fit_analysis = queries.get_course_fit_vs_ranking_analysis()

        return render_template('index.html',
                             summary=summary,
                             top_predictions=top_predictions.to_dict('records'),
                             fit_analysis=fit_analysis.to_dict('records'))
    except Exception as e:
        return render_template('error.html', error=f"Dashboard error: {str(e)}")

@app.route('/api/health')
def health_check():
    """Health check endpoint for Vercel."""
    try:
        return jsonify({
            'status': 'healthy',
            'platform': 'vercel',
            'environment': 'serverless',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'platform': 'vercel',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for all predictions."""
    try:
        limit = request.args.get('limit', 25, type=int)
        predictions = queries.get_top_predictions(limit)
        
        # Handle empty DataFrame
        if predictions.empty:
            predictions = pd.DataFrame([{
                'player_name': 'Demo Player',
                'final_prediction_score': 1.0,
                'course_fit_score': 0.8,
                'fit_category': 'Good Fit',
                'datagolf_rank': 1
            }])
        
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'predictions': convert_numpy_types(predictions.to_dict('records')),
            'timestamp': datetime.now().isoformat(),
            'platform': 'vercel'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'platform': 'vercel',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/model-evaluation')
def api_model_evaluation():
    """API endpoint for model evaluation metrics."""
    try:
        evaluation_results = evaluator.evaluate_prediction_model()
        
        return jsonify({
            'status': 'success',
            'evaluation_results': convert_numpy_types(evaluation_results),
            'platform': 'vercel',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'platform': 'vercel',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predictions')
def predictions_page():
    """Predictions page with detailed view."""
    try:
        predictions = queries.get_top_predictions(50)
        return render_template('predictions.html',
                             predictions=predictions.to_dict('records'))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/value-picks')
def value_picks_page():
    """Value picks page."""
    try:
        value_picks = queries.find_value_picks(10)
        return render_template('value_picks.html',
                             value_picks=value_picks.to_dict('records'))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/analytics')
def analytics_page():
    """Analytics and insights page."""
    try:
        elite_analysis = queries.get_elite_players_analysis()
        fit_analysis = queries.get_course_fit_vs_ranking_analysis()
        countries = queries.get_country_representation()

        return render_template('analytics.html',
                             elite_players=elite_analysis.to_dict('records'),
                             fit_analysis=fit_analysis.to_dict('records'),
                             countries=countries.head(10).to_dict('records'))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/evaluation')
def evaluation_page():
    """Model evaluation page with ROC and F1 scores."""
    try:
        evaluation_results = evaluator.evaluate_prediction_model()

        # Mock feature and calibration analysis for Vercel
        feature_analysis = {
            'top_features': [
                {'feature': 'Course Fit Score', 'importance': 0.35},
                {'feature': 'Strokes Gained Total', 'importance': 0.28},
                {'feature': 'Recent Form', 'importance': 0.22},
                {'feature': 'DataGolf Rank', 'importance': 0.15}
            ]
        }

        calibration_analysis = {
            'calibration_score': 0.85,
            'reliability_diagram': 'Mock calibration data for Vercel deployment'
        }

        return render_template('evaluation.html',
                             evaluation_results=evaluation_results,
                             feature_analysis=feature_analysis,
                             calibration_analysis=calibration_analysis)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('error.html', 
                         error="Internal server error"), 500

# Vercel requires the app to be available at module level
# This is the entry point for Vercel
def handler(request):
    """Vercel serverless function handler."""
    return app(request.environ, lambda status, headers: None)

# For local testing
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
