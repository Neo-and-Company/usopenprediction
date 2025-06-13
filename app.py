"""
Golf Prediction Flask Web Application
Serves golf predictions through REST API and web interface.
"""

from flask import Flask, render_template, jsonify, request
import pandas as pd
import sqlite3
import json
import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Add src to path
sys.path.append('src')

from data_pipeline.database_manager import GolfPredictionDB
from data_pipeline.query_examples import GolfPredictionQueries
from evaluation.model_evaluation import GolfModelEvaluator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'golf-prediction-secret-key-2025'

# Initialize database queries and evaluator
queries = GolfPredictionQueries()
evaluator = GolfModelEvaluator()

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
        # Handle pandas DataFrame/Series
        return convert_numpy_types(obj.to_dict('records'))
    return obj

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
        return render_template('error.html', error=str(e))

@app.route('/api/predictions')
def api_predictions():
    """API endpoint for all predictions."""
    try:
        limit = request.args.get('limit', 114, type=int)
        predictions = queries.get_top_predictions(limit)
        
        return jsonify({
            'status': 'success',
            'count': len(predictions),
            'predictions': convert_numpy_types(predictions.to_dict('records')),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predictions/top/<int:limit>')
def api_top_predictions(limit):
    """API endpoint for top N predictions."""
    try:
        if limit > 50:
            limit = 50  # Cap at 50 for performance
            
        predictions = queries.get_top_predictions(limit)
        
        return jsonify({
            'status': 'success',
            'limit': limit,
            'count': len(predictions),
            'predictions': convert_numpy_types(predictions.to_dict('records'))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/value-picks')
def api_value_picks():
    """API endpoint for value picks."""
    try:
        min_improvement = request.args.get('min_improvement', 10, type=int)
        value_picks = queries.find_value_picks(min_improvement)
        
        return jsonify({
            'status': 'success',
            'min_rank_improvement': min_improvement,
            'count': len(value_picks),
            'value_picks': convert_numpy_types(value_picks.to_dict('records'))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/elite-analysis')
def api_elite_analysis():
    """API endpoint for elite players analysis."""
    try:
        elite_analysis = queries.get_elite_players_analysis()
        
        return jsonify({
            'status': 'success',
            'count': len(elite_analysis),
            'elite_players': convert_numpy_types(elite_analysis.to_dict('records'))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/course-fit/<fit_category>')
def api_course_fit(fit_category):
    """API endpoint for players by course fit category."""
    try:
        valid_categories = ['Good Fit', 'Average Fit', 'Poor Fit', 'Very Poor Fit']
        if fit_category not in valid_categories:
            return jsonify({
                'status': 'error',
                'message': f'Invalid fit category. Valid options: {valid_categories}'
            }), 400
            
        players = queries.get_players_by_course_fit(fit_category)
        
        return jsonify({
            'status': 'success',
            'fit_category': fit_category,
            'count': len(players),
            'players': convert_numpy_types(players.to_dict('records'))
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for tournament and database statistics."""
    try:
        # Tournament field summary
        field_summary = queries.get_tournament_field_summary()
        
        # Course fit analysis
        fit_analysis = queries.get_course_fit_vs_ranking_analysis()
        
        # Country representation
        countries = queries.get_country_representation()
        
        return jsonify({
            'status': 'success',
            'tournament_summary': convert_numpy_types(field_summary),
            'course_fit_analysis': convert_numpy_types(fit_analysis.to_dict('records')),
            'country_representation': convert_numpy_types(countries.head(10).to_dict('records')),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/predictions')
def predictions_page():
    """Predictions page with detailed view."""
    try:
        predictions = queries.get_top_predictions(50)  # Show more predictions
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
        # Get various analytics
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
        # Get evaluation metrics
        evaluation_results = evaluator.evaluate_prediction_model()
        feature_analysis = evaluator.get_feature_importance_analysis()
        calibration_analysis = evaluator.get_model_calibration_analysis()

        return render_template('evaluation.html',
                             evaluation_results=evaluation_results,
                             feature_analysis=feature_analysis,
                             calibration_analysis=calibration_analysis)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/model-evaluation')
def api_model_evaluation():
    """API endpoint for model evaluation metrics."""
    try:
        evaluation_results = evaluator.evaluate_prediction_model()

        return jsonify({
            'status': 'success',
            'evaluation_results': convert_numpy_types(evaluation_results),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/feature-importance')
def api_feature_importance():
    """API endpoint for feature importance analysis."""
    try:
        feature_analysis = evaluator.get_feature_importance_analysis()

        return jsonify({
            'status': 'success',
            'feature_analysis': convert_numpy_types(feature_analysis),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/model-calibration')
def api_model_calibration():
    """API endpoint for model calibration analysis."""
    try:
        calibration_analysis = evaluator.get_model_calibration_analysis()

        return jsonify({
            'status': 'success',
            'calibration_analysis': convert_numpy_types(calibration_analysis),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    try:
        # Test database connection
        summary = queries.get_tournament_field_summary()

        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'players_in_db': convert_numpy_types(summary.get('total_players', 0)),
            'predictions_count': convert_numpy_types(summary.get('with_predictions', 0)),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

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

if __name__ == '__main__':
    # Ensure templates directory exists
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("=" * 60)
    print("GOLF PREDICTION FLASK APPLICATION")
    print("=" * 60)
    print("Starting server...")
    print("Dashboard: http://localhost:5001")
    print("API Docs: http://localhost:5001/api/health")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5001)
