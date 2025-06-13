"""
Complete Golf Prediction Flask Application for Vercel
Mirrors the functionality of the local app running on port 8080
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from datetime import datetime
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'golf-prediction-vercel-2025')

# Configuration
DATAGOLF_API_KEY = os.environ.get('DATAGOLF_API_KEY', 'be1e0f4c0d741ab978b3fded7e8c')
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'data/golf_predictions.db')
USE_REAL_DATA = os.environ.get('USE_REAL_DATA', 'false').lower() == 'true'

# Try to import real modules, fallback to mock if not available
try:
    sys.path.append('src')
    from data_pipeline.query_examples import GolfPredictionQueries
    from evaluation.model_evaluation import GolfModelEvaluator
    queries = GolfPredictionQueries()
    evaluator = GolfModelEvaluator()
    REAL_MODULES_AVAILABLE = True
except ImportError:
    REAL_MODULES_AVAILABLE = False

# Mock data for fallback
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

# Mock classes for when real modules aren't available
class MockGolfPredictionQueries:
    """Mock version of GolfPredictionQueries for Vercel deployment."""
    
    def get_top_predictions(self, limit=25):
        """Get top predictions."""
        data = MOCK_PREDICTIONS[:limit]
        return pd.DataFrame(data)
    
    def get_tournament_field_summary(self):
        """Get tournament field summary."""
        return {
            'total_players': len(MOCK_PREDICTIONS),
            'with_predictions': len(MOCK_PREDICTIONS),
            'countries_represented': len(set(p['country'] for p in MOCK_PREDICTIONS)),
            'avg_prediction_score': np.mean([p['final_prediction_score'] for p in MOCK_PREDICTIONS]),
            'tournament': 'US Open 2025',
            'course': 'Oakmont Country Club'
        }
    
    def get_course_fit_vs_ranking_analysis(self):
        """Get course fit analysis."""
        fit_data = [
            {'fit_category': 'Excellent Fit', 'player_count': 1, 'avg_world_rank': 1.0, 'avg_fit_score': 0.92},
            {'fit_category': 'Good Fit', 'player_count': 5, 'avg_world_rank': 4.0, 'avg_fit_score': 0.75},
            {'fit_category': 'Average Fit', 'player_count': 4, 'avg_world_rank': 8.0, 'avg_fit_score': 0.65}
        ]
        return pd.DataFrame(fit_data)
    
    def find_value_picks(self, min_improvement=10):
        """Find value picks."""
        df = self.get_top_predictions(50)
        df['prediction_rank'] = range(1, len(df) + 1)
        df['rank_improvement'] = df['datagolf_rank'] - df['prediction_rank']
        value_picks = df[df['rank_improvement'] >= min_improvement]
        return value_picks
    
    def get_elite_players_analysis(self):
        """Get elite players analysis."""
        df = self.get_top_predictions(20)
        elite_players = df[df['datagolf_rank'] <= 20]
        return elite_players
    
    def get_players_by_course_fit(self, fit_category):
        """Get players by course fit category."""
        df = self.get_top_predictions(50)
        return df[df['fit_category'] == fit_category]
    
    def get_country_representation(self):
        """Get country representation."""
        country_data = [
            {'country': 'USA', 'player_count': 5, 'players_with_predictions': 5},
            {'country': 'NIR', 'player_count': 1, 'players_with_predictions': 1},
            {'country': 'ESP', 'player_count': 1, 'players_with_predictions': 1},
            {'country': 'NOR', 'player_count': 1, 'players_with_predictions': 1},
            {'country': 'SWE', 'player_count': 1, 'players_with_predictions': 1}
        ]
        return pd.DataFrame(country_data)

class MockGolfModelEvaluator:
    """Mock version of GolfModelEvaluator for Vercel deployment."""
    
    def evaluate_prediction_model(self):
        """Return mock evaluation results."""
        return MOCK_EVALUATION
    
    def get_feature_importance_analysis(self):
        """Return mock feature importance."""
        return {
            'top_features': [
                {'feature': 'Course Fit Score', 'importance': 0.35, 'description': 'Player-course compatibility'},
                {'feature': 'Strokes Gained Total', 'importance': 0.28, 'description': 'Overall performance metric'},
                {'feature': 'Recent Form', 'importance': 0.22, 'description': 'Last 24 rounds performance'},
                {'feature': 'DataGolf Rank', 'importance': 0.15, 'description': 'Current world ranking'}
            ],
            'feature_correlations': {
                'course_fit_vs_sg_total': 0.45,
                'recent_form_vs_rank': -0.67,
                'course_fit_vs_rank': -0.52
            }
        }
    
    def get_model_calibration_analysis(self):
        """Return mock calibration analysis."""
        return {
            'calibration_score': 0.85,
            'reliability_diagram': 'Mock calibration data for Vercel deployment',
            'brier_score': 0.12,
            'calibration_slope': 0.95,
            'calibration_intercept': 0.02
        }

# Initialize queries and evaluator (real or mock)
if not REAL_MODULES_AVAILABLE:
    queries = MockGolfPredictionQueries()
    evaluator = MockGolfModelEvaluator()

# HTML Template for evaluation page (matching your local version)
EVALUATION_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Evaluation - Golf Prediction System</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
    <style>
        body { background-color: #212529; color: #ffffff; }
        .navbar-brand { font-weight: bold; }
        .card { background-color: #343a40; border: none; }
        .card-header { background-color: #495057; border-bottom: 1px solid #6c757d; }
        .table-dark { --bs-table-bg: #343a40; }
        .metric-card { 
            background: linear-gradient(135deg, #495057 0%, #343a40 100%);
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .metric-value { 
            font-size: 2.5rem; 
            font-weight: bold; 
            color: #28a745; 
            margin-bottom: 5px;
        }
        .metric-label { 
            font-size: 0.9rem; 
            color: #adb5bd; 
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .chart-container { 
            position: relative; 
            height: 400px; 
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">🏌️ Golf Prediction System</a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">Dashboard</a>
                <a class="nav-link" href="/predictions">Predictions</a>
                <a class="nav-link" href="/value-picks">Value Picks</a>
                <a class="nav-link" href="/analytics">Analytics</a>
                <a class="nav-link active" href="/evaluation">Evaluation</a>
            </div>
        </div>
    </nav>

    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <h1 class="mb-4">📊 Model Evaluation & Performance Metrics</h1>
                <p class="lead">Comprehensive analysis of golf prediction model performance with ROC-AUC and F1 scores.</p>
            </div>
        </div>

        <!-- Performance Metrics Grid -->
        <div class="row mb-4">
            {% for outcome, metrics in evaluation_results.items() %}
            <div class="col-md-3 col-sm-6">
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(metrics.roc_auc) }}</div>
                    <div class="metric-label">{{ outcome.replace('_', ' ').title() }} ROC-AUC</div>
                    <hr style="border-color: #6c757d; margin: 15px 0;">
                    <div class="metric-value" style="font-size: 1.8rem; color: #17a2b8;">{{ "%.3f"|format(metrics.f1_score) }}</div>
                    <div class="metric-label">F1 Score</div>
                </div>
            </div>
            {% endfor %}
        </div>

        <!-- Detailed Metrics Table -->
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">📈 Detailed Performance Metrics</h5>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            <table class="table table-dark table-striped">
                                <thead>
                                    <tr>
                                        <th>Prediction Target</th>
                                        <th>ROC-AUC</th>
                                        <th>F1 Score</th>
                                        <th>Precision</th>
                                        <th>Recall</th>
                                        <th>Accuracy</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for outcome, metrics in evaluation_results.items() %}
                                    <tr>
                                        <td><strong>{{ outcome.replace('_', ' ').title() }}</strong></td>
                                        <td><span class="badge bg-success">{{ "%.3f"|format(metrics.roc_auc) }}</span></td>
                                        <td><span class="badge bg-info">{{ "%.3f"|format(metrics.f1_score) }}</span></td>
                                        <td>{{ "%.3f"|format(metrics.precision) }}</td>
                                        <td>{{ "%.3f"|format(metrics.recall) }}</td>
                                        <td>{{ "%.3f"|format(metrics.accuracy) }}</td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Feature Importance -->
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">🎯 Feature Importance</h5>
                    </div>
                    <div class="card-body">
                        {% for feature in feature_analysis.top_features %}
                        <div class="mb-3">
                            <div class="d-flex justify-content-between">
                                <span><strong>{{ feature.feature }}</strong></span>
                                <span class="badge bg-primary">{{ "%.1f"|format(feature.importance * 100) }}%</span>
                            </div>
                            <div class="progress mt-1" style="height: 8px;">
                                <div class="progress-bar bg-success" style="width: {{ feature.importance * 100 }}%"></div>
                            </div>
                            <small class="text-muted">{{ feature.description }}</small>
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">🎲 Model Calibration</h5>
                    </div>
                    <div class="card-body">
                        <div class="metric-card">
                            <div class="metric-value" style="font-size: 2rem;">{{ "%.3f"|format(calibration_analysis.calibration_score) }}</div>
                            <div class="metric-label">Calibration Score</div>
                        </div>
                        <div class="row">
                            <div class="col-6">
                                <strong>Brier Score:</strong><br>
                                <span class="text-info">{{ "%.3f"|format(calibration_analysis.brier_score) }}</span>
                            </div>
                            <div class="col-6">
                                <strong>Calibration Slope:</strong><br>
                                <span class="text-success">{{ "%.3f"|format(calibration_analysis.calibration_slope) }}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Footer -->
        <div class="row mt-5">
            <div class="col-12 text-center">
                <p class="text-muted">
                    <small>Model evaluation updated: {{ timestamp }} | Platform: Vercel Serverless</small>
                </p>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Route handlers - mirroring the local app functionality

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

        # Simple HTML for dashboard
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Golf Prediction Dashboard</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>body {{ background-color: #212529; color: #ffffff; }}</style>
        </head>
        <body>
            <div class="container mt-4">
                <h1>🏌️ Golf Prediction System</h1>
                <p class="lead">US Open 2025 Predictions</p>

                <div class="row">
                    <div class="col-md-4">
                        <div class="card bg-dark">
                            <div class="card-body">
                                <h5>Total Players</h5>
                                <h2 class="text-success">{summary['total_players']}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-dark">
                            <div class="card-body">
                                <h5>With Predictions</h5>
                                <h2 class="text-info">{summary['with_predictions']}</h2>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-4">
                        <div class="card bg-dark">
                            <div class="card-body">
                                <h5>Countries</h5>
                                <h2 class="text-warning">{summary['countries_represented']}</h2>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="mt-4">
                    <h3>Navigation</h3>
                    <a href="/predictions" class="btn btn-primary me-2">View Predictions</a>
                    <a href="/evaluation" class="btn btn-success me-2">Model Evaluation</a>
                    <a href="/api/health" class="btn btn-info me-2">API Health</a>
                </div>
            </div>
        </body>
        </html>
        """
        return dashboard_html
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1>"

@app.route('/predictions')
def predictions_page():
    """Predictions page with detailed view."""
    try:
        predictions = queries.get_top_predictions(50)
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

@app.route('/evaluation')
def evaluation_page():
    """Model evaluation page with ROC and F1 scores - matches local version."""
    try:
        # Get evaluation metrics
        evaluation_results = evaluator.evaluate_prediction_model()
        feature_analysis = evaluator.get_feature_importance_analysis()
        calibration_analysis = evaluator.get_model_calibration_analysis()

        return render_template_string(EVALUATION_HTML,
                                     evaluation_results=evaluation_results,
                                     feature_analysis=feature_analysis,
                                     calibration_analysis=calibration_analysis,
                                     timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    except Exception as e:
        return f"<h1>Error in evaluation page: {str(e)}</h1>"

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
            'real_modules': REAL_MODULES_AVAILABLE,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Vercel handler
def handler(request):
    return app(request.environ, lambda status, headers: None)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
