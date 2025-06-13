"""
WSGI entry point for production deployment
"""

import os
import sys
from flask import Flask

# Add src to path
sys.path.append('src')

def create_app(config_name=None):
    """Application factory pattern for Flask app creation."""
    
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'production')
    
    app = Flask(__name__)
    
    # Load configuration
    if config_name == 'production':
        from config.production import ProductionConfig
        app.config.from_object(ProductionConfig)
        ProductionConfig.init_app(app)
    elif config_name == 'development':
        from config.production import DevelopmentConfig
        app.config.from_object(DevelopmentConfig)
    else:
        # Default to development
        from config.production import DevelopmentConfig
        app.config.from_object(DevelopmentConfig)
    
    # Import and register routes
    from app import (
        index, predictions_page, value_picks_page, analytics_page, evaluation_page,
        api_predictions, api_top_predictions, api_value_picks, api_elite_analysis,
        api_course_fit, api_stats, api_model_evaluation, api_feature_importance,
        api_model_calibration, health_check, not_found, internal_error
    )
    
    # Register routes
    app.add_url_rule('/', 'index', index)
    app.add_url_rule('/predictions', 'predictions_page', predictions_page)
    app.add_url_rule('/value-picks', 'value_picks_page', value_picks_page)
    app.add_url_rule('/analytics', 'analytics_page', analytics_page)
    app.add_url_rule('/evaluation', 'evaluation_page', evaluation_page)
    
    # API routes
    app.add_url_rule('/api/predictions', 'api_predictions', api_predictions)
    app.add_url_rule('/api/predictions/top/<int:limit>', 'api_top_predictions', api_top_predictions)
    app.add_url_rule('/api/value-picks', 'api_value_picks', api_value_picks)
    app.add_url_rule('/api/elite-analysis', 'api_elite_analysis', api_elite_analysis)
    app.add_url_rule('/api/course-fit/<fit_category>', 'api_course_fit', api_course_fit)
    app.add_url_rule('/api/stats', 'api_stats', api_stats)
    app.add_url_rule('/api/model-evaluation', 'api_model_evaluation', api_model_evaluation)
    app.add_url_rule('/api/feature-importance', 'api_feature_importance', api_feature_importance)
    app.add_url_rule('/api/model-calibration', 'api_model_calibration', api_model_calibration)
    app.add_url_rule('/api/health', 'health_check', health_check)
    
    # Error handlers
    app.register_error_handler(404, not_found)
    app.register_error_handler(500, internal_error)
    
    # Ensure required directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    return app

# Create application instance for WSGI servers
application = create_app('production')
app = application  # For compatibility

if __name__ == '__main__':
    # For development server
    app = create_app('development')
    app.run(debug=True, host='0.0.0.0', port=5001)
