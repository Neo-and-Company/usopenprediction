"""
Production configuration for Golf Prediction Flask Application
"""

import os
from datetime import timedelta

class ProductionConfig:
    """Production configuration settings."""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'golf-prediction-production-key-2025')
    DEBUG = False
    TESTING = False
    
    # Database settings
    DATABASE_PATH = os.environ.get('DATABASE_PATH', 'data/golf_predictions.db')
    
    # Security settings
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # CORS settings
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL', 'memory://')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    
    # API settings
    DATAGOLF_API_KEY = os.environ.get('DATAGOLF_API_KEY')
    
    # Performance settings
    SEND_FILE_MAX_AGE_DEFAULT = timedelta(hours=1)
    
    @staticmethod
    def init_app(app):
        """Initialize application with production settings."""
        # Configure logging
        import logging
        from logging.handlers import RotatingFileHandler
        
        if not app.debug:
            # Create logs directory if it doesn't exist
            if not os.path.exists('logs'):
                os.mkdir('logs')
            
            # Set up file handler
            file_handler = RotatingFileHandler(
                'logs/golf_prediction.log',
                maxBytes=10240000,  # 10MB
                backupCount=10
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
            ))
            file_handler.setLevel(logging.INFO)
            app.logger.addHandler(file_handler)
            
            app.logger.setLevel(logging.INFO)
            app.logger.info('Golf Prediction application startup')


class DevelopmentConfig:
    """Development configuration settings."""
    
    SECRET_KEY = 'golf-prediction-dev-key-2025'
    DEBUG = True
    TESTING = False
    DATABASE_PATH = 'data/golf_predictions.db'
    DATAGOLF_API_KEY = os.environ.get('DATAGOLF_API_KEY')


class TestingConfig:
    """Testing configuration settings."""
    
    SECRET_KEY = 'golf-prediction-test-key-2025'
    DEBUG = False
    TESTING = True
    DATABASE_PATH = ':memory:'  # In-memory database for testing
    DATAGOLF_API_KEY = 'test-api-key'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
