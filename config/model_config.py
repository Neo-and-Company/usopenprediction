"""
Configuration settings for the US Open prediction models.
"""

# Data Collection Settings
DATA_COLLECTION = {
    'years_to_collect': list(range(2017, 2025)),  # Historical years
    'major_event_ids': {
        'masters': 14,
        'pga_championship': 33,
        'us_open': 26,
        'open_championship': 28
    },
    'us_open_event_id': 26,
    'rate_limit_delay': 0.5,  # Seconds between API calls
    'max_retries': 3
}

# Feature Engineering Settings
FEATURE_ENGINEERING = {
    'lookback_periods': [4, 8, 16],  # Number of tournaments for form features
    'trend_window_size': 8,  # Window for trend calculations
    'strokes_gained_cols': [
        'sg_putt', 'sg_arg', 'sg_app', 'sg_ott', 'sg_t2g', 'sg_total'
    ],
    'traditional_stats_cols': [
        'distance', 'accuracy', 'gir', 'scrambling', 'putts_per_round'
    ]
}

# Model Training Settings
MODEL_TRAINING = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'scale_features': True,
    
    # Model hyperparameters
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    
    'lightgbm': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'verbose': -1
    }
}

# Prediction Settings
PREDICTION = {
    'targets': ['won', 'top_5', 'top_10', 'top_20'],
    'field_size': 156,  # Typical US Open field size
    'model_priority': ['xgboost', 'lightgbm', 'random_forest'],  # Order of preference
    'confidence_threshold': 0.01  # Minimum probability to consider
}

# Data Cleaning Settings
DATA_CLEANING = {
    'missing_value_strategy': 'median',  # 'median', 'mean', or 'drop'
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'outlier_factor': 1.5,
    'min_tournaments_for_trends': 8,
    'max_finish_position': 200,
    'min_driving_distance': 200,
    'max_driving_distance': 400
}

# US Open Specific Settings
US_OPEN_2025 = {
    'event_name': 'US Open',
    'year': 2025,
    'course': 'Oakmont Country Club',
    'dates': 'June 12-15, 2025',
    'par': 70,
    'yardage': 7219,
    'course_characteristics': {
        'difficulty': 'Very Hard',
        'rough': 'Thick',
        'greens': 'Fast',
        'fairways': 'Narrow'
    }
}

# File Paths
PATHS = {
    'raw_data': 'data/raw',
    'processed_data': 'data/processed',
    'predictions': 'data/predictions',
    'models': 'models',
    'reports': 'reports'
}
