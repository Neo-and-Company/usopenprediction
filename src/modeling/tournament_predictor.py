"""
Machine learning models for golf tournament prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, log_loss
import xgboost as xgb
# import lightgbm as lgb  # Commented out due to installation issues


class TournamentPredictor:
    """Machine learning models for predicting golf tournament outcomes."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the tournament predictor.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Define model configurations
        self.model_configs = {
            'random_forest': {
                'regressor': RandomForestRegressor(random_state=random_state, n_estimators=100),
                'classifier': RandomForestClassifier(random_state=random_state, n_estimators=100)
            },
            'xgboost': {
                'regressor': xgb.XGBRegressor(random_state=random_state, n_estimators=100),
                'classifier': xgb.XGBClassifier(random_state=random_state, n_estimators=100)
            },
            # 'lightgbm': {
            #     'regressor': lgb.LGBMRegressor(random_state=random_state, n_estimators=100, verbose=-1),
            #     'classifier': lgb.LGBMClassifier(random_state=random_state, n_estimators=100, verbose=-1)
            # }
        }
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = None) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            
        Returns:
            Features DataFrame and target Series
        """
        print("Preparing features for modeling...")
        
        # Make a copy
        data = df.copy()
        
        # Remove non-feature columns
        non_feature_cols = [
            'player_id', 'player_name', 'event_id', 'event_name', 
            'year', 'date', 'course_id', 'course_name'
        ]
        
        # Remove target columns (except the one we're predicting)
        target_cols = ['won', 'top_5', 'top_10', 'top_20', 'finish_position', 'performance_score']
        if target_col:
            target_cols = [col for col in target_cols if col != target_col]
        
        cols_to_remove = non_feature_cols + target_cols
        cols_to_remove = [col for col in cols_to_remove if col in data.columns]
        
        # Extract target
        if target_col and target_col in data.columns:
            target = data[target_col].copy()
        else:
            target = None
        
        # Remove non-feature columns
        features = data.drop(columns=cols_to_remove, errors='ignore')
        
        # Handle categorical variables
        categorical_cols = features.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col].astype(str))
        
        # Handle missing values
        features = features.fillna(features.median())
        
        print(f"Features shape: {features.shape}")
        if target is not None:
            print(f"Target shape: {target.shape}")
            print(f"Target distribution:\n{target.value_counts().head()}")
        
        return features, target
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   model_type: str = 'xgboost', 
                   task_type: str = 'classification',
                   scale_features: bool = True) -> Dict[str, Any]:
        """Train a model for tournament prediction.
        
        Args:
            X: Features DataFrame
            y: Target Series
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            task_type: Type of task ('classification' or 'regression')
            scale_features: Whether to scale features
            
        Returns:
            Dictionary with trained model and metadata
        """
        print(f"Training {model_type} {task_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y if task_type == 'classification' else None
        )
        
        # Scale features if requested
        scaler = None
        if scale_features and model_type in ['logistic', 'linear']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Get model
        if model_type in self.model_configs:
            model = self.model_configs[model_type][task_type.replace('classification', 'classifier').replace('regression', 'regressor')]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        if task_type == 'classification':
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            
            metrics = {'accuracy': accuracy, 'auc': auc}
            
        else:  # regression
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            metrics = {'mse': mse, 'rmse': rmse}
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        model_info = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_predictions': y_pred,
            'test_actual': y_test
        }
        
        print(f"Model performance: {metrics}")
        print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return model_info
    
    def train_multiple_targets(self, df: pd.DataFrame, 
                             targets: List[str] = None) -> Dict[str, Dict]:
        """Train models for multiple target variables.
        
        Args:
            df: Input DataFrame with features and targets
            targets: List of target columns to predict
            
        Returns:
            Dictionary of trained models for each target
        """
        if targets is None:
            targets = ['won', 'top_5', 'top_10', 'top_20']
        
        print(f"Training models for targets: {targets}")
        
        results = {}
        
        for target in targets:
            if target in df.columns:
                print(f"\n--- Training model for {target} ---")
                
                # Prepare features
                X, y = self.prepare_features(df, target_col=target)
                
                # Skip if not enough positive examples
                if y.sum() < 10:
                    print(f"Skipping {target}: not enough positive examples ({y.sum()})")
                    continue
                
                # Determine task type
                task_type = 'classification' if target in ['won', 'top_5', 'top_10', 'top_20'] else 'regression'
                
                # Train multiple model types
                target_results = {}
                for model_type in ['random_forest', 'xgboost', 'lightgbm']:
                    try:
                        model_info = self.train_model(X, y, model_type, task_type)
                        target_results[model_type] = model_info
                    except Exception as e:
                        print(f"Error training {model_type} for {target}: {e}")
                
                results[target] = target_results
            else:
                print(f"Target column {target} not found in data")
        
        self.models = results
        return results
    
    def predict_tournament(self, features_df: pd.DataFrame, 
                          target: str = 'top_10', 
                          model_type: str = 'xgboost') -> pd.DataFrame:
        """Make predictions for a tournament field.
        
        Args:
            features_df: DataFrame with player features
            target: Target to predict
            model_type: Type of model to use
            
        Returns:
            DataFrame with predictions
        """
        print(f"Making {target} predictions using {model_type}...")
        
        if target not in self.models or model_type not in self.models[target]:
            raise ValueError(f"Model not found for target={target}, model_type={model_type}")
        
        model_info = self.models[target][model_type]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Prepare features (same as training)
        X, _ = self.prepare_features(features_df, target_col=None)
        
        # Ensure same columns as training
        if hasattr(model, 'feature_names_in_'):
            missing_cols = set(model.feature_names_in_) - set(X.columns)
            extra_cols = set(X.columns) - set(model.feature_names_in_)
            
            if missing_cols:
                print(f"Adding missing columns: {missing_cols}")
                for col in missing_cols:
                    X[col] = 0
            
            if extra_cols:
                print(f"Removing extra columns: {extra_cols}")
                X = X.drop(columns=list(extra_cols))
            
            # Reorder columns
            X = X[model.feature_names_in_]
        
        # Scale if needed
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_scaled)[:, 1]
        else:
            probabilities = predictions
        
        # Create results DataFrame
        results = features_df[['player_id']].copy() if 'player_id' in features_df.columns else pd.DataFrame()
        results[f'{target}_prediction'] = predictions
        results[f'{target}_probability'] = probabilities
        
        # Sort by probability
        results = results.sort_values(f'{target}_probability', ascending=False)
        
        return results
    
    def get_feature_importance(self, target: str = 'top_10', 
                             model_type: str = 'xgboost', 
                             top_n: int = 20) -> pd.DataFrame:
        """Get feature importance for a specific model.
        
        Args:
            target: Target variable
            model_type: Model type
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if target not in self.models or model_type not in self.models[target]:
            raise ValueError(f"Model not found for target={target}, model_type={model_type}")
        
        feature_importance = self.models[target][model_type]['feature_importance']
        
        if feature_importance is not None:
            return feature_importance.head(top_n)
        else:
            print(f"Feature importance not available for {model_type}")
            return pd.DataFrame()
    
    def save_models(self, filepath: str):
        """Save trained models to file.
        
        Args:
            filepath: Path to save models
        """
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.models, f)
        
        print(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file.
        
        Args:
            filepath: Path to load models from
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            self.models = pickle.load(f)
        
        print(f"Models loaded from {filepath}")
