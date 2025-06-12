"""
Skill-based prediction model using current player ratings and rankings.
This approach works with basic DataGolf API access.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os
from typing import Dict, Tuple, List


class SkillBasedPredictor:
    """Prediction model based on current skill ratings and rankings."""
    
    def __init__(self, data_dir: str = 'data/raw'):
        """Initialize the predictor.
        
        Args:
            data_dir: Directory containing the collected data
        """
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data."""
        data = {}
        
        # Load each dataset
        datasets = [
            'players', 'current_rankings', 'skill_ratings', 
            'tournament_schedule', 'field_updates', 
            'us_open_prediction_archives', 'current_predictions'
        ]
        
        for dataset in datasets:
            filepath = os.path.join(self.data_dir, f'{dataset}.csv')
            if os.path.exists(filepath):
                data[dataset] = pd.read_csv(filepath)
                print(f"Loaded {dataset}: {data[dataset].shape}")
            else:
                print(f"File not found: {filepath}")
                data[dataset] = pd.DataFrame()
        
        return data
    
    def create_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create features for modeling from available data."""
        print("Creating features from available data...")
        
        # Start with current rankings
        features_df = data['current_rankings'].copy()
        
        # Merge with skill ratings
        if not data['skill_ratings'].empty:
            # Merge on player name or ID
            if 'player_name' in features_df.columns and 'player_name' in data['skill_ratings'].columns:
                features_df = features_df.merge(
                    data['skill_ratings'], 
                    on='player_name', 
                    how='left',
                    suffixes=('', '_skill')
                )
            elif 'dg_id' in features_df.columns and 'dg_id' in data['skill_ratings'].columns:
                features_df = features_df.merge(
                    data['skill_ratings'], 
                    on='dg_id', 
                    how='left',
                    suffixes=('', '_skill')
                )
        
        # Add player metadata
        if not data['players'].empty:
            player_cols = ['player_name', 'dg_id', 'country', 'amateur']
            available_cols = [col for col in player_cols if col in data['players'].columns]
            
            if available_cols:
                merge_col = 'dg_id' if 'dg_id' in available_cols else 'player_name'
                if merge_col in features_df.columns:
                    features_df = features_df.merge(
                        data['players'][available_cols], 
                        on=merge_col, 
                        how='left',
                        suffixes=('', '_player')
                    )
        
        # Create derived features
        features_df = self._create_derived_features(features_df)
        
        print(f"Created features dataset: {features_df.shape}")
        return features_df
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from base data."""
        
        # Ranking-based features
        if 'rank' in df.columns:
            df['rank_log'] = np.log(df['rank'] + 1)
            df['rank_inverse'] = 1 / (df['rank'] + 1)
            df['top_10_player'] = (df['rank'] <= 10).astype(int)
            df['top_50_player'] = (df['rank'] <= 50).astype(int)
        
        # Skill-based features (if available)
        skill_cols = [col for col in df.columns if 'sg_' in col.lower()]
        if skill_cols:
            # Overall skill composite
            df['skill_composite'] = df[skill_cols].mean(axis=1)
            
            # Skill consistency (lower std = more consistent)
            df['skill_consistency'] = -df[skill_cols].std(axis=1)
        
        # Country-based features
        if 'country' in df.columns:
            # Major golf countries
            major_golf_countries = ['USA', 'ENG', 'SCO', 'AUS', 'RSA', 'ESP', 'IRL']
            df['major_golf_country'] = df['country'].isin(major_golf_countries).astype(int)
        
        # Amateur status
        if 'amateur' in df.columns:
            df['is_professional'] = (df['amateur'] == 0).astype(int)
        
        return df
    
    def prepare_training_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data from prediction archives."""
        
        if data['us_open_prediction_archives'].empty:
            print("No historical prediction data available for training")
            return pd.DataFrame(), pd.Series()
        
        # Use prediction archives as training data
        training_df = data['us_open_prediction_archives'].copy()
        
        # Create features for historical data
        features_df = self.create_features(data)
        
        # For now, we'll use the prediction probabilities as our target
        # In a real scenario, you'd use actual tournament results
        if 'win_prob' in training_df.columns:
            target = training_df['win_prob']
        elif 'top_5_prob' in training_df.columns:
            target = training_df['top_5_prob']
        else:
            # Use first probability column found
            prob_cols = [col for col in training_df.columns if 'prob' in col.lower()]
            if prob_cols:
                target = training_df[prob_cols[0]]
            else:
                print("No suitable target variable found")
                return pd.DataFrame(), pd.Series()
        
        return features_df, target
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple models on the data."""
        
        if X.empty or y.empty:
            print("No training data available")
            return {}
        
        # Select numeric features only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].fillna(0)
        
        self.feature_columns = list(X_numeric.columns)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X_numeric)
        
        # Train multiple models
        models_to_train = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models_to_train.items():
            print(f"Training {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
            
            # Fit on full data
            model.fit(X_scaled, y)
            
            # Store model
            self.models[name] = model
            
            # Store results
            results[name] = {
                'cv_rmse': np.sqrt(-cv_scores.mean()),
                'cv_std': cv_scores.std(),
                'model': model
            }
            
            print(f"  {name} CV RMSE: {results[name]['cv_rmse']:.4f} (+/- {results[name]['cv_std']:.4f})")
        
        return results
    
    def predict_us_open_2025(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate predictions for US Open 2025."""
        
        # Create features for current players
        features_df = self.create_features(data)
        
        if features_df.empty:
            print("No feature data available for predictions")
            return pd.DataFrame()
        
        # Select same features used in training
        if not self.feature_columns:
            print("No trained model available")
            return pd.DataFrame()
        
        # Prepare features
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        X = features_df[available_features].fillna(0)
        
        # Scale features
        if 'standard' in self.scalers:
            X_scaled = self.scalers['standard'].transform(X)
        else:
            X_scaled = X.values
        
        # Generate predictions from all models
        predictions_df = features_df[['player_name', 'dg_id']].copy()
        
        for model_name, model in self.models.items():
            pred_col = f'{model_name}_prediction'
            predictions_df[pred_col] = model.predict(X_scaled)
        
        # Ensemble prediction (average of all models)
        pred_cols = [col for col in predictions_df.columns if '_prediction' in col]
        if pred_cols:
            predictions_df['ensemble_prediction'] = predictions_df[pred_cols].mean(axis=1)
            
            # Rank players by ensemble prediction
            predictions_df['predicted_rank'] = predictions_df['ensemble_prediction'].rank(ascending=False)
            
            # Sort by prediction
            predictions_df = predictions_df.sort_values('ensemble_prediction', ascending=False)
        
        return predictions_df
    
    def save_models(self, model_dir: str = 'models'):
        """Save trained models and scalers."""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            filepath = os.path.join(model_dir, f'{name}_model.joblib')
            joblib.dump(model, filepath)
        
        # Save scalers
        for name, scaler in self.scalers.items():
            filepath = os.path.join(model_dir, f'{name}_scaler.joblib')
            joblib.dump(scaler, filepath)
        
        # Save feature columns
        feature_filepath = os.path.join(model_dir, 'feature_columns.joblib')
        joblib.dump(self.feature_columns, feature_filepath)
        
        print(f"Models saved to {model_dir}")


if __name__ == "__main__":
    predictor = SkillBasedPredictor()
    
    # Load data
    data = predictor.load_data()
    
    # Prepare training data
    X, y = predictor.prepare_training_data(data)
    
    # Train models
    if not X.empty:
        results = predictor.train_models(X, y)
        
        # Generate predictions
        predictions = predictor.predict_us_open_2025(data)
        
        if not predictions.empty:
            print(f"\nTop 10 US Open 2025 Predictions:")
            print(predictions.head(10)[['player_name', 'ensemble_prediction', 'predicted_rank']])
            
            # Save predictions
            predictions.to_csv('data/predictions/us_open_2025_skill_based.csv', index=False)
            print(f"\nPredictions saved to data/predictions/us_open_2025_skill_based.csv")
        
        # Save models
        predictor.save_models()
    else:
        print("No training data available. Cannot train models.")
