"""
Test script for the skill-based US Open prediction model.
"""

import pandas as pd
import numpy as np
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def parse_json_column(df, column_name):
    """Parse JSON strings in a DataFrame column."""
    parsed_data = []
    for idx, row in df.iterrows():
        try:
            if pd.notna(row[column_name]):
                # Parse the JSON string
                data = eval(row[column_name])  # Using eval since it's a dict string
                parsed_data.append(data)
            else:
                parsed_data.append({})
        except:
            parsed_data.append({})
    
    return pd.DataFrame(parsed_data)


def load_and_process_data():
    """Load and process the collected DataGolf data."""
    
    # Load rankings data
    rankings_df = pd.read_csv('data/raw/current_rankings.csv')
    rankings_parsed = parse_json_column(rankings_df, 'rankings')
    
    # Load skill ratings data  
    skills_df = pd.read_csv('data/raw/skill_ratings.csv')
    skills_parsed = parse_json_column(skills_df, 'players')
    
    # Load player data
    players_df = pd.read_csv('data/raw/players.csv')
    
    print(f"Loaded {len(rankings_parsed)} rankings")
    print(f"Loaded {len(skills_parsed)} skill ratings")
    print(f"Loaded {len(players_df)} players")
    
    return rankings_parsed, skills_parsed, players_df


def create_features(rankings_df, skills_df, players_df):
    """Create features for modeling."""
    
    # Start with rankings data
    features_df = rankings_df.copy()
    
    # Merge with skills data on dg_id
    if 'dg_id' in features_df.columns and 'dg_id' in skills_df.columns:
        features_df = features_df.merge(skills_df, on='dg_id', how='left', suffixes=('', '_skill'))
    
    # Add player metadata
    if 'dg_id' in features_df.columns and 'dg_id' in players_df.columns:
        features_df = features_df.merge(players_df, on='dg_id', how='left', suffixes=('', '_player'))
    
    # Create derived features
    if 'datagolf_rank' in features_df.columns:
        features_df['rank_log'] = np.log(features_df['datagolf_rank'] + 1)
        features_df['rank_inverse'] = 1 / (features_df['datagolf_rank'] + 1)
        features_df['top_10_player'] = (features_df['datagolf_rank'] <= 10).astype(int)
        features_df['top_50_player'] = (features_df['datagolf_rank'] <= 50).astype(int)
    
    # Skill-based features
    skill_cols = [col for col in features_df.columns if 'sg_' in col.lower()]
    if skill_cols:
        features_df['skill_composite'] = features_df[skill_cols].mean(axis=1)
        features_df['skill_consistency'] = -features_df[skill_cols].std(axis=1)
    
    # Country-based features
    if 'country' in features_df.columns:
        major_golf_countries = ['USA', 'ENG', 'SCO', 'AUS', 'RSA', 'ESP', 'IRL']
        features_df['major_golf_country'] = features_df['country'].isin(major_golf_countries).astype(int)
    
    # Amateur status
    if 'am' in features_df.columns:
        features_df['is_professional'] = (features_df['am'] == 0).astype(int)
    
    print(f"Created features dataset with {features_df.shape[0]} rows and {features_df.shape[1]} columns")
    return features_df


def create_synthetic_target(features_df):
    """Create a synthetic target variable for demonstration."""
    # Use a combination of ranking and skills to create a "US Open performance" target
    
    target = np.zeros(len(features_df))
    
    # Base performance from ranking (inverse relationship)
    if 'datagolf_rank' in features_df.columns:
        target += 100 / (features_df['datagolf_rank'] + 1)
    
    # Add skill components
    if 'sg_total' in features_df.columns:
        target += features_df['sg_total'].fillna(0) * 10
    
    # Add some randomness to simulate real tournament variability
    np.random.seed(42)
    target += np.random.normal(0, 5, len(features_df))
    
    return target


def train_model(features_df, target):
    """Train a random forest model."""
    
    # Select numeric features only
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    X = features_df[numeric_cols].fillna(0)
    
    print(f"Using {len(numeric_cols)} numeric features: {list(numeric_cols)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"Training RMSE: {train_rmse:.3f}")
    print(f"Test RMSE: {test_rmse:.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return model, scaler, numeric_cols


def make_predictions(model, scaler, features_df, feature_cols, target):
    """Make US Open 2025 predictions."""
    
    X = features_df[feature_cols].fillna(0)
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    
    # Create results dataframe
    results_df = features_df[['player_name', 'dg_id', 'datagolf_rank', 'dg_skill_estimate']].copy()
    results_df['predicted_score'] = predictions
    results_df['actual_target'] = target
    
    # Rank by prediction
    results_df['predicted_rank'] = results_df['predicted_score'].rank(ascending=False)
    
    # Sort by prediction
    results_df = results_df.sort_values('predicted_score', ascending=False)
    
    return results_df


def main():
    """Main function to run the test."""
    
    print("=" * 60)
    print("US OPEN 2025 SKILL-BASED PREDICTION MODEL TEST")
    print("=" * 60)
    
    # Load data
    rankings_df, skills_df, players_df = load_and_process_data()
    
    # Create features
    features_df = create_features(rankings_df, skills_df, players_df)
    
    # Create synthetic target
    target = create_synthetic_target(features_df)
    
    # Train model
    model, scaler, feature_cols = train_model(features_df, target)
    
    # Make predictions
    predictions_df = make_predictions(model, scaler, features_df, feature_cols, target)
    
    print("\n" + "=" * 60)
    print("TOP 20 US OPEN 2025 PREDICTIONS")
    print("=" * 60)
    
    top_20 = predictions_df.head(20)
    for idx, row in top_20.iterrows():
        print(f"{int(row['predicted_rank']):2d}. {row['player_name']:<25} "
              f"(DG Rank: {int(row['datagolf_rank']):3d}, "
              f"Skill: {row['dg_skill_estimate']:.2f}, "
              f"Pred: {row['predicted_score']:.1f})")
    
    # Save predictions
    os.makedirs('data/predictions', exist_ok=True)
    predictions_df.to_csv('data/predictions/us_open_2025_test_predictions.csv', index=False)
    print(f"\nPredictions saved to data/predictions/us_open_2025_test_predictions.csv")
    
    print("\n" + "=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
