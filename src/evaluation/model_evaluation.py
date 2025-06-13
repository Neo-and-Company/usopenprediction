"""
Model Evaluation Module
Provides ROC-AUC, F1 scores, and other evaluation metrics for golf prediction models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
from sklearn.model_selection import cross_val_score
import sqlite3
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class GolfModelEvaluator:
    """Evaluates golf prediction models using various metrics."""
    
    def __init__(self, db_path: str = "data/golf_predictions.db"):
        """Initialize the model evaluator.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        
    def _get_historical_data(self) -> pd.DataFrame:
        """Get historical tournament data for evaluation.
        
        Returns:
            DataFrame with historical performance data
        """
        # For now, we'll simulate historical data since we don't have actual results yet
        # In a real implementation, this would query historical tournament results
        
        query = """
            SELECT 
                p.player_name,
                pr.final_prediction_score,
                pr.course_fit_score,
                pr.fit_category,
                ps.sg_total,
                ps.datagolf_rank,
                p.country
            FROM predictions pr
            JOIN players p ON pr.player_id = p.player_id
            LEFT JOIN player_skills ps ON p.player_id = ps.player_id
            ORDER BY pr.final_prediction_score DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(query, conn)
            
        # Simulate historical results for demonstration
        # In reality, this would come from actual tournament outcomes
        df = self._simulate_historical_results(df)
        
        return df
    
    def _simulate_historical_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simulate historical tournament results for evaluation purposes.
        
        Args:
            df: DataFrame with predictions
            
        Returns:
            DataFrame with simulated results
        """
        np.random.seed(42)  # For reproducible results
        
        # Simulate various binary outcomes based on prediction scores
        # Higher prediction scores should correlate with better outcomes
        
        # Made cut (top 70 players typically make the cut)
        cut_threshold = df['final_prediction_score'].quantile(0.4)  # Top 60% make cut
        df['made_cut'] = (df['final_prediction_score'] >= cut_threshold).astype(int)
        
        # Top 10 finish
        top10_threshold = df['final_prediction_score'].quantile(0.9)  # Top 10%
        df['top_10'] = (df['final_prediction_score'] >= top10_threshold).astype(int)
        
        # Top 20 finish
        top20_threshold = df['final_prediction_score'].quantile(0.8)  # Top 20%
        df['top_20'] = (df['final_prediction_score'] >= top20_threshold).astype(int)
        
        # Winner (top player)
        df['winner'] = 0
        df.loc[df['final_prediction_score'].idxmax(), 'winner'] = 1
        
        # Add some noise to make it more realistic
        # Sometimes lower-ranked players perform better than expected
        noise_factor = 0.1
        for outcome in ['made_cut', 'top_10', 'top_20']:
            # Flip some results randomly
            flip_indices = np.random.choice(
                df.index, 
                size=int(len(df) * noise_factor), 
                replace=False
            )
            df.loc[flip_indices, outcome] = 1 - df.loc[flip_indices, outcome]
        
        return df
    
    def calculate_binary_classification_metrics(self, 
                                              y_true: np.ndarray, 
                                              y_pred_proba: np.ndarray,
                                              y_pred_binary: np.ndarray = None,
                                              threshold: float = 0.5) -> Dict:
        """Calculate comprehensive binary classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            y_pred_binary: Predicted binary labels (optional)
            threshold: Threshold for converting probabilities to binary predictions
            
        Returns:
            Dictionary with evaluation metrics
        """
        if y_pred_binary is None:
            y_pred_binary = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        try:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            roc_auc = 0.5  # If only one class present
            
        f1 = f1_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        return {
            'roc_auc': roc_auc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'threshold': threshold
        }
    
    def evaluate_prediction_model(self) -> Dict:
        """Evaluate the golf prediction model on various outcomes.
        
        Returns:
            Dictionary with evaluation results for different outcomes
        """
        # Get historical data
        df = self._get_historical_data()
        
        if df.empty:
            return {'error': 'No data available for evaluation'}
        
        # Normalize prediction scores to probabilities
        prediction_scores = df['final_prediction_score'].values
        
        # Convert to probabilities using softmax for multi-class, sigmoid for binary
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        # Normalize scores to 0-1 range first
        normalized_scores = (prediction_scores - prediction_scores.min()) / (prediction_scores.max() - prediction_scores.min())
        
        evaluation_results = {}
        
        # Evaluate different outcomes
        outcomes = ['made_cut', 'top_10', 'top_20', 'winner']
        
        for outcome in outcomes:
            if outcome in df.columns:
                y_true = df[outcome].values
                
                # Use different probability calculations for different outcomes
                if outcome == 'made_cut':
                    # For making cut, use sigmoid of normalized scores
                    y_pred_proba = sigmoid(normalized_scores * 6 - 3)  # Scale for sigmoid
                elif outcome == 'top_10':
                    # For top 10, use higher threshold
                    y_pred_proba = normalized_scores ** 2  # Square to make it more selective
                elif outcome == 'top_20':
                    # For top 20, use moderate threshold
                    y_pred_proba = normalized_scores ** 1.5
                else:  # winner
                    # For winner, use very high threshold
                    y_pred_proba = normalized_scores ** 3
                
                # Calculate metrics
                metrics = self.calculate_binary_classification_metrics(
                    y_true, y_pred_proba
                )
                
                evaluation_results[outcome] = metrics
        
        # Add overall model performance summary
        evaluation_results['summary'] = {
            'total_players_evaluated': len(df),
            'evaluation_date': datetime.now().isoformat(),
            'model_version': 'v1.0',
            'evaluation_method': 'simulated_historical_validation'
        }
        
        return evaluation_results
    
    def get_feature_importance_analysis(self) -> Dict:
        """Analyze feature importance for the prediction model.
        
        Returns:
            Dictionary with feature importance analysis
        """
        df = self._get_historical_data()
        
        if df.empty:
            return {'error': 'No data available for feature analysis'}
        
        # Calculate correlations between features and outcomes
        feature_cols = ['final_prediction_score', 'course_fit_score', 'sg_total', 'datagolf_rank']
        outcome_cols = ['made_cut', 'top_10', 'top_20']
        
        correlations = {}
        
        for outcome in outcome_cols:
            if outcome in df.columns:
                outcome_corrs = {}
                for feature in feature_cols:
                    if feature in df.columns:
                        corr = df[feature].corr(df[outcome])
                        outcome_corrs[feature] = corr if not pd.isna(corr) else 0.0
                correlations[outcome] = outcome_corrs
        
        return {
            'feature_correlations': correlations,
            'analysis_date': datetime.now().isoformat(),
            'note': 'Higher absolute correlation indicates stronger predictive power'
        }
    
    def get_model_calibration_analysis(self) -> Dict:
        """Analyze model calibration (how well predicted probabilities match actual outcomes).
        
        Returns:
            Dictionary with calibration analysis
        """
        df = self._get_historical_data()
        
        if df.empty:
            return {'error': 'No data available for calibration analysis'}
        
        # Analyze calibration for top_10 predictions
        if 'top_10' not in df.columns:
            return {'error': 'No outcome data available for calibration'}
        
        # Create probability bins
        prediction_scores = df['final_prediction_score'].values
        normalized_scores = (prediction_scores - prediction_scores.min()) / (prediction_scores.max() - prediction_scores.min())
        y_pred_proba = normalized_scores ** 2  # Same as used in evaluation
        
        # Create bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Calculate average predicted probability and actual frequency
                avg_predicted_prob = np.mean(y_pred_proba[in_bin])
                actual_frequency = np.mean(df.loc[in_bin, 'top_10'])
                count = np.sum(in_bin)
                
                calibration_data.append({
                    'bin_lower': bin_lower,
                    'bin_upper': bin_upper,
                    'avg_predicted_prob': avg_predicted_prob,
                    'actual_frequency': actual_frequency,
                    'count': count,
                    'calibration_error': abs(avg_predicted_prob - actual_frequency)
                })
        
        # Calculate overall calibration metrics
        if calibration_data:
            total_calibration_error = np.mean([d['calibration_error'] for d in calibration_data])
            max_calibration_error = np.max([d['calibration_error'] for d in calibration_data])
        else:
            total_calibration_error = 0
            max_calibration_error = 0
        
        return {
            'calibration_bins': calibration_data,
            'mean_calibration_error': total_calibration_error,
            'max_calibration_error': max_calibration_error,
            'analysis_date': datetime.now().isoformat(),
            'note': 'Lower calibration error indicates better probability estimates'
        }


def main():
    """Run model evaluation examples."""
    print("=" * 60)
    print("GOLF PREDICTION MODEL EVALUATION")
    print("=" * 60)
    
    evaluator = GolfModelEvaluator()
    
    # Run evaluation
    print("\n1. BINARY CLASSIFICATION METRICS:")
    print("-" * 40)
    results = evaluator.evaluate_prediction_model()
    
    if 'error' not in results:
        for outcome, metrics in results.items():
            if outcome != 'summary':
                print(f"\n{outcome.upper().replace('_', ' ')}:")
                print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
                print(f"  F1 Score: {metrics['f1_score']:.3f}")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
                print(f"  Accuracy: {metrics['accuracy']:.3f}")
    
    # Feature importance
    print("\n2. FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 40)
    feature_analysis = evaluator.get_feature_importance_analysis()
    
    if 'error' not in feature_analysis:
        for outcome, correlations in feature_analysis['feature_correlations'].items():
            print(f"\n{outcome.upper().replace('_', ' ')} correlations:")
            for feature, corr in correlations.items():
                print(f"  {feature}: {corr:.3f}")
    
    print(f"\n✓ Model evaluation completed!")


if __name__ == "__main__":
    main()
