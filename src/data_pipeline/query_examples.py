"""
Database Query Examples
Demonstrates SQL queries for the golf prediction database with improved architecture.
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Optional, Any
import logging


class GolfDB:
    """Database context manager for golf predictions."""

    def __init__(self, db_path: str = "data/golf_predictions.db"):
        """Initialize database connection manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.conn = None

    def __enter__(self):
        """Context manager entry - establish connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            return self
        except sqlite3.Error as e:
            logging.error(f"Database connection failed: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close connection."""
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

    def _execute_query(self, query: str, params: tuple = (),
                      return_df: bool = True) -> Optional[pd.DataFrame]:
        """Centralized query execution with error handling.

        Args:
            query: SQL query string
            params: Query parameters
            return_df: Whether to return DataFrame or raw result

        Returns:
            DataFrame with query results or None on error
        """
        try:
            if return_df:
                return pd.read_sql_query(query, self.conn, params=params)
            else:
                cursor = self.conn.execute(query, params)
                return cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Query execution failed: {e}")
            logging.error(f"Query: {query}")
            logging.error(f"Params: {params}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error in query execution: {e}")
            return None


class GolfPredictionQueries:
    """Enhanced queries for the golf prediction database."""

    def __init__(self, db_path: str = "data/golf_predictions.db"):
        """Initialize query interface.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
    
    def get_top_predictions(self, limit: int = 15) -> pd.DataFrame:
        """Get top predictions for the tournament.

        Args:
            limit: Number of top predictions to return

        Returns:
            DataFrame with top predictions
        """
        query = """
            SELECT
                p.player_name,
                pr.final_prediction_score,
                pr.course_fit_score,
                pr.course_fit_penalty AS fit_multiplier,
                pr.general_form_score,
                pr.fit_category,
                ps.sg_total,
                ps.datagolf_rank,
                p.country
            FROM predictions pr
            JOIN players p ON pr.player_id = p.player_id
            LEFT JOIN player_skills ps ON p.player_id = ps.player_id
            ORDER BY pr.final_prediction_score DESC
            LIMIT ?
        """

        with GolfDB(self.db_path) as db:
            return db._execute_query(query, (limit,))
    
    def get_players_by_course_fit(self, fit_category: str) -> pd.DataFrame:
        """Get players by course fit category.

        Args:
            fit_category: Course fit category (Good Fit, Average Fit, Poor Fit)

        Returns:
            DataFrame with players in the category
        """
        query = """
            SELECT
                p.player_name,
                pr.course_fit_score,
                pr.final_prediction_score,
                pr.course_fit_penalty AS fit_multiplier,
                ps.sg_total,
                ps.datagolf_rank
            FROM predictions pr
            JOIN players p ON pr.player_id = p.player_id
            LEFT JOIN player_skills ps ON p.player_id = ps.player_id
            WHERE pr.fit_category = ?
            ORDER BY pr.course_fit_score DESC
        """

        with GolfDB(self.db_path) as db:
            return db._execute_query(query, (fit_category,))

    def find_value_picks(self, min_rank_improvement: int = 10) -> pd.DataFrame:
        """Find value picks - players ranked much higher by model than world ranking.

        Args:
            min_rank_improvement: Minimum rank improvement to be considered a value pick

        Returns:
            DataFrame with value picks
        """
        query = """
            WITH ranked_predictions AS (
                SELECT
                    p.player_name,
                    ps.datagolf_rank,
                    pr.final_prediction_score,
                    pr.course_fit_score,
                    pr.fit_category,
                    pr.course_fit_penalty AS fit_multiplier,
                    RANK() OVER (ORDER BY pr.final_prediction_score DESC) as prediction_rank
                FROM predictions pr
                JOIN players p ON pr.player_id = p.player_id
                JOIN player_skills ps ON p.player_id = ps.player_id
                WHERE ps.datagolf_rank IS NOT NULL
            )
            SELECT
                player_name,
                datagolf_rank,
                prediction_rank,
                (datagolf_rank - prediction_rank) as rank_improvement,
                final_prediction_score,
                course_fit_score,
                fit_category,
                fit_multiplier
            FROM ranked_predictions
            WHERE (datagolf_rank - prediction_rank) >= ?
            ORDER BY rank_improvement DESC
        """

        with GolfDB(self.db_path) as db:
            return db._execute_query(query, (min_rank_improvement,))

    def get_elite_players_analysis(self) -> pd.DataFrame:
        """Get analysis of elite players (top 20 in world).

        Returns:
            DataFrame with elite player analysis
        """
        query = """
            SELECT
                p.player_name,
                ps.datagolf_rank,
                ps.sg_total,
                pr.course_fit_score,
                pr.fit_category,
                pr.course_fit_penalty,
                pr.final_prediction_score,
                RANK() OVER (ORDER BY pr.final_prediction_score DESC) as prediction_rank
            FROM predictions pr
            JOIN players p ON pr.player_id = p.player_id
            JOIN player_skills ps ON p.player_id = ps.player_id
            WHERE ps.datagolf_rank <= 20
            ORDER BY ps.datagolf_rank
        """

        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_course_fit_vs_ranking_analysis(self) -> pd.DataFrame:
        """Analyze relationship between course fit and world ranking.
        
        Returns:
            DataFrame with course fit vs ranking analysis
        """
        query = """
            SELECT
                pr.fit_category,
                COUNT(*) as player_count,
                AVG(ps.datagolf_rank) as avg_world_rank,
                AVG(pr.course_fit_score) as avg_fit_score,
                AVG(pr.final_prediction_score) as avg_prediction_score,
                MIN(pr.final_prediction_score) as min_prediction,
                MAX(pr.final_prediction_score) as max_prediction
            FROM predictions pr
            JOIN players pl ON pr.player_id = pl.player_id
            LEFT JOIN player_skills ps ON pl.player_id = ps.player_id
            GROUP BY pr.fit_category
            ORDER BY avg_prediction_score DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)
    
    def get_tournament_field_summary(self) -> Dict:
        """Get tournament field summary statistics.
        
        Returns:
            Dictionary with field summary
        """
        queries = {
            'total_players': "SELECT COUNT(*) as count FROM tournament_field",
            'professionals': "SELECT COUNT(*) as count FROM players WHERE amateur_status = 0",
            'amateurs': "SELECT COUNT(*) as count FROM players WHERE amateur_status = 1",
            'countries': "SELECT COUNT(DISTINCT country) as count FROM players",
            'with_predictions': "SELECT COUNT(*) as count FROM predictions",
            'with_skills_data': "SELECT COUNT(*) as count FROM player_skills"
        }
        
        summary = {}
        
        with sqlite3.connect(self.db_path) as conn:
            for key, query in queries.items():
                result = pd.read_sql_query(query, conn)
                summary[key] = result.iloc[0]['count']
        
        return summary
    
    def get_country_representation(self) -> pd.DataFrame:
        """Get country representation in the field.
        
        Returns:
            DataFrame with country representation
        """
        query = """
            SELECT
                pl.country,
                COUNT(*) as player_count,
                COUNT(pr.prediction_id) as players_with_predictions,
                AVG(pr.final_prediction_score) as avg_prediction_score,
                MAX(pr.final_prediction_score) as best_prediction
            FROM players pl
            LEFT JOIN predictions pr ON pl.player_id = pr.player_id
            GROUP BY pl.country
            HAVING COUNT(*) > 1
            ORDER BY player_count DESC
        """
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn)


def display_dataframe(df: pd.DataFrame, title: str, max_rows: int = 10):
    """Display DataFrame with professional formatting.

    Args:
        df: DataFrame to display
        title: Title for the display
        max_rows: Maximum number of rows to show
    """
    print(f"\n{title}")
    print("-" * len(title))
    if not df.empty:
        print(df.head(max_rows).to_string(index=False))
    else:
        print("No data available")


def main():
    """Run example queries and display results."""
    print("=" * 80)
    print("GOLF PREDICTION DATABASE QUERIES")
    print("=" * 80)

    queries = GolfPredictionQueries()

    # Top predictions
    print("\n1. TOP 10 PREDICTIONS:")
    print("-" * 40)
    top_preds = queries.get_top_predictions(10)
    for i, (_, player) in enumerate(top_preds.iterrows(), 1):
        print(f"{i:2d}. {player['player_name']:<20} | Score: {player['final_prediction_score']:.3f} | "
              f"Fit: {player['course_fit_score']:.3f} ({player['fit_category']}) | "
              f"Rank: {player['datagolf_rank']}")

    # Value picks - NEW FEATURE
    print("\n2. VALUE PICKS (Model rank >> World rank):")
    print("-" * 50)
    value_picks = queries.find_value_picks(min_rank_improvement=10)
    display_dataframe(value_picks, "Players with 10+ rank improvement", 10)

    # Elite players analysis
    print("\n3. ELITE PLAYERS (Top 20 World Ranking) ANALYSIS:")
    print("-" * 60)
    elite = queries.get_elite_players_analysis()
    for _, player in elite.iterrows():
        print(f"#{player['datagolf_rank']:2d} {player['player_name']:<20} | "
              f"Pred Rank: #{player['prediction_rank']:2d} | "
              f"Fit: {player['course_fit_score']:.3f} ({player['fit_category']}) | "
              f"Penalty: {player['course_fit_penalty']:.3f}")

    # Course fit analysis
    print("\n4. COURSE FIT CATEGORY ANALYSIS:")
    print("-" * 50)
    fit_analysis = queries.get_course_fit_vs_ranking_analysis()
    display_dataframe(fit_analysis, "Course Fit vs Performance Analysis")

    # Tournament summary
    print("\n5. TOURNAMENT FIELD SUMMARY:")
    print("-" * 40)
    summary = queries.get_tournament_field_summary()
    for key, value in summary.items():
        print(f"{key.replace('_', ' ').title()}: {value}")

    # Country representation
    print("\n6. COUNTRY REPRESENTATION (2+ players):")
    print("-" * 50)
    countries = queries.get_country_representation()
    display_dataframe(countries.head(10), "Top Countries by Player Count")

    print(f"\n✓ Database queries completed successfully!")


if __name__ == "__main__":
    main()
