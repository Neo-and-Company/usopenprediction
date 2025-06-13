"""
Golf Prediction Database Manager
Handles SQLite database creation, data loading, and queries for the golf prediction system.
"""

import sqlite3
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')


class GolfPredictionDB:
    """Database manager for golf prediction system."""
    
    def __init__(self, db_path: str = "data/golf_predictions.db"):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
    def connect(self):
        """Connect to the database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
    def disconnect(self):
        """Disconnect from the database."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def create_schema(self):
        """Create database schema for golf predictions."""
        print("Creating database schema...")
        
        # Players table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS players (
                player_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT UNIQUE NOT NULL,
                dg_id INTEGER,
                country TEXT,
                pga_number INTEGER,
                amateur_status INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Player skills table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS player_skills (
                skill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_id INTEGER,
                sg_total REAL,
                sg_ott REAL,
                sg_app REAL,
                sg_arg REAL,
                sg_putt REAL,
                driving_dist REAL,
                driving_acc REAL,
                datagolf_rank INTEGER,
                dg_skill_estimate REAL,
                data_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Tournaments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tournaments (
                tournament_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_name TEXT NOT NULL,
                course_name TEXT NOT NULL,
                start_date DATE,
                end_date DATE,
                field_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tournament field table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tournament_field (
                field_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER,
                player_id INTEGER,
                r1_teetime TEXT,
                r2_teetime TEXT,
                start_hole INTEGER,
                dk_salary REAL,
                fd_salary REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tournament_id) REFERENCES tournaments (tournament_id),
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Course conditions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS course_conditions (
                condition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_name TEXT NOT NULL,
                tournament_id INTEGER,
                green_speed REAL,
                green_firmness REAL,
                rough_height REAL,
                bunker_penalty REAL,
                course_length REAL,
                weather_conditions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (tournament_id) REFERENCES tournaments (tournament_id)
            )
        """)
        
        # Predictions table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tournament_id INTEGER,
                player_id INTEGER,
                prediction_type TEXT,
                final_prediction_score REAL,
                course_fit_score REAL,
                course_fit_penalty REAL,
                historical_performance_score REAL,
                general_form_score REAL,
                reliability_factor REAL,
                confidence_level REAL,
                fit_category TEXT,
                key_advantages TEXT,
                key_vulnerabilities TEXT,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT,
                FOREIGN KEY (tournament_id) REFERENCES tournaments (tournament_id),
                FOREIGN KEY (player_id) REFERENCES players (player_id)
            )
        """)
        
        # Create indexes for better performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_players_name ON players (player_name)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_players_dg_id ON players (dg_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_skills_player ON player_skills (player_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_tournament ON predictions (tournament_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions (player_id)")
        
        self.conn.commit()
        print("✓ Database schema created successfully")
    
    def load_players_data(self, players_df: pd.DataFrame) -> int:
        """Load players data into database.
        
        Args:
            players_df: DataFrame with player information
            
        Returns:
            Number of players loaded
        """
        print("Loading players data...")
        
        players_loaded = 0
        
        for _, player in players_df.iterrows():
            try:
                self.conn.execute("""
                    INSERT OR REPLACE INTO players 
                    (player_name, dg_id, country, pga_number, amateur_status, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """, (
                    player.get('player_name', ''),
                    player.get('dg_id', None),
                    player.get('country', ''),
                    player.get('pga_number', None),
                    player.get('am', 0)
                ))
                players_loaded += 1
                
            except Exception as e:
                print(f"Error loading player {player.get('player_name', 'Unknown')}: {e}")
        
        self.conn.commit()
        print(f"✓ Loaded {players_loaded} players")
        return players_loaded
    
    def load_player_skills(self, skills_df: pd.DataFrame) -> int:
        """Load player skills data into database.
        
        Args:
            skills_df: DataFrame with player skills
            
        Returns:
            Number of skill records loaded
        """
        print("Loading player skills...")
        
        skills_loaded = 0
        current_date = datetime.now().date()
        
        for _, player in skills_df.iterrows():
            try:
                # Get player_id
                player_id = self.get_player_id(player.get('player_name', ''))
                if not player_id:
                    continue
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO player_skills 
                    (player_id, sg_total, sg_ott, sg_app, sg_arg, sg_putt, 
                     driving_dist, driving_acc, datagolf_rank, dg_skill_estimate, data_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    player_id,
                    player.get('sg_total', None),
                    player.get('sg_ott', None),
                    player.get('sg_app', None),
                    player.get('sg_arg', None),
                    player.get('sg_putt', None),
                    player.get('driving_dist', None),
                    player.get('driving_acc', None),
                    player.get('datagolf_rank', None),
                    player.get('dg_skill_estimate', None),
                    current_date
                ))
                skills_loaded += 1
                
            except Exception as e:
                print(f"Error loading skills for {player.get('player_name', 'Unknown')}: {e}")
        
        self.conn.commit()
        print(f"✓ Loaded {skills_loaded} skill records")
        return skills_loaded
    
    def create_tournament(self, tournament_name: str, course_name: str, 
                         start_date: str, field_size: int = 156) -> int:
        """Create a tournament record.
        
        Args:
            tournament_name: Name of the tournament
            course_name: Name of the course
            start_date: Tournament start date
            field_size: Number of players in field
            
        Returns:
            Tournament ID
        """
        cursor = self.conn.execute("""
            INSERT INTO tournaments (tournament_name, course_name, start_date, field_size)
            VALUES (?, ?, ?, ?)
        """, (tournament_name, course_name, start_date, field_size))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def load_tournament_field(self, tournament_id: int, field_df: pd.DataFrame) -> int:
        """Load tournament field data.
        
        Args:
            tournament_id: Tournament ID
            field_df: DataFrame with field information
            
        Returns:
            Number of field entries loaded
        """
        print("Loading tournament field...")
        
        field_loaded = 0
        
        for _, player in field_df.iterrows():
            try:
                player_id = self.get_player_id(player.get('player_name', ''))
                if not player_id:
                    continue
                
                self.conn.execute("""
                    INSERT OR REPLACE INTO tournament_field 
                    (tournament_id, player_id, r1_teetime, r2_teetime, start_hole, dk_salary, fd_salary)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    tournament_id,
                    player_id,
                    player.get('r1_teetime', ''),
                    player.get('r2_teetime', ''),
                    player.get('start_hole', 1),
                    player.get('dk_salary', None),
                    player.get('fd_salary', None)
                ))
                field_loaded += 1
                
            except Exception as e:
                print(f"Error loading field entry for {player.get('player_name', 'Unknown')}: {e}")
        
        self.conn.commit()
        print(f"✓ Loaded {field_loaded} field entries")
        return field_loaded
    
    def get_player_id(self, player_name: str) -> Optional[int]:
        """Get player ID by name.
        
        Args:
            player_name: Player name
            
        Returns:
            Player ID or None if not found
        """
        cursor = self.conn.execute(
            "SELECT player_id FROM players WHERE player_name = ?", 
            (player_name,)
        )
        result = cursor.fetchone()
        return result['player_id'] if result else None
    
    def get_tournament_field_data(self, tournament_id: int) -> pd.DataFrame:
        """Get tournament field data with player information.
        
        Args:
            tournament_id: Tournament ID
            
        Returns:
            DataFrame with field data
        """
        query = """
            SELECT 
                p.player_name,
                p.dg_id,
                p.country,
                p.amateur_status,
                ps.sg_total,
                ps.sg_ott,
                ps.sg_app,
                ps.sg_arg,
                ps.sg_putt,
                ps.driving_dist,
                ps.driving_acc,
                ps.datagolf_rank,
                ps.dg_skill_estimate,
                tf.r1_teetime,
                tf.r2_teetime,
                tf.start_hole,
                tf.dk_salary,
                tf.fd_salary
            FROM tournament_field tf
            JOIN players p ON tf.player_id = p.player_id
            LEFT JOIN player_skills ps ON p.player_id = ps.player_id
            WHERE tf.tournament_id = ?
            ORDER BY p.player_name
        """
        
        return pd.read_sql_query(query, self.conn, params=(tournament_id,))
    
    def save_predictions(self, tournament_id: int, predictions_df: pd.DataFrame, 
                        model_version: str = "v1.0") -> int:
        """Save prediction results to database.
        
        Args:
            tournament_id: Tournament ID
            predictions_df: DataFrame with predictions
            model_version: Model version identifier
            
        Returns:
            Number of predictions saved
        """
        print("Saving predictions...")
        
        predictions_saved = 0
        
        for _, pred in predictions_df.iterrows():
            try:
                player_id = self.get_player_id(pred.get('player_name', ''))
                if not player_id:
                    continue
                
                self.conn.execute("""
                    INSERT INTO predictions 
                    (tournament_id, player_id, prediction_type, final_prediction_score,
                     course_fit_score, course_fit_penalty, historical_performance_score,
                     general_form_score, reliability_factor, confidence_level,
                     fit_category, key_advantages, key_vulnerabilities, model_version)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    tournament_id,
                    player_id,
                    "enhanced_course_prediction",
                    pred.get('final_prediction_score', None),
                    pred.get('course_fit_score', None),
                    pred.get('course_fit_penalty', None),
                    pred.get('historical_performance_score', None),
                    pred.get('general_form_score', None),
                    pred.get('reliability_factor', None),
                    pred.get('confidence_level', None),
                    pred.get('fit_category', ''),
                    json.dumps(pred.get('key_advantages', [])),
                    json.dumps(pred.get('key_vulnerabilities', [])),
                    model_version
                ))
                predictions_saved += 1
                
            except Exception as e:
                print(f"Error saving prediction for {pred.get('player_name', 'Unknown')}: {e}")
        
        self.conn.commit()
        print(f"✓ Saved {predictions_saved} predictions")
        return predictions_saved
    
    def get_database_stats(self) -> Dict:
        """Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        # Count records in each table
        tables = ['players', 'player_skills', 'tournaments', 'tournament_field', 'predictions']
        
        for table in tables:
            cursor = self.conn.execute(f"SELECT COUNT(*) as count FROM {table}")
            stats[table] = cursor.fetchone()['count']
        
        return stats
