"""
US Open Field Data Filter
Extracts actual tournament field from field updates and filters player data accordingly.
"""

import pandas as pd
import ast
from typing import List, Dict, Set
import warnings
warnings.filterwarnings('ignore')


class USOpenFieldFilter:
    """Filter player data to actual US Open field participants."""
    
    def __init__(self, field_updates_path: str = "data/raw/field_updates.csv"):
        """Initialize the field filter.
        
        Args:
            field_updates_path: Path to field updates CSV file
        """
        self.field_updates_path = field_updates_path
        self.tournament_field = None
        
    def extract_tournament_field(self) -> pd.DataFrame:
        """Extract the actual US Open field from field updates.
        
        Returns:
            DataFrame with tournament field players
        """
        print("Extracting US Open tournament field...")
        
        # Load field updates
        field_updates = pd.read_csv(self.field_updates_path)
        
        # Filter for US Open
        us_open_updates = field_updates[field_updates['event_name'] == 'U.S. Open'].copy()
        
        if us_open_updates.empty:
            raise ValueError("No U.S. Open field data found")
        
        # Parse field data
        field_players = []
        
        for _, row in us_open_updates.iterrows():
            try:
                # Parse the field string (it's a dictionary representation)
                field_dict = ast.literal_eval(row['field'])
                
                # Extract player information
                player_info = {
                    'player_name': field_dict.get('player_name', ''),
                    'dg_id': field_dict.get('dg_id', 0),
                    'country': field_dict.get('country', ''),
                    'am': field_dict.get('am', 0),  # Amateur status
                    'pga_number': field_dict.get('pga_number', 0),
                    'dk_salary': field_dict.get('dk_salary', 0),
                    'fd_salary': field_dict.get('fd_salary', 0),
                    'r1_teetime': field_dict.get('r1_teetime', ''),
                    'r2_teetime': field_dict.get('r2_teetime', ''),
                    'start_hole': field_dict.get('start_hole', 1),
                    'course': field_dict.get('course', 'Oakmont Country Club')
                }
                
                field_players.append(player_info)
                
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing field data: {e}")
                continue
        
        # Create DataFrame and remove duplicates
        self.tournament_field = pd.DataFrame(field_players)
        self.tournament_field = self.tournament_field.drop_duplicates(subset=['player_name'])
        
        print(f"✓ Extracted {len(self.tournament_field)} players in US Open field")
        
        return self.tournament_field
    
    def get_field_player_names(self) -> Set[str]:
        """Get set of player names in the tournament field.
        
        Returns:
            Set of player names
        """
        if self.tournament_field is None:
            self.extract_tournament_field()
        
        return set(self.tournament_field['player_name'].tolist())
    
    def filter_player_data(self, player_data: pd.DataFrame) -> pd.DataFrame:
        """Filter player data to only include tournament field participants.
        
        Args:
            player_data: Full player dataset
            
        Returns:
            Filtered player data for tournament field only
        """
        field_names = self.get_field_player_names()
        
        # Filter to field participants only
        filtered_data = player_data[player_data['player_name'].isin(field_names)].copy()
        
        print(f"✓ Filtered from {len(player_data)} to {len(filtered_data)} field participants")
        
        # Add tournament field information
        if not filtered_data.empty:
            filtered_data = filtered_data.merge(
                self.tournament_field[['player_name', 'dg_id', 'country', 'am', 'r1_teetime', 'r2_teetime']],
                on='player_name',
                how='left',
                suffixes=('', '_field')
            )
        
        return filtered_data
    
    def get_field_statistics(self) -> Dict:
        """Get statistics about the tournament field.
        
        Returns:
            Dictionary with field statistics
        """
        if self.tournament_field is None:
            self.extract_tournament_field()
        
        stats = {
            'total_players': len(self.tournament_field),
            'professionals': len(self.tournament_field[self.tournament_field['am'] == 0]),
            'amateurs': len(self.tournament_field[self.tournament_field['am'] == 1]),
            'countries_represented': self.tournament_field['country'].nunique(),
            'countries': self.tournament_field['country'].value_counts().to_dict()
        }
        
        return stats
    
    def validate_field_data(self, player_skills_data: pd.DataFrame) -> Dict:
        """Validate field data against available player skills.
        
        Args:
            player_skills_data: Player skills dataset
            
        Returns:
            Validation report
        """
        field_names = self.get_field_player_names()
        skills_names = set(player_skills_data['player_name'].tolist())
        
        # Find matches and mismatches
        matched_players = field_names.intersection(skills_names)
        field_missing_skills = field_names - skills_names
        skills_not_in_field = skills_names - field_names
        
        validation_report = {
            'field_players': len(field_names),
            'skills_players': len(skills_names),
            'matched_players': len(matched_players),
            'field_missing_skills': len(field_missing_skills),
            'skills_not_in_field': len(skills_not_in_field),
            'match_percentage': (len(matched_players) / len(field_names)) * 100,
            'missing_skills_players': list(field_missing_skills)[:10],  # Show first 10
            'extra_skills_players': list(skills_not_in_field)[:10]  # Show first 10
        }
        
        return validation_report


def main():
    """Test the field filter functionality."""
    print("=" * 60)
    print("US OPEN FIELD FILTER TEST")
    print("=" * 60)
    
    # Initialize filter
    field_filter = USOpenFieldFilter()
    
    # Extract tournament field
    tournament_field = field_filter.extract_tournament_field()
    
    # Show field statistics
    stats = field_filter.get_field_statistics()
    print(f"\nField Statistics:")
    print(f"  Total Players: {stats['total_players']}")
    print(f"  Professionals: {stats['professionals']}")
    print(f"  Amateurs: {stats['amateurs']}")
    print(f"  Countries: {stats['countries_represented']}")
    
    # Show top countries
    print(f"\nTop Countries:")
    for country, count in list(stats['countries'].items())[:5]:
        print(f"  {country}: {count} players")
    
    # Test with player skills data
    try:
        # Load skills data for validation
        skills_df = pd.read_csv('data/raw/current_skills.csv')
        
        # Parse skills data
        players_data = []
        for _, row in skills_df.iterrows():
            player_str = row['players']
            player_dict = ast.literal_eval(player_str)
            players_data.append(player_dict)
        
        skills_players_df = pd.DataFrame(players_data)
        
        # Validate
        validation = field_filter.validate_field_data(skills_players_df)
        
        print(f"\nValidation Report:")
        print(f"  Field Players: {validation['field_players']}")
        print(f"  Skills Players: {validation['skills_players']}")
        print(f"  Matched: {validation['matched_players']}")
        print(f"  Match Rate: {validation['match_percentage']:.1f}%")
        
        if validation['missing_skills_players']:
            print(f"  Missing Skills (sample): {validation['missing_skills_players']}")
        
        # Filter the data
        filtered_data = field_filter.filter_player_data(skills_players_df)
        print(f"\n✓ Filtered dataset ready: {len(filtered_data)} players")
        
    except Exception as e:
        print(f"Error in validation: {e}")


if __name__ == "__main__":
    main()
