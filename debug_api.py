#!/usr/bin/env python3
"""
Debug script to inspect DataGolf API response structure
"""

import sys
import os
sys.path.append('src')

def debug_api():
    print("🔍 DataGolf API Structure Debug")
    print("=" * 40)
    
    try:
        from data_collection.datagolf_client import DataGolfClient
        
        client = DataGolfClient()
        
        print("1. Testing get_dg_rankings()...")
        rankings = client.get_dg_rankings()
        print(f"   Type: {type(rankings)}")
        print(f"   Shape: {rankings.shape}")
        print(f"   Columns: {list(rankings.columns)}")
        print(f"   First row:")
        print(rankings.iloc[0])
        print()
        
        print("2. Testing get_skill_ratings()...")
        skills = client.get_skill_ratings()
        print(f"   Type: {type(skills)}")
        print(f"   Shape: {skills.shape}")
        print(f"   Columns: {list(skills.columns)}")
        print(f"   First row:")
        print(skills.iloc[0])
        print()
        
        print("3. Testing get_player_list()...")
        players = client.get_player_list()
        print(f"   Type: {type(players)}")
        print(f"   Shape: {players.shape}")
        print(f"   Columns: {list(players.columns)}")
        print(f"   First row:")
        print(players.iloc[0])
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_api()
