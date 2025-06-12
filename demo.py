#!/usr/bin/env python3
"""
Quick demo of the US Open 2025 Prediction System
"""

import sys
import os
sys.path.append('src')

def main():
    print("🏌️ US Open 2025 Prediction System Demo")
    print("=" * 50)
    
    try:
        from data_collection.datagolf_client import DataGolfClient
        
        print("🔌 Connecting to DataGolf API...")
        client = DataGolfClient()
        
        print("📊 Fetching current player rankings...")
        rankings = client.get_dg_rankings()

        print(f"✅ Successfully retrieved {len(rankings)} player rankings")
        print(f"📋 Available columns: {rankings.columns.tolist()}")
        print()

        # Inspect the first few rows to understand the data structure
        print("🔍 Sample data structure:")
        print(rankings.head(2))
        print()

        # Find the correct column names
        name_col = None
        rank_col = None
        skill_col = None

        # Common variations for player name column
        name_candidates = ['player_name', 'dg_player_name', 'name', 'full_name', 'player']
        for col in name_candidates:
            if col in rankings.columns:
                name_col = col
                break

        # Common variations for rank column
        rank_candidates = ['rank', 'dg_rank', 'ranking', 'position']
        for col in rank_candidates:
            if col in rankings.columns:
                rank_col = col
                break

        # Common variations for skill column
        skill_candidates = ['skill_estimate', 'skill_rating', 'skill', 'rating']
        for col in skill_candidates:
            if col in rankings.columns:
                skill_col = col
                break

        print("🏆 Top 10 Players (DataGolf Rankings):")
        print("-" * 40)

        for i in range(min(10, len(rankings))):
            player = rankings.iloc[i]

            # Use the identified column names or fallback values
            name = player[name_col] if name_col else f"Player {i+1}"
            rank = player[rank_col] if rank_col else i+1
            skill = player[skill_col] if skill_col else 0

            print(f"{rank:2d}. {name:<25} (Skill: {skill:.2f})")
        
        print()
        print("🎯 Fetching current skill ratings...")
        skills = client.get_skill_ratings()
        print(f"✅ Retrieved skill ratings for {len(skills)} players")
        print(f"📋 Skills columns: {skills.columns.tolist()}")

        # Find player name column in skills data
        skills_name_col = None
        for col in name_candidates:
            if col in skills.columns:
                skills_name_col = col
                break

        # Find SG Total column
        sg_total_col = None
        sg_candidates = ['sg_total', 'strokes_gained_total', 'total_sg', 'sg_t']
        for col in sg_candidates:
            if col in skills.columns:
                sg_total_col = col
                break

        if sg_total_col and len(skills) > 0:
            print()
            print("📈 Top 5 Players by Strokes Gained Total:")
            print("-" * 45)
            top_skills = skills.nlargest(5, sg_total_col)
            for i, (_, player) in enumerate(top_skills.iterrows(), 1):
                name = player[skills_name_col] if skills_name_col else f"Player {i}"
                sg_total = player[sg_total_col]
                print(f"{i:2d}. {name:<25} (SG Total: {sg_total:.2f})")
        else:
            print("📊 SG Total data structure:")
            print(skills.head(2) if len(skills) > 0 else "No skills data available")
        
        print()
        print("📈 System Capabilities:")
        print("  ✓ Historical data collection (2017-2024)")
        print("  ✓ Advanced feature engineering")
        print("  ✓ Machine learning models (XGBoost, Random Forest)")
        print("  ✓ Multiple prediction targets (Win, Top 5, Top 10, Top 20)")
        print("  ✓ Comprehensive analysis and reporting")
        
        print()
        print("🚀 Ready to generate US Open 2025 predictions!")
        print("📝 To run the full system:")
        print("   python main.py --step full")
        print()
        print("📊 For detailed analysis:")
        print("   jupyter notebook notebooks/us_open_analysis.ipynb")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("🔧 Troubleshooting:")
        print("  1. Check your DataGolf API key in .env file")
        print("  2. Ensure internet connection")
        print("  3. Verify all dependencies are installed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
