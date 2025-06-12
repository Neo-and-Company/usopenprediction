#!/usr/bin/env python3
"""
Test script to check DataGolf API access and available historical data.
"""

import sys
import os
sys.path.append('src')

from data_collection.datagolf_client import DataGolfClient
import pandas as pd

def test_api_access():
    """Test what data we can access with current API key."""
    print("Testing DataGolf API Access...")
    print("=" * 50)
    
    client = DataGolfClient()
    
    # Test 1: Current rankings (should work)
    print("\n1. Testing current rankings...")
    try:
        rankings = client.get_current_rankings()
        print(f"✅ Current rankings: {len(rankings)} players")
        print(f"   Columns: {list(rankings.columns)}")
    except Exception as e:
        print(f"❌ Current rankings failed: {e}")
    
    # Test 2: Historical events list
    print("\n2. Testing historical events list...")
    try:
        events = client.get_historical_events()
        print(f"✅ Historical events: {len(events)} events")
        print(f"   Columns: {list(events.columns)}")
        
        # Show some recent events
        if not events.empty:
            print("\n   Recent events:")
            recent_events = events.head(10)
            for _, event in recent_events.iterrows():
                print(f"   - {event.get('event_name', 'Unknown')}: ID {event.get('event_id', 'N/A')}")
                
    except Exception as e:
        print(f"❌ Historical events failed: {e}")
    
    # Test 3: Try to get specific tournament data
    print("\n3. Testing specific tournament data access...")
    
    # Try US Open 2024 (event_id=26 based on previous attempts)
    try:
        us_open_2024 = client.get_historical_data(
            tour='pga',
            event_id=26,
            year=2024
        )
        print(f"✅ US Open 2024: {len(us_open_2024)} records")
        print(f"   Columns: {list(us_open_2024.columns)}")
        
    except Exception as e:
        print(f"❌ US Open 2024 failed: {e}")
    
    # Test 4: Try different event ID or approach
    print("\n4. Testing alternative data access...")
    try:
        # Try getting any recent PGA tour data
        recent_pga = client.get_historical_data(
            tour='pga',
            event_id='all',  # Try getting all events
            year=2024
        )
        print(f"✅ PGA 2024 (all events): {len(recent_pga)} records")
        
    except Exception as e:
        print(f"❌ PGA 2024 (all events) failed: {e}")
    
    # Test 5: Check what endpoints are actually available
    print("\n5. Testing basic API connectivity...")
    try:
        # Try the simplest endpoint
        response = client._make_request('pga-tour-rankings', {})
        print(f"✅ Basic API connectivity works")
        print(f"   Response type: {type(response)}")
        
    except Exception as e:
        print(f"❌ Basic API connectivity failed: {e}")

if __name__ == "__main__":
    test_api_access()
