#!/usr/bin/env python3
"""
Test script for Vercel deployment
Verifies that the app works correctly in serverless environment
"""

import requests
import json
import sys
from datetime import datetime

def test_vercel_app(base_url):
    """Test the Vercel-deployed app endpoints."""
    
    print(f"🧪 Testing Golf Prediction App at: {base_url}")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test endpoints
    endpoints = [
        ("/api/health", "Health Check"),
        ("/", "Main Dashboard"),
        ("/predictions", "Predictions Page"),
        ("/value-picks", "Value Picks Page"),
        ("/analytics", "Analytics Page"),
        ("/evaluation", "Model Evaluation Page"),
        ("/api/predictions", "Predictions API"),
        ("/api/model-evaluation", "Model Evaluation API")
    ]
    
    for endpoint, description in endpoints:
        tests_total += 1
        url = f"{base_url}{endpoint}"
        
        try:
            print(f"Testing {description}... ", end="")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                print("✅ PASS")
                tests_passed += 1
                
                # Additional checks for API endpoints
                if endpoint.startswith("/api/"):
                    try:
                        data = response.json()
                        if "status" in data and data["status"] in ["success", "healthy"]:
                            print(f"  ✅ API Response: {data.get('status', 'unknown')}")
                        else:
                            print(f"  ⚠️  Unexpected API response: {data}")
                    except json.JSONDecodeError:
                        print(f"  ⚠️  Non-JSON response from API endpoint")
                        
            else:
                print(f"❌ FAIL (Status: {response.status_code})")
                print(f"  Error: {response.text[:100]}...")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ FAIL (Connection Error)")
            print(f"  Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        print("🎉 All tests passed! Your Vercel deployment is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the errors above.")
        return False

def test_local_app():
    """Test the local development server."""
    print("🧪 Testing Local Development Server")
    return test_vercel_app("http://localhost:5001")

def main():
    """Main test function."""
    if len(sys.argv) > 1:
        # Test provided URL
        url = sys.argv[1]
        if not url.startswith("http"):
            url = f"https://{url}"
        success = test_vercel_app(url)
    else:
        # Test local server
        success = test_local_app()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
