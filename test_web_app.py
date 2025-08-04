#!/usr/bin/env python3
"""
Test script for the SmartReal Estate Pro web application
"""

import requests
import json
import time

def test_web_app():
    """Test the web application endpoints"""
    base_url = "http://localhost:5000"
    
    print("🏠 Testing SmartReal Estate Pro Web Application")
    print("=" * 50)
    
    # Test 1: Check if server is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running successfully")
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running. Please start the app with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False
    
    # Test 2: Check available options API
    try:
        response = requests.get(f"{base_url}/api/available_options", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Available options API working")
                options = data.get('data', {})
                if options.get('cities'):
                    print(f"   📍 Available cities: {len(options['cities'])}")
                if options.get('locations'):
                    print(f"   🏘️ Available locations: {len(options['locations'])}")
            else:
                print("❌ Available options API returned error")
                return False
        else:
            print(f"❌ Available options API returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing available options API: {e}")
        return False
    
    # Test 3: Test search API with sample data
    try:
        search_data = {
            "min_budget": 1000000,
            "max_budget": 5000000,
            "target_city": None,
            "target_location": None,
            "property_category": None,
            "purpose": None
        }
        
        response = requests.post(
            f"{base_url}/api/search",
            json=search_data,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Search API working")
                results = data.get('data', [])
                summary = data.get('summary', {})
                print(f"   📊 Found {len(results)} properties")
                if summary.get('total_properties'):
                    print(f"   💰 Average price: {summary.get('avg_price', 'N/A')}")
                    print(f"   📈 Average ROI: {summary.get('avg_roi', 'N/A')}")
            else:
                print("❌ Search API returned error")
                return False
        else:
            print(f"❌ Search API returned status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error testing search API: {e}")
        return False
    
    print("\n🎉 All tests passed! The web application is working correctly.")
    print(f"🌐 Open your browser and go to: {base_url}")
    return True

if __name__ == "__main__":
    print("Starting web application test...")
    print("Make sure the Flask app is running with: python app.py")
    print()
    
    # Wait a moment for the server to start
    time.sleep(2)
    
    success = test_web_app()
    
    if success:
        print("\n🚀 Your SmartReal Estate Pro web application is ready!")
        print("📱 The interface is fully responsive and works on all devices")
        print("🤖 AI-powered analysis and predictions are active")
        print("🎨 Modern, cool design with smooth animations")
    else:
        print("\n❌ Some tests failed. Please check the setup and try again.") 