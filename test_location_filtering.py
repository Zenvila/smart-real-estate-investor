#!/usr/bin/env python3
"""
Test script to verify location filtering functionality
"""

import pandas as pd
from filter_agent import FilterAgent

def test_location_filtering():
    """Test the location filtering functionality"""
    
    print("🧪 Testing Location Filtering...")
    
    # Initialize filter agent
    filter_agent = FilterAgent()
    
    # Test 1: Get available options
    print("\n1. Testing available options...")
    options = filter_agent.get_available_options()
    
    print(f"Available cities: {options.get('cities', [])}")
    print(f"Total locations: {len(options.get('locations', []))}")
    
    # Test 2: Test Islamabad filtering
    print("\n2. Testing Islamabad filtering...")
    islamabad_data = filter_agent.apply_filters(
        target_city="Islamabad",
        min_budget=0,
        max_budget=float('inf')
    )
    print(f"Total Islamabad properties: {len(islamabad_data)}")
    
    # Test 3: Test specific location filtering
    print("\n3. Testing specific location filtering...")
    
    # Get some sample locations from Islamabad
    islamabad_locations = islamabad_data['location'].unique()[:5]
    print(f"Sample Islamabad locations: {islamabad_locations}")
    
    for location in islamabad_locations:
        filtered_data = filter_agent.apply_filters(
            target_city="Islamabad",
            target_location=location,
            min_budget=0,
            max_budget=float('inf')
        )
        print(f"Location '{location}': {len(filtered_data)} properties")
        
        if len(filtered_data) > 0:
            print(f"  Sample properties:")
            for idx, row in filtered_data.head(2).iterrows():
                print(f"    - {row['title'][:50]}... | {row['location']}")
    
    # Test 4: Test Blue Area specifically
    print("\n4. Testing Blue Area filtering...")
    blue_area_data = filter_agent.apply_filters(
        target_city="Islamabad",
        target_location="Blue Area",
        min_budget=0,
        max_budget=float('inf')
    )
    print(f"Blue Area properties: {len(blue_area_data)}")
    
    if len(blue_area_data) > 0:
        print("Sample Blue Area properties:")
        for idx, row in blue_area_data.head(3).iterrows():
            print(f"  - {row['title'][:50]}... | {row['location']}")
    
    # Test 5: Test DHA filtering
    print("\n5. Testing DHA filtering...")
    dha_data = filter_agent.apply_filters(
        target_city="Islamabad",
        target_location="DHA",
        min_budget=0,
        max_budget=float('inf')
    )
    print(f"DHA properties: {len(dha_data)}")
    
    if len(dha_data) > 0:
        print("Sample DHA properties:")
        for idx, row in dha_data.head(3).iterrows():
            print(f"  - {row['title'][:50]}... | {row['location']}")
    
    print("\n✅ Location filtering test completed!")

if __name__ == "__main__":
    test_location_filtering() 