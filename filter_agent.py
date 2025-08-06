#!/usr/bin/env python3
"""
Filter Agent: Handles data loading and filtering based on user criteria.
Filters properties by budget, location, property type, and purpose.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FilterAgent:
    """
    Filter Agent: Handles data loading and filtering based on user criteria.
    """
    
    def __init__(self, data_path: str = "ml_ready_data.csv"):
        """
        Initialize the Filter Agent.
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.data = None
        self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """
        Load the data from CSV file with encoding handling.
        
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            # Try different encodings to handle non-UTF-8 characters
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.data_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    if encoding == encodings[-1]:  # Last encoding tried
                        raise
                    continue
            
            logger.info(f"Loaded {len(self.data)} properties")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def apply_filters(self, 
                     min_budget: float = 0,
                     max_budget: float = float('inf'),
                     target_city: Optional[str] = None,
                     target_location: Optional[str] = None,
                     property_category: Optional[str] = None,
                     purpose: Optional[str] = None) -> pd.DataFrame:
        """
        Apply filters to the data based on user criteria.
        
        Args:
            min_budget (float): Minimum budget in PKR
            max_budget (float): Maximum budget in PKR
            target_city (str): Target city name
            target_location (str): Target location name
            property_category (str): Property category (Commercial, Plots, Homes)
            purpose (str): Property purpose (for_sale, for_rent)
            
        Returns:
            pd.DataFrame: Filtered data
        """
        if self.data is None:
            logger.error("No data loaded")
            return pd.DataFrame()
        
        logger.info("Applying filters to data...")
        filtered = self.data.copy()
        
        # Budget filter
        if min_budget > 0:
            filtered = filtered[filtered['price'] >= min_budget]
            logger.info(f"Budget filter (min): {len(filtered)} properties >= {min_budget:,.0f} PKR")
        
        if max_budget < float('inf'):
            filtered = filtered[filtered['price'] <= max_budget]
            logger.info(f"Budget filter (max): {len(filtered)} properties <= {max_budget:,.0f} PKR")
        
        # City filter
        if target_city:
            # Handle encoded city names (e.g., lahore_homes, lahore_plots)
            city_pattern = target_city.lower()
            filtered = filtered[
                filtered['city'].str.lower().str.contains(city_pattern, na=False)
            ]
            logger.info(f"City filter: {len(filtered)} properties in {target_city}")
        
        # Location filter - IMPROVED LOGIC
        if target_location:
            # Clean location data and handle special characters
            location_lower = target_location.lower().strip()
            
            # More precise location filtering
            if location_lower != "all locations" and location_lower != "all cities - all locations":
                # Check if location contains the target location (case insensitive)
                location_match = filtered['location'].astype(str).str.lower().str.contains(
                    location_lower, case=False, na=False, regex=False
                )
                
                # Also check if the location starts with the target location
                location_start = filtered['location'].astype(str).str.lower().str.startswith(
                    location_lower, na=False
                )
                
                # Combine both conditions
                location_filter = location_match | location_start
                filtered = filtered[location_filter]
                
                logger.info(f"Location filter: {len(filtered)} properties in {target_location}")
        
        # Property category filter
        if property_category:
            property_category_lower = property_category.lower()
            if property_category_lower == 'commercial':
                # Commercial properties: shop, office, building, factory, other
                commercial_types = ['shop', 'office', 'building', 'factory', 'other']
                filtered = filtered[
                    filtered['type'].str.lower().isin(commercial_types)
                ]
                logger.info(f"Commercial property filter: {len(filtered)} commercial properties")
            elif property_category_lower == 'plots':
                # Plot properties
                filtered = filtered[
                    filtered['type'].str.lower() == 'plot'
                ]
                logger.info(f"Plot property filter: {len(filtered)} plot properties")
            elif property_category_lower == 'homes':
                # Home properties: house, flat
                home_types = ['house', 'flat']
                filtered = filtered[
                    filtered['type'].str.lower().isin(home_types)
                ]
                logger.info(f"Home property filter: {len(filtered)} home properties")
            else:
                # Fallback to exact match
                filtered = filtered[
                    filtered['type'].str.contains(property_category, case=False, na=False)
            ]
                logger.info(f"Property category filter: {len(filtered)} properties of category {property_category}")
        
        # Purpose filter
        if purpose:
            purpose_filter = purpose.lower()
            if purpose_filter in ['for sale', 'sale']:
                purpose_filter = 'for_sale'
            elif purpose_filter in ['for rent', 'rent']:
                purpose_filter = 'for_rent'
            
            # Handle the case where we only have for_sale properties
            if purpose_filter == 'for_rent':
                # If user selects rent but we only have sale properties, show a message
                logger.warning(f"Rent properties requested but dataset only contains sale properties")
                # Return empty dataset for rent requests
                filtered = pd.DataFrame()
            else:
                # For sale properties, filter normally
                filtered = filtered[
                    filtered['purpose'].str.lower() == purpose_filter
                ]
            
            logger.info(f"Purpose filter: {len(filtered)} properties for {purpose}")
        
        logger.info(f"Final filtered dataset: {len(filtered)} properties")
        return filtered
    
    def get_available_options(self) -> Dict:
        """
        Get available options for filters.
            
        Returns:
            Dict: Available options for each filter
        """
        if self.data is None:
            return {}
        
        # Clean and filter data
        clean_data = self.data.copy()
        
        # Handle price column - convert to numeric and remove non-numeric values
        clean_data['price'] = pd.to_numeric(clean_data['price'], errors='coerce')
        clean_data = clean_data.dropna(subset=['price'])
        
        # Get unique values, filtering out None, NaN, and empty strings
        cities = clean_data['city'].dropna().unique()
        cities = [city for city in cities if city and str(city).strip()]
        
        # Extract clean city names from encoded values
        clean_cities = []
        for city in cities:
            if '_' in str(city):
                # Extract base city name from encoded values like 'lahore_homes'
                base_city = str(city).split('_')[0].title()
                if base_city not in clean_cities:
                    clean_cities.append(base_city)
            else:
                clean_cities.append(str(city).title())
        
        # Ensure we have the main cities
        main_cities = ['Islamabad', 'Lahore', 'Karachi', 'Rawalpindi']
        for city in main_cities:
            if city not in clean_cities:
                clean_cities.append(city)
        
        locations = clean_data['location'].dropna().unique()
        locations = [loc for loc in locations if loc and str(loc).strip() and str(loc) != 'nan' and str(loc) != 'None']
        
        # Map property types to user-friendly categories
        property_type_mapping = {
            'house': 'Homes',
            'flat': 'Homes', 
            'plot': 'Plots',
            'shop': 'Commercial',
            'office': 'Commercial',
            'building': 'Commercial',
            'factory': 'Commercial',
            'other': 'Commercial'
        }
        
        # Get unique property types and map them
        raw_property_types = clean_data['type'].dropna().unique()
        property_types = []
        for pt in raw_property_types:
            if pt and str(pt).strip():
                mapped_type = property_type_mapping.get(str(pt).lower(), 'Commercial')
                if mapped_type not in property_types:
                    property_types.append(mapped_type)
        
        purposes = clean_data['purpose'].dropna().unique()
        purposes = [p for p in purposes if p and str(p).strip()]
        
        # Group locations by city - IMPROVED LOGIC
        locations_by_city = {}
        for city in clean_cities:
            city_lower = city.lower()
            city_locations = []
            
            # Find locations for this city
            for location in locations:
                location_lower = location.lower()
                location_str = str(location).strip()
                
                # More precise city-location matching
                if (city_lower in location_lower or 
                    location_lower.endswith(f', {city_lower}') or
                    location_lower.startswith(f'{city_lower} ') or
                    f', {city_lower}' in location_lower or
                    # Check for specific area patterns
                    (city_lower == 'islamabad' and any(area in location_lower for area in ['blue area', 'dha', 'f-7', 'f-8', 'g-9', 'g-10', 'g-11'])) or
                    (city_lower == 'lahore' and any(area in location_lower for area in ['dha', 'gulberg', 'defence', 'model town'])) or
                    (city_lower == 'karachi' and any(area in location_lower for area in ['dha', 'clifton', 'defence'])) or
                    (city_lower == 'rawalpindi' and any(area in location_lower for area in ['dha', 'defence', 'saddar']))):
                    city_locations.append(location_str)
            
            # Remove duplicates and sort
            city_locations = sorted(list(set(city_locations)))
            
            # If no specific locations found, add some general ones
            if not city_locations:
                city_locations = [f"All {city} Locations"]
            
            locations_by_city[city] = city_locations
        
        options = {
            'cities': sorted(clean_cities),
            'locations': sorted(locations),
            'locations_by_city': locations_by_city,
            'property_types': sorted(property_types),
            'purposes': sorted(purposes),
            'price_range': {
                'min': int(clean_data['price'].min()) if len(clean_data) > 0 else 0,
                'max': int(clean_data['price'].max()) if len(clean_data) > 0 else 0
            }
        }
        
        return options
    
    def display_available_options(self):
        """Display available options for user reference"""
        options = self.get_available_options()
        
        print("\n" + "="*60)
        print("📋 AVAILABLE FILTER OPTIONS")
        print("="*60)
        
        print(f"\n🏙️ CITIES ({len(options['cities'])}):")
        for city in options['cities'][:10]:  # Show first 10
            print(f"   • {city}")
        if len(options['cities']) > 10:
            print(f"   ... and {len(options['cities']) - 10} more")
        
        print(f"\n📍 LOCATIONS ({len(options['locations'])}):")
        for location in options['locations'][:10]:  # Show first 10
            print(f"   • {location}")
        if len(options['locations']) > 10:
            print(f"   ... and {len(options['locations']) - 10} more")
        
        print(f"\n🏘️ PROPERTY TYPES ({len(options['property_types'])}):")
        for prop_type in options['property_types']:
            print(f"   • {prop_type}")
        
        print(f"\n🎯 PURPOSES ({len(options['purposes'])}):")
        for purpose in options['purposes']:
            print(f"   • {purpose}")
        
        print(f"\n💰 PRICE RANGE:")
        print(f"   • Minimum: {options['price_range']['min']:,.0f} PKR")
        print(f"   • Maximum: {options['price_range']['max']:,.0f} PKR")
        
        print("="*60)

def get_user_input():
    """
    Get user input for filtering criteria.
    
    Returns:
        Dict: User filter criteria
    """
    print("\n" + "="*60)
    print("🏠 REAL ESTATE INVESTMENT ANALYSIS SYSTEM")
    print("="*60)
    print("Please enter your investment criteria:")
    
    # Budget input
    print("\n💰 BUDGET RANGE:")
    while True:
        try:
            min_budget = input("Enter minimum budget (PKR): ").strip()
            if min_budget.lower() == 'skip' or min_budget == '':
                min_budget = 0
            else:
                min_budget = float(min_budget.replace(',', ''))
            break
        except ValueError:
            print("❌ Please enter a valid number or 'skip' for no minimum")
    
    while True:
        try:
            max_budget = input("Enter maximum budget (PKR): ").strip()
            if max_budget.lower() == 'skip' or max_budget == '':
                max_budget = float('inf')
            else:
                max_budget = float(max_budget.replace(',', ''))
            break
        except ValueError:
            print("❌ Please enter a valid number or 'skip' for no maximum")
    
    # Location input
    print("\n📍 LOCATION:")
    target_city = input("Enter city name (Islamabad, Lahore, Karachi, Rawalpindi) or press Enter to skip: ").strip()
    if target_city == '':
        target_city = None
    
    target_location = input("Enter specific location (or press Enter to skip): ").strip()
    if target_location == '':
        target_location = None
    
    # Property type input
    print("\n🏘️ PROPERTY TYPE:")
    print("Available categories: Commercial, Plots, Homes")
    property_category = input("Enter property category (or press Enter to skip): ").strip()
    if property_category == '':
        property_category = None
    
    # Purpose input
    print("\n🎯 PURPOSE:")
    print("Available purposes: For Sale (for_sale)")
    purpose = input("Enter purpose (or press Enter to skip): ").strip()
    if purpose == '':
        purpose = None
    elif purpose.lower() in ['for sale', 'sale']:
        purpose = 'for_sale'
    
    return {
        "min_budget": min_budget,
        "max_budget": max_budget,
        "target_city": target_city,
        "target_location": target_location,
        "property_category": property_category,
        "purpose": purpose
    }

if __name__ == "__main__":
    # Initialize filter agent
    filter_agent = FilterAgent()
    
    # Display available options
    filter_agent.display_available_options()
    
    # Get user input
    user_filters = get_user_input()
    
    # Apply filters
    filtered_data = filter_agent.apply_filters(**user_filters)
    
    print(f"\n✅ Found {len(filtered_data)} properties matching your criteria!")
    
    if len(filtered_data) > 0:
        print("\n📊 Sample of filtered properties:")
        print(filtered_data[['title', 'city', 'location', 'price', 'type']].head()) 