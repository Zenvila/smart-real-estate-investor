#!/usr/bin/env python3
"""
Real Estate Investment Analyzer - Web Interface
A comprehensive web application for real estate investment analysis and predictions.
"""

from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

# Import our custom agents
from filter_agent import FilterAgent
from prediction_agent import PredictionAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'real_estate_analyzer_secret_key_2024'

# Initialize agents
filter_agent = None
prediction_agent = None

def initialize_agents():
    """Initialize the filter and prediction agents"""
    global filter_agent, prediction_agent
    try:
        filter_agent = FilterAgent()
        prediction_agent = PredictionAgent()
        prediction_agent.load_models()
        logger.info("Agents initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing agents: {e}")

@app.route('/')
def index():
    """Main page with the search interface"""
    return render_template('index.html')

@app.route('/api/available_options')
def get_available_options():
    """Get available options for dropdowns"""
    try:
        if filter_agent is None:
            initialize_agents()
        
        options = filter_agent.get_available_options()
        return jsonify({
            'success': True,
            'data': options
        })
    except Exception as e:
        logger.error(f"Error getting available options: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/search', methods=['POST'])
def search_properties():
    """Search properties based on user criteria"""
    try:
        if filter_agent is None or prediction_agent is None:
            initialize_agents()
        
        # Get form data
        data = request.get_json()
        
        min_budget = float(data.get('min_budget', 0))
        max_budget = float(data.get('max_budget', float('inf')))
        target_city = data.get('target_city', None)
        target_location = data.get('target_location', None)
        property_category = data.get('property_category', None)
        purpose = data.get('purpose', None)
        
        # Apply filters
        filtered_data = filter_agent.apply_filters(
            min_budget=min_budget,
            max_budget=max_budget,
            target_city=target_city,
            target_location=target_location,
            property_category=property_category,
            purpose=purpose
        )
        
        if filtered_data.empty:
            return jsonify({
                'success': True,
                'message': 'No properties found matching your criteria',
                'data': [],
                'summary': {
                    'total_properties': 0,
                    'avg_price': 0,
                    'price_range': {'min': 0, 'max': 0}
                }
            })
        
        # Make predictions
        predictions = prediction_agent.make_comprehensive_predictions(filtered_data)
        
        # Get top investments
        top_investments = prediction_agent.get_top_investments(predictions, top_n=10)
        
        # Get comprehensive summary
        summary = prediction_agent.get_comprehensive_summary(predictions)
        
        # Prepare results for frontend
        results = []
        for _, row in top_investments.iterrows():
            results.append({
                'id': row.get('id', 'N/A'),
                'title': f"{row.get('property_category', 'Property')} in {row.get('location', 'Unknown')}",
                'price': f"PKR {row.get('price', 0):,.0f}",
                'area': f"{row.get('area_marla', 0):.1f} Marla",
                'location': row.get('location', 'Unknown'),
                'city': row.get('city', 'Unknown'),
                'property_category': row.get('property_category', 'Unknown'),
                'purpose': row.get('purpose', 'Unknown'),
                'roi_prediction': f"{row.get('predicted_roi_percent', 0):.1f}%",
                'risk_level': row.get('predicted_risk_level', 'Unknown'),
                'investment_score': f"{row.get('investment_score', 0):.1f}",
                'recommendation': row.get('investment_recommendation', 'Unknown'),

            })
        
        # Calculate price statistics from filtered data
        price_stats = {
            'avg_price': filtered_data['price'].mean(),
            'min_price': filtered_data['price'].min(),
            'max_price': filtered_data['price'].max()
        }
        
        return jsonify({
            'success': True,
            'data': results,
            'summary': {
                'total_properties': len(filtered_data),
                'avg_price': f"PKR {price_stats['avg_price']:,.0f}",
                'price_range': {
                    'min': f"PKR {price_stats['min_price']:,.0f}",
                    'max': f"PKR {price_stats['max_price']:,.0f}"
                },
                'avg_roi': f"{summary['roi_statistics']['mean_roi']:.1f}%",
                'risk_distribution': summary['risk_statistics']['risk_distribution'],
                'roi_distribution': {
                    'min': f"{summary['roi_statistics']['min_roi']:.1f}%",
                    'max': f"{summary['roi_statistics']['max_roi']:.1f}%",
                    'mean': f"{summary['roi_statistics']['mean_roi']:.1f}%"
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error in search_properties: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/property/<property_id>')
def get_property_details(property_id):
    """Get detailed information about a specific property"""
    try:
        if filter_agent is None:
            initialize_agents()
        
        # Get all data and find the specific property
        all_data = filter_agent.data
        property_data = all_data[all_data['id'] == property_id]
        
        if property_data.empty:
            return jsonify({
                'success': False,
                'error': 'Property not found'
            }), 404
        
        property_row = property_data.iloc[0]
        
        # Make predictions for this specific property
        predictions = prediction_agent.make_comprehensive_predictions(property_data)
        prediction_row = predictions.iloc[0]
        
        details = {
            'id': property_row.get('id', 'N/A'),
            'title': f"{property_row.get('property_category', 'Property')} in {property_row.get('location', 'Unknown')}",
            'price': f"PKR {property_row.get('price', 0):,.0f}",
            'area': f"{property_row.get('area_marla', 0):.1f} Marla",
            'location': property_row.get('location', 'Unknown'),
            'city': property_row.get('city', 'Unknown'),
            'property_category': property_row.get('property_category', 'Unknown'),
            'purpose': property_row.get('purpose', 'Unknown'),
            'roi_prediction': f"{prediction_row.get('predicted_roi_percent', 0):.1f}%",
            'risk_level': prediction_row.get('predicted_risk_level', 'Unknown'),
            'investment_score': f"{prediction_row.get('investment_score', 0):.1f}",
            'recommendation': prediction_row.get('investment_recommendation', 'Unknown'),
            'risk_adjusted_roi': f"{prediction_row.get('risk_adjusted_roi', 0):.1f}%",
            'investment_category': prediction_row.get('investment_category', 'Unknown'),

        }
        
        return jsonify({
            'success': True,
            'data': details
        })
        
    except Exception as e:
        logger.error(f"Error getting property details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize agents on startup
    initialize_agents()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 