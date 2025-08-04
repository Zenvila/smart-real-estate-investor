#!/usr/bin/env python3
"""
User Filter Test: Tests the complete system with user-selected filters.
Simulates real user input and provides comprehensive investment analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# Import our agents
from filter_agent import FilterAgent
from roi_agent import ROIAgent
from risk_analysis_agent import RiskAnalysisAgent
from prediction_agent import PredictionAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_user_filters():
    """
    Get filter criteria from user input using FilterAgent.
    
    Returns:
        Dict: User filter criteria
    """
    from filter_agent import get_user_input
    return get_user_input()

def run_user_analysis(user_filters: Dict):
    """
    Run complete analysis with user-selected filters.
    
    Args:
        user_filters (Dict): User filter criteria
    """
    print("\n🚀 Starting Analysis with Your Criteria...")
    print("="*60)
    
    try:
        # Initialize all agents
        print("📋 Initializing agents...")
        filter_agent = FilterAgent()
        roi_agent = ROIAgent()
        risk_agent = RiskAnalysisAgent()
        prediction_agent = PredictionAgent()
        
        # Step 1: Apply user filters
        print(f"\n🔍 Step 1: Filtering properties...")
        print(f"   Budget: {user_filters['min_budget']:,.0f} - {user_filters['max_budget']:,.0f} PKR")
        print(f"   City: {user_filters['target_city'] or 'Any'}")
        print(f"   Location: {user_filters['target_location'] or 'Any'}")
        print(f"   Property Category: {user_filters['property_category'] or 'Any'}")
        print(f"   Purpose: {user_filters['purpose'] or 'Any'}")
        
        filtered_data = filter_agent.apply_filters(**user_filters)
        
        if filtered_data is None or len(filtered_data) == 0:
            print("❌ No properties found matching your criteria.")
            print("💡 Try adjusting your filters (broader budget range, different location, etc.)")
            return
        
        print(f"✅ Found {len(filtered_data):,} properties matching your criteria!")
        
        # Step 2: ROI Analysis
        print(f"\n💰 Step 2: ROI Analysis...")
        roi_results = roi_agent.predict_roi(filtered_data)
        roi_summary = roi_agent.get_roi_summary(roi_results)
        
        print(f"   Average ROI: {roi_summary['roi_statistics']['mean_roi']:.2f}%")
        print(f"   Highest ROI: {roi_summary['roi_statistics']['max_roi']:.2f}%")
        print(f"   Estimated Annual Rent: {roi_summary['rental_estimates']['mean_annual_rent']:,.0f} PKR")
        
        # Step 3: Risk Analysis
        print(f"\n⚠️ Step 3: Risk Analysis...")
        risk_results = risk_agent.predict_risks(filtered_data)
        risk_summary = risk_agent.get_risk_summary(risk_results)
        
        print(f"   Average Risk Level: {risk_summary['risk_statistics']['mean_risk_level']:.2f}/5")
        print(f"   Safe Investments: {risk_summary['risk_insights']['low_risk_count']} properties")
        print(f"   High Risk Investments: {risk_summary['risk_insights']['high_risk_count']} properties")
        
        # Step 4: Comprehensive Analysis
        print(f"\n🔮 Step 4: Comprehensive Investment Analysis...")
        comprehensive_results = prediction_agent.make_comprehensive_predictions(filtered_data)
        comprehensive_summary = prediction_agent.get_comprehensive_summary(comprehensive_results)
        
        print(f"   Average Investment Score: {comprehensive_summary['investment_statistics']['mean_investment_score']:.2f}/10")
        print(f"   Strong Buy Recommendations: {comprehensive_summary['recommendations']['strong_buy_count']} properties")
        print(f"   Buy Recommendations: {comprehensive_summary['recommendations']['buy_count']} properties")
        
        # Step 5: Generate Results
        print(f"\n📊 Step 5: Generating Your Investment Report...")
        generate_user_report(comprehensive_results, user_filters)
        
        # Save results
        output_file = f"user_analysis_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
        prediction_agent.save_comprehensive_results(comprehensive_results, output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
        print("\n" + "="*60)
        print("✅ Analysis Completed Successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Please ensure models are trained first using ml_model_trainer.py")

def generate_user_report(results: pd.DataFrame, user_filters: Dict):
    """
    Generate a personalized report for the user.
    
    Args:
        results (pd.DataFrame): Comprehensive analysis results
        user_filters (Dict): User's original filter criteria
    """
    print("\n" + "="*60)
    print("📋 YOUR PERSONALIZED INVESTMENT REPORT")
    print("="*60)
    
    # Get comprehensive summary
    from prediction_agent import PredictionAgent
    prediction_agent = PredictionAgent()
    summary = prediction_agent.get_comprehensive_summary(results)
    
    # Display user criteria
    print(f"\n🎯 YOUR INVESTMENT CRITERIA:")
    print(f"   Budget Range: {user_filters['min_budget']:,.0f} - {user_filters['max_budget']:,.0f} PKR")
    print(f"   Target City: {user_filters['target_city'] or 'Any'}")
    print(f"   Property Type: {user_filters['property_category'] or 'Any'}")
    print(f"   Purpose: {user_filters['purpose'] or 'Any'}")
    
    # Display executive summary
    print(f"\n📈 EXECUTIVE SUMMARY:")
    print(f"   Properties Found: {summary['total_properties']:,}")
    print(f"   Average ROI: {summary['roi_statistics']['mean_roi']:.2f}%")
    print(f"   Average Risk Level: {summary['risk_statistics']['mean_risk_level']:.2f}/5")
    print(f"   Average Investment Score: {summary['investment_statistics']['mean_investment_score']:.2f}/10")
    
    # Display top 5 investment opportunities
    print(f"\n🏆 TOP 5 INVESTMENT OPPORTUNITIES:")
    top_investments = prediction_agent.get_top_investments(results, top_n=5)
    
    for i, (_, investment) in enumerate(top_investments.iterrows(), 1):
        print(f"\n   {i}. {investment['title'][:60]}...")
        print(f"      📍 Location: {investment['city']}, {investment['location']}")
        print(f"      💰 Price: {investment['price']:,.0f} PKR")
        print(f"      📏 Area: {investment['area_marla']:.1f} Marla")
        print(f"      🏘️ Type: {investment['type']}")
        print(f"      💸 ROI: {investment['predicted_roi_percent']:.2f}%")
        print(f"      ⚠️ Risk: {investment['predicted_risk_level']}/5")
        print(f"      ⭐ Investment Score: {investment['investment_score']:.2f}/10")
        print(f"      🎯 Recommendation: {investment['investment_recommendation']}")
    
    # Display investment breakdown
    print(f"\n📊 INVESTMENT BREAKDOWN:")
    investment_cats = summary['investment_statistics']['investment_categories']
    for category, count in investment_cats.items():
        percentage = (count / summary['total_properties']) * 100
        print(f"   {category}: {count} properties ({percentage:.1f}%)")
    
    # Display recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    recs = summary['recommendations']
    print(f"   Strong Buy: {recs['strong_buy_count']} properties")
    print(f"   Buy: {recs['buy_count']} properties")
    print(f"   Consider: {recs['consider_count']} properties")
    print(f"   Avoid: {recs['avoid_count']} properties")
    
    # Display risk-adjusted ROI
    print(f"\n💰 RISK-ADJUSTED ROI ANALYSIS:")
    risk_adjusted_roi = results['risk_adjusted_roi'].describe()
    print(f"   Average Risk-Adjusted ROI: {risk_adjusted_roi['mean']:.2f}%")
    print(f"   Best Risk-Adjusted ROI: {risk_adjusted_roi['max']:.2f}%")
    print(f"   Worst Risk-Adjusted ROI: {risk_adjusted_roi['min']:.2f}%")
    
    print("\n" + "="*60)
    print("📋 REPORT GENERATION COMPLETED!")
    print("="*60)

def main():
    """Main function to run user filter test"""
    try:
        print("🚀 Welcome to the Real Estate Investment Analysis System!")
        print("This system will analyze properties based on your criteria.")
        
        # Get user filters
        user_filters = get_user_filters()
        
        # Run analysis
        run_user_analysis(user_filters)
        
        # Ask if user wants to try different filters
        while True:
            try_again = input("\n🤔 Would you like to try different filters? (y/n): ").strip().lower()
            if try_again in ['y', 'yes']:
                user_filters = get_user_filters()
                run_user_analysis(user_filters)
            elif try_again in ['n', 'no']:
                print("👋 Thank you for using the Real Estate Investment Analysis System!")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        
    except KeyboardInterrupt:
        print("\n👋 Analysis cancelled by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure models are trained first using ml_model_trainer.py")

if __name__ == "__main__":
    main() 