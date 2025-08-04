#!/usr/bin/env python3
"""
Complete System Test: Demonstrates the full multi-agent real estate investment system.
Tests Filter Agent, ROI Agent, Risk Analysis Agent, and Prediction Agent together.
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

class CompleteSystemTest:
    """
    Complete System Test: Tests the full multi-agent real estate investment system.
    """
    
    def __init__(self):
        """Initialize the complete system test"""
        self.filter_agent = None
        self.roi_agent = None
        self.risk_agent = None
        self.prediction_agent = None
        self.filtered_data = None
        self.roi_results = None
        self.risk_results = None
        self.comprehensive_results = None
    
    def initialize_agents(self):
        """Initialize all agents"""
        logger.info("Initializing all agents...")
        
        # Initialize Filter Agent
        self.filter_agent = FilterAgent()
        
        # Initialize ROI Agent
        self.roi_agent = ROIAgent()
        
        # Initialize Risk Analysis Agent
        self.risk_agent = RiskAnalysisAgent()
        
        # Initialize Prediction Agent
        self.prediction_agent = PredictionAgent()
        
        logger.info("All agents initialized successfully!")
    
    def test_filter_agent(self, test_criteria: Dict = None):
        """Test the Filter Agent with sample criteria"""
        logger.info("Testing Filter Agent...")
        
        if test_criteria is None:
            # Sample test criteria
            test_criteria = {
                "min_budget": 5000000,  # 5M PKR
                "max_budget": 25000000,  # 25M PKR
                "target_city": "Lahore",
                "target_location": None,
                "property_category": "Homes",
                "purpose": "For Sale"
            }
        
        # Apply filters
        self.filtered_data = self.filter_agent.apply_filters(**test_criteria)
        
        logger.info(f"Filter Agent test completed. Found {len(self.filtered_data)} properties.")
        return self.filtered_data
    
    def test_roi_agent(self):
        """Test the ROI Agent"""
        logger.info("Testing ROI Agent...")
        
        if self.filtered_data is None or len(self.filtered_data) == 0:
            logger.error("No filtered data available for ROI analysis")
            return None
        
        # Make ROI predictions
        self.roi_results = self.roi_agent.predict_roi(self.filtered_data)
        
        # Get ROI summary
        roi_summary = self.roi_agent.get_roi_summary(self.roi_results)
        
        logger.info(f"ROI Agent test completed. Average ROI: {roi_summary['roi_statistics']['mean_roi']:.2f}%")
        return self.roi_results
    
    def test_risk_agent(self):
        """Test the Risk Analysis Agent"""
        logger.info("Testing Risk Analysis Agent...")
        
        if self.filtered_data is None or len(self.filtered_data) == 0:
            logger.error("No filtered data available for risk analysis")
            return None
        
        # Make risk predictions
        self.risk_results = self.risk_agent.predict_risks(self.filtered_data)
        
        # Get risk summary
        risk_summary = self.risk_agent.get_risk_summary(self.risk_results)
        
        logger.info(f"Risk Analysis Agent test completed. Average Risk Level: {risk_summary['risk_statistics']['mean_risk_level']:.2f}")
        return self.risk_results
    
    def test_prediction_agent(self):
        """Test the Prediction Agent (comprehensive analysis)"""
        logger.info("Testing Prediction Agent...")
        
        if self.filtered_data is None or len(self.filtered_data) == 0:
            logger.error("No filtered data available for comprehensive analysis")
            return None
        
        # Make comprehensive predictions
        self.comprehensive_results = self.prediction_agent.make_comprehensive_predictions(self.filtered_data)
        
        # Get comprehensive summary
        comprehensive_summary = self.prediction_agent.get_comprehensive_summary(self.comprehensive_results)
        
        logger.info(f"Prediction Agent test completed. Average Investment Score: {comprehensive_summary['investment_statistics']['mean_investment_score']:.2f}")
        return self.comprehensive_results
    
    def run_complete_test(self, test_criteria: Dict = None):
        """Run complete system test"""
        print("🚀 Starting Complete Multi-Agent System Test...")
        print("="*70)
        
        try:
            # Initialize agents
            self.initialize_agents()
            
            # Test Filter Agent
            print("\n📋 Step 1: Testing Filter Agent...")
            filtered_data = self.test_filter_agent(test_criteria)
            
            if filtered_data is None or len(filtered_data) == 0:
                print("❌ No properties found matching criteria. Test stopped.")
                return
            
            # Test ROI Agent
            print("\n💰 Step 2: Testing ROI Agent...")
            roi_results = self.test_roi_agent()
            
            # Test Risk Analysis Agent
            print("\n⚠️ Step 3: Testing Risk Analysis Agent...")
            risk_results = self.test_risk_agent()
            
            # Test Prediction Agent
            print("\n🔮 Step 4: Testing Prediction Agent...")
            comprehensive_results = self.test_prediction_agent()
            
            # Generate comprehensive report
            print("\n📊 Step 5: Generating Comprehensive Report...")
            self.generate_comprehensive_report()
            
            print("\n" + "="*70)
            print("✅ Complete System Test Finished Successfully!")
            print("="*70)
            
        except Exception as e:
            print(f"❌ Error during complete system test: {e}")
            logger.error(f"Complete system test failed: {e}")
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive report of all results"""
        if self.comprehensive_results is None:
            print("❌ No comprehensive results available for report generation")
            return
        
        print("\n" + "="*70)
        print("📊 COMPREHENSIVE INVESTMENT ANALYSIS REPORT")
        print("="*70)
        
        # Get comprehensive summary
        summary = self.prediction_agent.get_comprehensive_summary(self.comprehensive_results)
        
        # Display executive summary
        print(f"\n📈 EXECUTIVE SUMMARY:")
        print(f"   Total Properties Analyzed: {summary['total_properties']:,}")
        print(f"   Average ROI: {summary['roi_statistics']['mean_roi']:.2f}%")
        print(f"   Average Risk Level: {summary['risk_statistics']['mean_risk_level']:.2f}/5")
        print(f"   Average Investment Score: {summary['investment_statistics']['mean_investment_score']:.2f}/10")
        
        # Display ROI statistics
        print(f"\n💰 ROI ANALYSIS:")
        print(f"   Highest ROI: {summary['roi_statistics']['max_roi']:.2f}%")
        print(f"   Lowest ROI: {summary['roi_statistics']['min_roi']:.2f}%")
        print(f"   Median ROI: {summary['roi_statistics']['median_roi']:.2f}%")
        
        # Display risk statistics
        print(f"\n⚠️ RISK ANALYSIS:")
        risk_dist = summary['risk_statistics']['risk_distribution']
        for risk_level, count in risk_dist.items():
            percentage = (count / summary['total_properties']) * 100
            print(f"   Risk Level {risk_level}: {count} properties ({percentage:.1f}%)")
        
        # Display investment recommendations
        print(f"\n🎯 INVESTMENT RECOMMENDATIONS:")
        recs = summary['recommendations']
        print(f"   Strong Buy: {recs['strong_buy_count']} properties")
        print(f"   Buy: {recs['buy_count']} properties")
        print(f"   Consider: {recs['consider_count']} properties")
        print(f"   Avoid: {recs['avoid_count']} properties")
        
        # Display top investments
        print(f"\n🏆 TOP INVESTMENT OPPORTUNITIES:")
        top_investments = self.prediction_agent.get_top_investments(self.comprehensive_results, top_n=5)
        
        for i, (_, investment) in enumerate(top_investments.iterrows(), 1):
            print(f"   {i}. {investment['title'][:50]}...")
            print(f"      Location: {investment['city']}, {investment['location']}")
            print(f"      Price: {investment['price']:,.0f} PKR")
            print(f"      ROI: {investment['predicted_roi_percent']:.2f}%")
            print(f"      Risk: {investment['predicted_risk_level']}/5")
            print(f"      Investment Score: {investment['investment_score']:.2f}/10")
            print()
        
        # Save results
        print(f"\n💾 SAVING RESULTS...")
        self.prediction_agent.save_comprehensive_results(self.comprehensive_results, "complete_analysis_results.csv")
        print(f"   Results saved to: complete_analysis_results.csv")
        
        print("\n" + "="*70)
        print("📋 REPORT GENERATION COMPLETED!")
        print("="*70)
    
    def test_different_scenarios(self):
        """Test different investment scenarios"""
        scenarios = [
            {
                "name": "Budget Homes in Lahore",
                "criteria": {
                    "min_budget": 3000000,
                    "max_budget": 15000000,
                    "target_city": "Lahore",
                    "property_category": "Homes",
                    "purpose": "For Sale"
                }
            },
            {
                "name": "Premium Commercial in Islamabad",
                "criteria": {
                    "min_budget": 50000000,
                    "max_budget": 200000000,
                    "target_city": "Islamabad",
                    "property_category": "Commercial",
                    "purpose": "For Sale"
                }
            },
            {
                "name": "Affordable Plots in Karachi",
                "criteria": {
                    "min_budget": 1000000,
                    "max_budget": 10000000,
                    "target_city": "Karachi",
                    "property_category": "Plots",
                    "purpose": "For Sale"
                }
            }
        ]
        
        print("🧪 Testing Different Investment Scenarios...")
        print("="*70)
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n📋 Scenario {i}: {scenario['name']}")
            print("-" * 50)
            
            try:
                # Test this scenario
                filtered_data = self.test_filter_agent(scenario['criteria'])
                
                if filtered_data is not None and len(filtered_data) > 0:
                    # Get comprehensive results
                    comprehensive_results = self.prediction_agent.make_comprehensive_predictions(filtered_data)
                    summary = self.prediction_agent.get_comprehensive_summary(comprehensive_results)
                    
                    print(f"   Properties Found: {len(filtered_data):,}")
                    print(f"   Average ROI: {summary['roi_statistics']['mean_roi']:.2f}%")
                    print(f"   Average Risk: {summary['risk_statistics']['mean_risk_level']:.2f}/5")
                    print(f"   Investment Score: {summary['investment_statistics']['mean_investment_score']:.2f}/10")
                    
                    # Get top investment
                    top_investment = self.prediction_agent.get_top_investments(comprehensive_results, top_n=1)
                    if len(top_investment) > 0:
                        best = top_investment.iloc[0]
                        print(f"   Best Investment: {best['title'][:40]}...")
                        print(f"   Best ROI: {best['predicted_roi_percent']:.2f}%")
                else:
                    print("   No properties found matching criteria")
                    
            except Exception as e:
                print(f"   Error testing scenario: {e}")
        
        print("\n" + "="*70)
        print("✅ Scenario Testing Completed!")
        print("="*70)

# Main execution
if __name__ == "__main__":
    try:
        # Create test instance
        test_system = CompleteSystemTest()
        
        # Run complete test
        test_system.run_complete_test()
        
        # Test different scenarios
        test_system.test_different_scenarios()
        
    except Exception as e:
        print(f"❌ Error in complete system test: {e}")
        print("Please ensure all models are trained first using ml_model_trainer.py") 