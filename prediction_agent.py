#!/usr/bin/env python3
"""
Prediction Agent: Comprehensive investment prediction system.
Combines ROI and Risk predictions to provide complete investment analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionAgent:
    """
    Prediction Agent: Comprehensive investment prediction system.
    Combines ROI and Risk predictions to provide complete investment analysis.
    """
    
    def __init__(self, model_dir: str = "ml_models"):
        """
        Initialize the Prediction Agent.
        
        Args:
            model_dir (str): Directory containing trained models
        """
        self.model_dir = model_dir
        self.roi_regressor = None
        self.risk_classifier = None
        self.price_regressor = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.models_loaded = False
        
    def load_models(self):
        """Load all pre-trained models and preprocessing objects"""
        try:
            logger.info(f"Loading all prediction models from {self.model_dir}/")
            
            self.roi_regressor = joblib.load(f"{self.model_dir}/roi_regressor.pkl")
            self.risk_classifier = joblib.load(f"{self.model_dir}/risk_classifier.pkl")
            self.price_regressor = joblib.load(f"{self.model_dir}/price_regressor.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.label_encoders = joblib.load(f"{self.model_dir}/label_encoders.pkl")
            self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
            
            self.models_loaded = True
            logger.info("All prediction models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading prediction models: {e}")
            logger.error("Please run ml_model_trainer.py first to train the models.")
            raise
    
    def prepare_data_for_predictions(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare filtered data for all predictions using the same preprocessing as training.
        
        Args:
            filtered_data (pd.DataFrame): Filtered data from Filter Agent
            
        Returns:
            pd.DataFrame: Prepared features for predictions
        """
        if not self.models_loaded:
            self.load_models()
        
        logger.info("Preparing data for comprehensive predictions...")
        
        # Create a copy for feature engineering
        features = filtered_data.copy()
        
        # Apply the same feature engineering as training
        features = self._engineer_features(features)
        features = self._handle_missing_values(features)
        features = self._encode_categorical_features(features)
        features = self._select_and_scale_features(features)
        
        logger.info(f"Prediction data preparation completed. Shape: {features.shape}")
        return features
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for predictions (same as training)"""
        features = data.copy()
        
        # Price features - handle division by zero properly
        # Replace 0 values with NaN to avoid division by zero
        area_marla_safe = features['area_marla'].replace(0, np.nan)
        features['price_per_marla'] = features['price'] / area_marla_safe
        features['price_per_sqft'] = features['price'] / (area_marla_safe * 272.25)
        
        # Area features
        try:
            features['area_category'] = pd.cut(features['area_marla'],
                                             bins=[0, 5, 10, 25, float('inf')],
                                             labels=['Small', 'Medium', 'Large', 'Extra Large'],
                                             include_lowest=True)
        except:
            # Fallback if categorical creation fails
            features['area_category'] = 'Medium'
        
        # Add log_area feature
        features['log_area'] = np.log(features['area_marla'].replace(0, 1))
        
        # Location features
        features['city_main'] = features['city'].str.split('_').str[0]
        features['city_category'] = features['city'].str.split('_').str[1]
        
        # Property type features
        features['property_category'] = features['type'].map({
            'plot': 'Land',
            'house': 'Residential',
            'flat': 'Residential',
            'shop': 'Commercial',
            'office': 'Commercial',
            'building': 'Commercial',
            'factory': 'Industrial',
            'other': 'Other'
        })
        
        # Binary features
        features['has_bedrooms'] = features['bedrooms'].notna() & (features['bedrooms'] != '-')
        features['has_bathrooms'] = features['bathrooms'].notna() & (features['bathrooms'] != '-')
        features['has_description'] = features['description'].str.len() > 10
        features['has_area'] = features['area_marla'].notna() & (features['area_marla'] > 0)
        
        # Investment features
        features['price_to_area_ratio'] = features['price'] / area_marla_safe
        features['is_affordable'] = features['price'] <= features['price'].quantile(0.5)
        features['is_premium'] = features['price'] >= features['price'].quantile(0.8)
        
        return features
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numeric columns with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                median_val = data[col].median()
                if pd.isna(median_val):
                    median_val = 0
                data[col].fillna(median_val, inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                mode_vals = data[col].mode()
                if len(mode_vals) > 0:
                    data[col].fillna(mode_vals[0], inplace=True)
                else:
                    data[col].fillna('Unknown', inplace=True)
        
        # Handle categorical columns that might have issues
        for col in data.columns:
            if data[col].dtype.name == 'category':
                # Convert categorical to string to avoid issues
                data[col] = data[col].astype(str)
        
        # Handle any remaining NaN values
        data = data.fillna(0)
        
        return data
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using pre-trained encoders"""
        categorical_cols = ['city_main', 'city_category', 'property_category', 'type', 'purpose', 'area_category']
        
        for col in categorical_cols:
            if col in data.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Handle unseen categories by using a default value
                data[f'{col}_encoded'] = data[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
        
        return data
    
    def _select_and_scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select and scale features for predictions"""
        # Filter to only available features that were used in training
        available_features = [col for col in self.feature_columns if col in data.columns]
        
        # Create feature matrix
        X = data[available_features].copy()
        
        # Handle infinity and extreme values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for each column
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Scale features using pre-trained scaler
        X_scaled = self.scaler.transform(X)
        
        # Create final feature dataframe
        features_df = pd.DataFrame(X_scaled, columns=available_features, index=data.index)
        
        return features_df
    
    def make_comprehensive_predictions(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Make comprehensive predictions including ROI, Risk, and Price predictions.
        
        Args:
            filtered_data (pd.DataFrame): Filtered data from Filter Agent
            
        Returns:
            pd.DataFrame: Original data with all predictions
        """
        if not self.models_loaded:
            self.load_models()
        
        logger.info("Making comprehensive predictions...")
        
        # Prepare features
        features = self.prepare_data_for_predictions(filtered_data)
        
        # Make all predictions
        roi_predictions = self.roi_regressor.predict(features)
        risk_predictions = self.risk_classifier.predict(features)
        risk_probabilities = self.risk_classifier.predict_proba(features)
        price_predictions = self.price_regressor.predict(features)
        
        # Add predictions to original data
        results = filtered_data.copy()
        
        # ROI predictions
        results['predicted_roi_percent'] = roi_predictions
        results['predicted_roi_decimal'] = roi_predictions / 100
        results['estimated_annual_rent'] = results['price'] * results['predicted_roi_decimal']
        results['monthly_rent_estimate'] = results['estimated_annual_rent'] / 12
        
        # Risk predictions
        results['predicted_risk_level'] = risk_predictions
        results['risk_confidence'] = np.max(risk_probabilities, axis=1)
        results['risk_description'] = results['predicted_risk_level'].map({
            1: 'Very Low Risk',
            2: 'Low Risk', 
            3: 'Medium Risk',
            4: 'High Risk',
            5: 'Very High Risk'
        })
        
        # Price predictions
        results['predicted_price'] = np.exp(price_predictions) - 1
        results['price_prediction_error'] = np.abs(results['price'] - results['predicted_price'])
        results['price_prediction_accuracy'] = 1 - (results['price_prediction_error'] / results['price'])
        
        # Calculate combined investment metrics
        results['investment_score'] = self._calculate_investment_score(results)
        results['risk_adjusted_roi'] = self._calculate_risk_adjusted_roi(results)
        results['investment_recommendation'] = self._get_comprehensive_recommendation(results)
        results['investment_category'] = self._categorize_investment(results)
        
        # Calculate profit potential
        results['annual_profit'] = results['estimated_annual_rent']
        results['monthly_profit'] = results['monthly_rent_estimate']
        
        # Monthly profit projections (next 12 months)
        results['profit_month_1'] = results['monthly_profit']
        results['profit_month_2'] = results['monthly_profit']
        results['profit_month_3'] = results['monthly_profit']
        results['profit_month_6'] = results['monthly_profit'] * 6
        results['profit_month_12'] = results['annual_profit']
        
        # Yearly profit projections with maintenance costs
        results['profit_after_1_year'] = results['annual_profit'] - (results['price'] * 0.02)  # 2% maintenance cost
        results['profit_after_2_years'] = (results['annual_profit'] * 2) - (results['price'] * 0.04)  # 4% maintenance cost
        results['profit_after_3_years'] = (results['annual_profit'] * 3) - (results['price'] * 0.06)  # 6% maintenance cost
        results['profit_after_5_years'] = (results['annual_profit'] * 5) - (results['price'] * 0.1)  # 10% total maintenance
        results['profit_after_10_years'] = (results['annual_profit'] * 10) - (results['price'] * 0.2)  # 20% total maintenance
        
        # Calculate ROI percentages
        results['roi_percentage'] = (results['annual_profit'] / results['price']) * 100
        results['roi_after_1_year'] = (results['profit_after_1_year'] / results['price']) * 100
        results['roi_after_2_years'] = (results['profit_after_2_years'] / results['price']) * 100
        results['roi_after_3_years'] = (results['profit_after_3_years'] / results['price']) * 100
        results['roi_after_5_years'] = (results['profit_after_5_years'] / results['price']) * 100
        results['roi_after_10_years'] = (results['profit_after_10_years'] / results['price']) * 100
        
        logger.info(f"Comprehensive predictions completed for {len(results)} properties")
        return results
    
    def _calculate_investment_score(self, results: pd.DataFrame) -> pd.Series:
        """Calculate overall investment score combining ROI and Risk"""
        # ROI component (0-10 scale)
        roi_score = (results['predicted_roi_percent'] - 3) / (20 - 3) * 10  # Normalize 3-20% to 0-10
        
        # Risk component (inverse of risk level, 0-10 scale)
        risk_score = (6 - results['predicted_risk_level']) * 2  # 1-5 risk to 10-2 score
        
        # Confidence adjustment
        confidence_adjustment = results['risk_confidence'] * 2
        
        # Calculate weighted investment score
        investment_score = (roi_score * 0.4 + risk_score * 0.4 + confidence_adjustment * 0.2)
        
        return investment_score.clip(1, 10)
    
    def _calculate_risk_adjusted_roi(self, results: pd.DataFrame) -> pd.Series:
        """Calculate risk-adjusted ROI"""
        # Base ROI
        base_roi = results['predicted_roi_percent']
        
        # Risk penalty (higher risk = lower adjusted ROI)
        risk_penalty = (results['predicted_risk_level'] - 1) * 0.5
        
        # Calculate risk-adjusted ROI
        risk_adjusted_roi = base_roi - risk_penalty
        
        return risk_adjusted_roi.clip(0, 20)
    
    def _get_comprehensive_recommendation(self, results: pd.DataFrame) -> pd.Series:
        """Generate comprehensive investment recommendations"""
        recommendations = []
        
        for _, row in results.iterrows():
            roi = row['predicted_roi_percent']
            risk_level = row['predicted_risk_level']
            investment_score = row['investment_score']
            price = row['price']
            
            # Base recommendation on investment score
            if investment_score >= 8:
                base_rec = "Strong Buy"
            elif investment_score >= 6:
                base_rec = "Buy"
            elif investment_score >= 4:
                base_rec = "Consider"
            elif investment_score >= 2:
                base_rec = "Hold"
            else:
                base_rec = "Avoid"
            
            # Add ROI context
            if roi >= 12:
                roi_context = " (High ROI)"
            elif roi >= 8:
                roi_context = " (Good ROI)"
            elif roi >= 5:
                roi_context = " (Moderate ROI)"
            else:
                roi_context = " (Low ROI)"
            
            # Add risk context
            if risk_level <= 2:
                risk_context = " (Low Risk)"
            elif risk_level == 3:
                risk_context = " (Moderate Risk)"
            else:
                risk_context = " (High Risk)"
            
            # Add investor type context
            if price <= 10000000:
                investor_context = " - Small Investor Friendly"
            elif price >= 50000000:
                investor_context = " - Large Investor Suitable"
            else:
                investor_context = " - Medium Investor Suitable"
            
            recommendations.append(base_rec + roi_context + risk_context + investor_context)
        
        return pd.Series(recommendations, index=results.index)
    
    def _categorize_investment(self, results: pd.DataFrame) -> pd.Series:
        """Categorize investments based on ROI and Risk"""
        categories = []
        
        for _, row in results.iterrows():
            roi = row['predicted_roi_percent']
            risk_level = row['predicted_risk_level']
            
            if roi >= 10 and risk_level <= 3:
                category = "Premium Investment"
            elif roi >= 8 and risk_level <= 3:
                category = "Good Investment"
            elif roi >= 6 and risk_level <= 4:
                category = "Moderate Investment"
            elif roi >= 4:
                category = "Speculative Investment"
            else:
                category = "High Risk Investment"
            
            categories.append(category)
        
        return pd.Series(categories, index=results.index)
    
    def get_comprehensive_summary(self, results: pd.DataFrame) -> Dict:
        """
        Get a comprehensive summary of all predictions.
        
        Args:
            results (pd.DataFrame): Results with all predictions
            
        Returns:
            Dict: Comprehensive summary statistics
        """
        summary = {
            "total_properties": len(results),
            "roi_statistics": {
                "mean_roi": results['predicted_roi_percent'].mean(),
                "median_roi": results['predicted_roi_percent'].median(),
                "min_roi": results['predicted_roi_percent'].min(),
                "max_roi": results['predicted_roi_percent'].max()
            },
            "risk_statistics": {
                "mean_risk_level": results['predicted_risk_level'].mean(),
                "risk_distribution": results['predicted_risk_level'].value_counts().sort_index().to_dict(),
                "mean_risk_confidence": results['risk_confidence'].mean()
            },
            "investment_statistics": {
                "mean_investment_score": results['investment_score'].mean(),
                "mean_risk_adjusted_roi": results['risk_adjusted_roi'].mean(),
                "investment_categories": results['investment_category'].value_counts().to_dict()
            },
            "top_investments": {
                "highest_roi": results.loc[results['predicted_roi_percent'].idxmax()].to_dict(),
                "lowest_risk": results.loc[results['predicted_risk_level'].idxmin()].to_dict(),
                "best_investment_score": results.loc[results['investment_score'].idxmax()].to_dict(),
                "best_risk_adjusted_roi": results.loc[results['risk_adjusted_roi'].idxmax()].to_dict()
            },
            "recommendations": {
                "strong_buy_count": len(results[results['investment_score'] >= 8]),
                "buy_count": len(results[(results['investment_score'] >= 6) & (results['investment_score'] < 8)]),
                "consider_count": len(results[(results['investment_score'] >= 4) & (results['investment_score'] < 6)]),
                "avoid_count": len(results[results['investment_score'] < 4])
            }
        }
        
        return summary
    
    def get_top_investments(self, results: pd.DataFrame, top_n: int = 10, 
                           sort_by: str = 'investment_score') -> pd.DataFrame:
        """
        Get top investment opportunities.
        
        Args:
            results (pd.DataFrame): Results with all predictions
            top_n (int): Number of top investments to return
            sort_by (str): Column to sort by ('investment_score', 'predicted_roi_percent', 'risk_adjusted_roi')
            
        Returns:
            pd.DataFrame: Top investment opportunities
        """
        # Sort by specified column
        top_investments = results.sort_values(sort_by, ascending=False).head(top_n)
        
        # Select relevant columns for display
        display_columns = [
            'title', 'city', 'location', 'price', 'area_marla', 'type',
            'predicted_roi_percent', 'predicted_risk_level', 'investment_score',
            'risk_adjusted_roi', 'investment_category', 'investment_recommendation'
        ]
        
        return top_investments[display_columns]
    
    def get_investment_report(self, results: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive investment report.
        
        Args:
            results (pd.DataFrame): Results with all predictions
            
        Returns:
            Dict: Comprehensive investment report
        """
        report = {
            "executive_summary": {
                "total_properties_analyzed": len(results),
                "average_roi": results['predicted_roi_percent'].mean(),
                "average_risk_level": results['predicted_risk_level'].mean(),
                "average_investment_score": results['investment_score'].mean(),
                "premium_investments_count": len(results[results['investment_category'] == 'Premium Investment'])
            },
            "investment_breakdown": {
                "by_roi_category": self._analyze_by_roi_category(results),
                "by_risk_category": self._analyze_by_risk_category(results),
                "by_investment_category": results['investment_category'].value_counts().to_dict()
            },
            "top_opportunities": {
                "highest_roi": self.get_top_investments(results, top_n=5, sort_by='predicted_roi_percent').to_dict('records'),
                "lowest_risk": self.get_top_investments(results, top_n=5, sort_by='predicted_risk_level').to_dict('records'),
                "best_overall": self.get_top_investments(results, top_n=5, sort_by='investment_score').to_dict('records')
            },
            "market_insights": {
                "roi_distribution": results['predicted_roi_percent'].describe().to_dict(),
                "risk_distribution": results['predicted_risk_level'].value_counts().sort_index().to_dict(),
                "investment_score_distribution": results['investment_score'].describe().to_dict()
            },
            "recommendations": {
                "for_income_investors": self._get_income_investor_recommendations(results),
                "for_growth_investors": self._get_growth_investor_recommendations(results),
                "for_conservative_investors": self._get_conservative_investor_recommendations(results)
            }
        }
        
        return report
    
    def _analyze_by_roi_category(self, results: pd.DataFrame) -> Dict:
        """Analyze properties by ROI category"""
        roi_categories = {
            "High ROI (10%+)": len(results[results['predicted_roi_percent'] >= 10]),
            "Good ROI (8-10%)": len(results[(results['predicted_roi_percent'] >= 8) & (results['predicted_roi_percent'] < 10)]),
            "Moderate ROI (6-8%)": len(results[(results['predicted_roi_percent'] >= 6) & (results['predicted_roi_percent'] < 8)]),
            "Low ROI (<6%)": len(results[results['predicted_roi_percent'] < 6])
        }
        return roi_categories
    
    def _analyze_by_risk_category(self, results: pd.DataFrame) -> Dict:
        """Analyze properties by risk category"""
        risk_categories = {
            "Low Risk (1-2)": len(results[results['predicted_risk_level'] <= 2]),
            "Moderate Risk (3)": len(results[results['predicted_risk_level'] == 3]),
            "High Risk (4-5)": len(results[results['predicted_risk_level'] >= 4])
        }
        return risk_categories
    
    def _get_income_investor_recommendations(self, results: pd.DataFrame) -> List[Dict]:
        """Get recommendations for income-focused investors"""
        income_focused = results[
            (results['predicted_roi_percent'] >= 8) & 
            (results['predicted_risk_level'] <= 3)
        ].sort_values('predicted_roi_percent', ascending=False).head(3)
        
        return income_focused.to_dict('records')
    
    def _get_growth_investor_recommendations(self, results: pd.DataFrame) -> List[Dict]:
        """Get recommendations for growth-focused investors"""
        growth_focused = results[
            (results['investment_score'] >= 7) & 
            (results['predicted_risk_level'] <= 4)
        ].sort_values('investment_score', ascending=False).head(3)
        
        return growth_focused.to_dict('records')
    
    def _get_conservative_investor_recommendations(self, results: pd.DataFrame) -> List[Dict]:
        """Get recommendations for conservative investors"""
        conservative = results[
            (results['predicted_risk_level'] <= 2) & 
            (results['investment_score'] >= 6)
        ].sort_values('investment_score', ascending=False).head(3)
        
        return conservative.to_dict('records')
    
    def save_comprehensive_results(self, results: pd.DataFrame, output_path: str = "comprehensive_predictions.csv"):
        """
        Save comprehensive prediction results to CSV.
        
        Args:
            results (pd.DataFrame): Results with all predictions
            output_path (str): Path to save the results
        """
        results.to_csv(output_path, index=False)
        logger.info(f"Comprehensive predictions saved to {output_path}")

# Example usage
if __name__ == "__main__":
    try:
        print("🔮 Starting Comprehensive Prediction System...")
        print("="*60)
        
        # Initialize prediction agent
        predictor = PredictionAgent()
        
        # Load models
        predictor.load_models()
        
        print("✅ All prediction models loaded successfully!")
        print("🎯 Ready to make comprehensive investment predictions")
        print("💡 Use this with your Filter Agent to get complete investment analysis")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure models are trained first using ml_model_trainer.py") 