#!/usr/bin/env python3
"""
Risk Analysis Agent: Predicts investment risks for real estate properties.
Uses pre-trained ML models to assess risk levels and provide risk insights.
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

class RiskAnalysisAgent:
    """
    Risk Analysis Agent: Predicts investment risks for real estate properties.
    Assesses risk levels, provides risk insights, and generates risk reports.
    """
    
    def __init__(self, model_dir: str = "ml_models"):
        """
        Initialize the Risk Analysis Agent.
        
        Args:
            model_dir (str): Directory containing trained models
        """
        self.model_dir = model_dir
        self.risk_classifier = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.models_loaded = False
        
    def load_models(self):
        """Load pre-trained risk analysis models and preprocessing objects"""
        try:
            logger.info(f"Loading Risk Analysis models from {self.model_dir}/")
            
            self.risk_classifier = joblib.load(f"{self.model_dir}/risk_classifier.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.label_encoders = joblib.load(f"{self.model_dir}/label_encoders.pkl")
            self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
            
            self.models_loaded = True
            logger.info("Risk Analysis models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading Risk Analysis models: {e}")
            logger.error("Please run ml_model_trainer.py first to train the models.")
            raise
    
    def prepare_data_for_risk_analysis(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare filtered data for risk analysis using the same preprocessing as training.
        
        Args:
            filtered_data (pd.DataFrame): Filtered data from Filter Agent
        
        Returns:
            pd.DataFrame: Prepared features for risk analysis
        """
        if not self.models_loaded:
            self.load_models()
            
        logger.info("Preparing data for risk analysis...")
        
        # Create a copy for feature engineering
        features = filtered_data.copy()
        
        # Apply the same feature engineering as training
        features = self._engineer_features(features)
        features = self._handle_missing_values(features)
        features = self._encode_categorical_features(features)
        features = self._select_and_scale_features(features)
        
        logger.info(f"Risk analysis data preparation completed. Shape: {features.shape}")
        return features
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for risk analysis (same as training)"""
        features = data.copy()
        
        # Price features - handle division by zero properly
        # Replace 0 values with NaN to avoid division by zero
        area_marla_safe = features['area_marla'].replace(0, np.nan)
        features['price_per_marla'] = features['price'] / area_marla_safe
        features['price_per_sqft'] = features['price'] / (area_marla_safe * 272.25)
        
        # Area features
        features['area_category'] = pd.cut(features['area_marla'],
                                         bins=[0, 5, 10, 25, float('inf')],
                                         labels=['Small', 'Medium', 'Large', 'Extra Large'],
                                         include_lowest=True)
        
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
        
        # Risk-specific features
        features['price_to_area_ratio'] = features['price'] / area_marla_safe
        features['is_affordable'] = features['price'] <= features['price'].quantile(0.5)
        features['is_premium'] = features['price'] >= features['price'].quantile(0.8)
        
        # Historical trend features for risk analysis
        if 'price_index_now' in features.columns and 'price_index_6mo' in features.columns:
            # 6-month growth rate
            features['growth_6mo'] = ((features['price_index_now'] - features['price_index_6mo']) / 
                                     features['price_index_6mo'].replace(0, np.nan)) * 100
            
            # 12-month growth rate
            if 'price_index_12mo' in features.columns:
                features['growth_12mo'] = ((features['price_index_now'] - features['price_index_12mo']) / 
                                         features['price_index_12mo'].replace(0, np.nan)) * 100
            
            # 24-month growth rate
            if 'price_index_24mo' in features.columns:
                features['growth_24mo'] = ((features['price_index_now'] - features['price_index_24mo']) / 
                                         features['price_index_24mo'].replace(0, np.nan)) * 100
            
            # Annualized growth rate
            features['annualized_growth'] = features['growth_24mo'] / 2
            
            # Recent momentum
            features['recent_momentum'] = features['growth_6mo'] - (features['growth_12mo'] / 2)
            
            # Growth acceleration
            features['growth_acceleration'] = features['growth_6mo'] - features['growth_12mo']
        
        # Use existing volatility and trend if available
        if 'price_volatility' in features.columns:
            features['market_volatility'] = features['price_volatility']
        else:
            features['market_volatility'] = 0
            
        if 'price_trend' in features.columns:
            features['price_trend_direction'] = features['price_trend']
        else:
            features['price_trend_direction'] = 0
        
        # Trend-based risk features
        if 'growth_6mo' in features.columns:
            features['is_volatile_market'] = (features['growth_6mo'] > 10) | (features['growth_6mo'] < -5)
            features['is_stable_market'] = (features['growth_6mo'] >= -2) & (features['growth_6mo'] <= 2)
            features['is_declining_market'] = features['growth_6mo'] < -2
            features['is_growing_market'] = features['growth_6mo'] > 5
        
        return features
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Fill numeric columns with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown', inplace=True)
        
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
        """Select and scale features for risk analysis"""
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
    
    def predict_risks(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict risk levels for filtered properties.
        
        Args:
            filtered_data (pd.DataFrame): Filtered data from Filter Agent
            
        Returns:
            pd.DataFrame: Original data with risk predictions
        """
        if not self.models_loaded:
            self.load_models()
        
        logger.info("Making risk predictions...")
        
        # Prepare features
        features = self.prepare_data_for_risk_analysis(filtered_data)
        
        # Make risk predictions
        risk_predictions = self.risk_classifier.predict(features)
        risk_probabilities = self.risk_classifier.predict_proba(features)
        
        # Add predictions to original data
        results = filtered_data.copy()
        results['predicted_risk_level'] = risk_predictions
        results['risk_confidence'] = np.max(risk_probabilities, axis=1)
        
        # Map risk levels to descriptions and categories
        results['risk_description'] = results['predicted_risk_level'].map({
            1: 'Very Low Risk',
            2: 'Low Risk', 
            3: 'Medium Risk',
            4: 'High Risk',
            5: 'Very High Risk'
        })
        
        results['risk_category'] = results['predicted_risk_level'].map({
            1: 'Safe Investment',
            2: 'Low Risk Investment',
            3: 'Moderate Risk Investment',
            4: 'High Risk Investment',
            5: 'Very High Risk Investment'
        })
        
        # Calculate risk-adjusted metrics
        results['risk_adjusted_score'] = self._calculate_risk_adjusted_score(results)
        results['risk_recommendation'] = self._get_risk_recommendation(results)
        
        logger.info(f"Risk predictions completed for {len(results)} properties")
        return results
    
    def _calculate_risk_adjusted_score(self, results: pd.DataFrame) -> pd.Series:
        """Calculate risk-adjusted investment score"""
        # Base score (inverse of risk level)
        base_score = 6 - results['predicted_risk_level']
        
        # Adjust for confidence
        confidence_adjustment = results['risk_confidence'] * 0.5
        
        # Adjust for price (affordable properties get bonus)
        price_adjustment = np.where(results['price'] <= results['price'].quantile(0.5), 0.5, 0)
        
        # Calculate final risk-adjusted score
        risk_adjusted_score = base_score + confidence_adjustment + price_adjustment
        
        return risk_adjusted_score.clip(1, 10)  # Scale 1-10
    
    def _get_risk_recommendation(self, results: pd.DataFrame) -> pd.Series:
        """Generate risk-based investment recommendations"""
        recommendations = []
        
        for _, row in results.iterrows():
            risk_level = row['predicted_risk_level']
            risk_score = row['risk_adjusted_score']
            price = row['price']
            
            # Base recommendation on risk level
            if risk_level <= 2:
                base_rec = "Safe Investment"
            elif risk_level == 3:
                base_rec = "Moderate Risk"
            elif risk_level == 4:
                base_rec = "High Risk"
            else:
                base_rec = "Very High Risk"
            
            # Add risk score context
            if risk_score >= 8:
                score_context = " (Excellent Risk-Adjusted Return)"
            elif risk_score >= 6:
                score_context = " (Good Risk-Adjusted Return)"
            elif risk_score >= 4:
                score_context = " (Moderate Risk-Adjusted Return)"
            else:
                score_context = " (Poor Risk-Adjusted Return)"
            
            # Add price context
            if price <= 10000000:
                price_context = " - Suitable for Small Investors"
            elif price >= 50000000:
                price_context = " - Suitable for Large Investors"
            else:
                price_context = " - Suitable for Medium Investors"
            
            recommendations.append(base_rec + score_context + price_context)
        
        return pd.Series(recommendations, index=results.index)
    
    def get_risk_summary(self, results: pd.DataFrame) -> Dict:
        """
        Get a comprehensive summary of risk analysis.
        
        Args:
            results (pd.DataFrame): Results with risk predictions
            
        Returns:
            Dict: Summary statistics
        """
        summary = {
            "total_properties": len(results),
            "risk_distribution": results['predicted_risk_level'].value_counts().sort_index().to_dict(),
            "risk_categories": results['risk_category'].value_counts().to_dict(),
            "risk_statistics": {
                "mean_risk_level": results['predicted_risk_level'].mean(),
                "median_risk_level": results['predicted_risk_level'].median(),
                "mean_risk_confidence": results['risk_confidence'].mean(),
                "mean_risk_adjusted_score": results['risk_adjusted_score'].mean()
            },
            "risk_recommendations": results['risk_recommendation'].value_counts().to_dict(),
            "safest_investments": {
                "lowest_risk": results.loc[results['predicted_risk_level'].idxmin()].to_dict(),
                "best_risk_adjusted": results.loc[results['risk_adjusted_score'].idxmax()].to_dict()
            },
            "risk_insights": {
                "high_risk_count": len(results[results['predicted_risk_level'] >= 4]),
                "low_risk_count": len(results[results['predicted_risk_level'] <= 2]),
                "safe_investment_percentage": len(results[results['predicted_risk_level'] <= 2]) / len(results) * 100
            }
        }
        
        return summary
    
    def get_safe_investments(self, results: pd.DataFrame, max_risk_level: int = 3) -> pd.DataFrame:
        """
        Get safe investment opportunities based on risk level.
        
        Args:
            results (pd.DataFrame): Results with risk predictions
            max_risk_level (int): Maximum acceptable risk level (1-5)
            
        Returns:
            pd.DataFrame: Safe investment opportunities
        """
        # Filter for safe investments
        safe_investments = results[results['predicted_risk_level'] <= max_risk_level].copy()
        
        # Sort by risk-adjusted score (best first)
        safe_investments = safe_investments.sort_values('risk_adjusted_score', ascending=False)
        
        # Select relevant columns for display
        display_columns = [
            'title', 'city', 'location', 'price', 'area_marla', 'type',
            'predicted_risk_level', 'risk_description', 'risk_adjusted_score',
            'risk_confidence', 'risk_recommendation'
        ]
        
        return safe_investments[display_columns]
    
    def get_risk_analysis_report(self, results: pd.DataFrame) -> Dict:
        """
        Generate a comprehensive risk analysis report.
        
        Args:
            results (pd.DataFrame): Results with risk predictions
            
        Returns:
            Dict: Comprehensive risk analysis report
        """
        report = {
            "executive_summary": {
                "total_properties_analyzed": len(results),
                "average_risk_level": results['predicted_risk_level'].mean(),
                "risk_distribution": results['predicted_risk_level'].value_counts().sort_index().to_dict(),
                "safe_investment_percentage": len(results[results['predicted_risk_level'] <= 2]) / len(results) * 100
            },
            "risk_breakdown": {
                "by_city": results.groupby('city')['predicted_risk_level'].mean().to_dict(),
                "by_property_type": results.groupby('type')['predicted_risk_level'].mean().to_dict(),
                "by_price_range": self._analyze_risk_by_price_range(results)
            },
            "top_safe_investments": self.get_safe_investments(results, max_risk_level=2).head(5).to_dict('records'),
            "risk_insights": {
                "highest_risk_properties": results[results['predicted_risk_level'] >= 4].head(3).to_dict('records'),
                "best_risk_adjusted_returns": results.nlargest(5, 'risk_adjusted_score').to_dict('records'),
                "confidence_analysis": {
                    "high_confidence_count": len(results[results['risk_confidence'] >= 0.8]),
                    "low_confidence_count": len(results[results['risk_confidence'] < 0.6]),
                    "average_confidence": results['risk_confidence'].mean()
                }
            },
            "recommendations": {
                "for_conservative_investors": self._get_conservative_recommendations(results),
                "for_moderate_investors": self._get_moderate_recommendations(results),
                "for_aggressive_investors": self._get_aggressive_recommendations(results)
            }
        }
        
        return report
    
    def _analyze_risk_by_price_range(self, results: pd.DataFrame) -> Dict:
        """Analyze risk levels by price ranges"""
        price_ranges = [
            (0, 10000000, "Budget (0-10M)"),
            (10000000, 25000000, "Mid-Range (10M-25M)"),
            (25000000, 50000000, "High-Range (25M-50M)"),
            (50000000, float('inf'), "Premium (50M+)")
        ]
        
        risk_by_price = {}
        for min_price, max_price, label in price_ranges:
            mask = (results['price'] >= min_price) & (results['price'] < max_price)
            if mask.sum() > 0:
                avg_risk = results[mask]['predicted_risk_level'].mean()
                risk_by_price[label] = round(avg_risk, 2)
        
        return risk_by_price
    
    def _get_conservative_recommendations(self, results: pd.DataFrame) -> List[Dict]:
        """Get recommendations for conservative investors"""
        conservative = results[
            (results['predicted_risk_level'] <= 2) & 
            (results['risk_adjusted_score'] >= 7)
        ].head(3)
        
        return conservative.to_dict('records')
    
    def _get_moderate_recommendations(self, results: pd.DataFrame) -> List[Dict]:
        """Get recommendations for moderate investors"""
        moderate = results[
            (results['predicted_risk_level'] <= 3) & 
            (results['risk_adjusted_score'] >= 6)
        ].head(3)
        
        return moderate.to_dict('records')
    
    def _get_aggressive_recommendations(self, results: pd.DataFrame) -> List[Dict]:
        """Get recommendations for aggressive investors"""
        aggressive = results[
            (results['predicted_risk_level'] <= 4) & 
            (results['risk_adjusted_score'] >= 5)
        ].head(3)
        
        return aggressive.to_dict('records')
    
    def save_risk_results(self, results: pd.DataFrame, output_path: str = "risk_analysis.csv"):
        """
        Save risk analysis results to CSV.
        
        Args:
            results (pd.DataFrame): Results with risk predictions
            output_path (str): Path to save the results
        """
        results.to_csv(output_path, index=False)
        logger.info(f"Risk analysis results saved to {output_path}")

# Example usage
if __name__ == "__main__":
    try:
        print("⚠️ Starting Risk Analysis System...")
        print("="*50)
        
        # Initialize risk analysis agent
        risk_agent = RiskAnalysisAgent()
        
        # Load models
        risk_agent.load_models()
        
        print("✅ Risk Analysis models loaded successfully!")
        print("🎯 Ready to assess investment risks and provide risk insights")
        print("💡 Use this with your Filter Agent to get risk analysis")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure models are trained first using ml_model_trainer.py") 