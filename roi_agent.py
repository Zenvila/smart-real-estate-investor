#!/usr/bin/env python3
"""
ROI Agent: Predicts Return on Investment (ROI) for real estate properties.
Uses pre-trained ML models to estimate rental yields and investment returns.
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

class ROIAgent:
    """
    ROI Agent: Predicts Return on Investment for real estate properties.
    Estimates rental yields, investment returns, and profitability metrics.
    """
    
    def __init__(self, model_dir: str = "ml_models"):
        """
        Initialize the ROI Agent.
        
        Args:
            model_dir (str): Directory containing trained models
        """
        self.model_dir = model_dir
        self.roi_regressor = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.models_loaded = False
        
    def load_models(self):
        """Load pre-trained ROI model and preprocessing objects"""
        try:
            logger.info(f"Loading ROI models from {self.model_dir}/")
            
            self.roi_regressor = joblib.load(f"{self.model_dir}/roi_regressor.pkl")
            self.scaler = joblib.load(f"{self.model_dir}/scaler.pkl")
            self.label_encoders = joblib.load(f"{self.model_dir}/label_encoders.pkl")
            self.feature_columns = joblib.load(f"{self.model_dir}/feature_columns.pkl")
            
            self.models_loaded = True
            logger.info("ROI models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading ROI models: {e}")
            logger.error("Please run ml_model_trainer.py first to train the models.")
            raise
    
    def prepare_data_for_roi(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare filtered data for ROI predictions using the same preprocessing as training.
        
        Args:
            filtered_data (pd.DataFrame): Filtered data from Filter Agent
            
        Returns:
            pd.DataFrame: Prepared features for ROI prediction
        """
        if not self.models_loaded:
            self.load_models()
        
        logger.info("Preparing data for ROI predictions...")
        
        # Create a copy for feature engineering
        features = filtered_data.copy()
        
        # Apply the same feature engineering as training
        features = self._engineer_features(features)
        features = self._handle_missing_values(features)
        features = self._encode_categorical_features(features)
        features = self._select_and_scale_features(features)
        
        logger.info(f"ROI data preparation completed. Shape: {features.shape}")
        return features
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ROI predictions (same as training)"""
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
        
        # Investment features
        features['price_to_area_ratio'] = features['price'] / area_marla_safe
        features['is_affordable'] = features['price'] <= features['price'].quantile(0.5)
        features['is_premium'] = features['price'] >= features['price'].quantile(0.8)
        
        # Historical trend features for ROI
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
        
        # Trend-based investment features
        if 'growth_6mo' in features.columns:
            features['is_growing_market'] = features['growth_6mo'] > 0
            features['is_high_growth'] = features['growth_6mo'] > 5
            features['is_stable_market'] = (features['growth_6mo'] >= -2) & (features['growth_6mo'] <= 2)
            features['is_declining_market'] = features['growth_6mo'] < -2
        
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
        """Select and scale features for ROI predictions"""
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
    
    def predict_roi(self, filtered_data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict ROI (Return on Investment) for filtered properties.
        
        Args:
            filtered_data (pd.DataFrame): Filtered data from Filter Agent
            
        Returns:
            pd.DataFrame: Original data with ROI predictions
        """
        if not self.models_loaded:
            self.load_models()
        
        logger.info("Making ROI predictions...")
        
        # Prepare features
        features = self.prepare_data_for_roi(filtered_data)
        
        # Make ROI predictions
        roi_predictions = self.roi_regressor.predict(features)
        
        # Add predictions to original data
        results = filtered_data.copy()
        results['predicted_roi_percent'] = roi_predictions
        results['predicted_roi_decimal'] = roi_predictions / 100
        
        # Calculate additional ROI metrics
        results['estimated_annual_rent'] = results['price'] * results['predicted_roi_decimal']
        results['monthly_rent_estimate'] = results['estimated_annual_rent'] / 12
        
        # Create categorized ROI with proper index
        roi_categories = self._categorize_roi(roi_predictions)
        results['roi_category'] = roi_categories.values
        
        # Create investment recommendations with proper index
        investment_recs = self._get_investment_recommendation(results)
        results['investment_recommendation'] = investment_recs.values
        
        logger.info(f"ROI predictions completed for {len(results)} properties")
        return results
    
    def _categorize_roi(self, roi_predictions: np.ndarray) -> pd.Series:
        """Categorize ROI predictions into investment categories"""
        categories = []
        for roi in roi_predictions:
            if roi >= 12:
                categories.append('Excellent Investment')
            elif roi >= 9:
                categories.append('Good Investment')
            elif roi >= 6:
                categories.append('Moderate Investment')
            elif roi >= 4:
                categories.append('Low Return')
            else:
                categories.append('Poor Investment')
        
        # Create a pandas Series with a proper index
        return pd.Series(categories, index=range(len(categories)))
    
    def _get_investment_recommendation(self, results: pd.DataFrame) -> pd.Series:
        """Generate investment recommendations based on ROI and other factors"""
        recommendations = []
        
        for i, row in results.iterrows():
            try:
                roi = row['predicted_roi_percent']
                price = row['price']
                area = row['area_marla']
                
                # Base recommendation on ROI
                if roi >= 12:
                    base_rec = "Strong Buy"
                elif roi >= 9:
                    base_rec = "Buy"
                elif roi >= 6:
                    base_rec = "Consider"
                elif roi >= 4:
                    base_rec = "Hold"
                else:
                    base_rec = "Avoid"
                
                # Adjust based on price and area
                if price <= 10000000 and area <= 10:  # Affordable and manageable
                    size_factor = " (Good for small investors)"
                elif price >= 50000000 and area >= 25:  # Premium and large
                    size_factor = " (Suitable for large investors)"
                else:
                    size_factor = ""
                
                recommendations.append(base_rec + size_factor)
            except (KeyError, IndexError) as e:
                # Fallback recommendation if data is missing
                recommendations.append("Consider (Limited Data)")
        
        return pd.Series(recommendations, index=results.index)
    
    def _get_best_value_investment(self, results: pd.DataFrame) -> Dict:
        """Safely get the best value investment without causing indexing errors"""
        try:
            # Filter for good ROI and affordable properties
            filtered = results.loc[(results['predicted_roi_percent'] >= 10) & 
                                 (results['price'] <= results['price'].quantile(0.7))]
            
            if len(filtered) > 0:
                return filtered.head(1).iloc[0].to_dict()
            else:
                # Fallback to highest ROI if no affordable options
                return results.loc[results['predicted_roi_percent'].idxmax()].to_dict()
        except Exception:
            # Return empty dict if any error occurs
            return {}
    
    def get_roi_summary(self, results: pd.DataFrame) -> Dict:
        """
        Get a comprehensive summary of ROI predictions.
        
        Args:
            results (pd.DataFrame): Results with ROI predictions
            
        Returns:
            Dict: Summary statistics
        """
        summary = {
            "total_properties": len(results),
            "roi_statistics": {
                "mean_roi": results['predicted_roi_percent'].mean(),
                "median_roi": results['predicted_roi_percent'].median(),
                "min_roi": results['predicted_roi_percent'].min(),
                "max_roi": results['predicted_roi_percent'].max(),
                "std_roi": results['predicted_roi_percent'].std()
            },
            "roi_categories": results['roi_category'].value_counts().to_dict(),
            "investment_recommendations": results['investment_recommendation'].value_counts().to_dict(),
            "top_investments": {
                "highest_roi": results.loc[results['predicted_roi_percent'].idxmax()].to_dict(),
                "best_value": self._get_best_value_investment(results)
            },
            "rental_estimates": {
                "mean_annual_rent": results['estimated_annual_rent'].mean(),
                "mean_monthly_rent": results['monthly_rent_estimate'].mean(),
                "total_annual_rental_potential": results['estimated_annual_rent'].sum()
            }
        }
        
        return summary
    
    def get_top_investments(self, results: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Get top investment opportunities based on ROI.
        
        Args:
            results (pd.DataFrame): Results with ROI predictions
            top_n (int): Number of top investments to return
            
        Returns:
            pd.DataFrame: Top investment opportunities
        """
        # Sort by ROI and return top N
        top_investments = results.sort_values('predicted_roi_percent', ascending=False).head(top_n)
        
        # Select relevant columns for display
        display_columns = [
            'title', 'city', 'location', 'price', 'area_marla', 'type',
            'predicted_roi_percent', 'estimated_annual_rent', 'monthly_rent_estimate',
            'roi_category', 'investment_recommendation'
        ]
        
        return top_investments[display_columns]
    
    def save_roi_results(self, results: pd.DataFrame, output_path: str = "roi_predictions.csv"):
        """
        Save ROI prediction results to CSV.
        
        Args:
            results (pd.DataFrame): Results with ROI predictions
            output_path (str): Path to save the results
        """
        results.to_csv(output_path, index=False)
        logger.info(f"ROI predictions saved to {output_path}")

# Example usage
if __name__ == "__main__":
    try:
        print("💰 Starting ROI Analysis System...")
        print("="*50)
        
        # Initialize ROI agent
        roi_agent = ROIAgent()
        
        # Load models
        roi_agent.load_models()
        
        print("✅ ROI models loaded successfully!")
        print("🎯 Ready to predict rental yields and investment returns")
        print("💡 Use this with your Filter Agent to get ROI predictions")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please ensure models are trained first using ml_model_trainer.py") 