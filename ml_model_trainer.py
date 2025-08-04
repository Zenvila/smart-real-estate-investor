#!/usr/bin/env python3
"""
Fast ML Model Trainer: Quick training without grid search for immediate results.
This version skips hyperparameter tuning to get results in 5-10 minutes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastMLTrainer:
    """
    Fast ML Trainer: Trains models quickly without grid search for immediate results.
    """
    
    def __init__(self, data_path: str = "ml_ready_data.csv"):
        self.data_path = data_path
        self.data = None
        
        # Models
        self.roi_regressor = None
        self.risk_classifier = None
        self.price_regressor = None
        
        # Preprocessing
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
        # Training status
        self.trained = False
        
    def load_full_data(self) -> pd.DataFrame:
        """Load the full dataset for training"""
        try:
            logger.info(f"Loading full dataset from {self.data_path}")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.data_path, encoding=encoding)
                    logger.info(f"Successfully loaded data with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    if encoding == encodings[-1]:
                        raise
                    continue
            
            logger.info(f"Loaded {len(self.data)} properties for training")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_training_data(self) -> pd.DataFrame:
        """Prepare the dataset for ML training"""
        if self.data is None:
            self.load_full_data()
            
        logger.info("Preparing dataset for ML training...")
        
        # Create a copy for feature engineering
        features = self.data.copy()
        
        # 1. Create target variables
        features['roi_target'] = self._create_roi_target(features)
        features['risk_level'] = self._create_risk_target(features)
        features['log_price'] = np.log(features['price'] + 1)
        
        # 2. Feature engineering (simplified)
        features = self._engineer_features(features)
        
        # 3. Handle missing values
        features = self._handle_missing_values(features)
        
        # 4. Encode categorical variables
        features = self._encode_categorical_features(features)
        
        # 5. Select and scale features
        features = self._select_and_scale_features(features)
        
        logger.info(f"Training data preparation completed. Shape: {features.shape}")
        return features
    
    def _create_roi_target(self, data: pd.DataFrame) -> pd.Series:
        """Create ROI target for regression"""
        logger.info("Creating ROI target...")
        
        # Simplified ROI calculation
        roi_factors = []
        
        # Price-based ROI (inverse relationship)
        price_percentiles = data['price'].quantile([0.2, 0.4, 0.6, 0.8])
        price_roi = 15 - pd.cut(data['price'], 
                                bins=[0] + list(price_percentiles) + [float('inf')],
                                labels=[12, 10, 8, 6, 4], include_lowest=True).astype(int)
        roi_factors.append(price_roi)
        
        # Location-based ROI
        location_roi = data['city'].str.split('_').str[0].map({
            'islamabad': 8.5, 'lahore': 7.5, 'karachi': 9.0, 'rawalpindi': 7.0
        }).fillna(7.5)
        roi_factors.append(location_roi)
        
        # Property type ROI
        type_roi = data['type'].map({
            'house': 8.0, 'flat': 9.5, 'plot': 6.0, 'shop': 12.0,
            'office': 10.0, 'building': 11.0, 'factory': 15.0, 'other': 7.5
        }).fillna(7.5)
        roi_factors.append(type_roi)
        
        # Area-based ROI
        area_roi = np.where(data['area_marla'] <= 5, 10.0,
                   np.where(data['area_marla'] <= 10, 8.5,
                   np.where(data['area_marla'] <= 25, 7.0, 6.0)))
        roi_factors.append(pd.Series(area_roi, index=data.index))
        
        # Calculate average ROI
        roi_target = pd.concat(roi_factors, axis=1).mean(axis=1)
        
        # Add noise and clip
        noise = np.random.normal(0, 1.5, len(roi_target))
        roi_target = (roi_target + noise).clip(3, 20)
        
        logger.info(f"ROI target distribution: {roi_target.describe()}")
        return roi_target
    
    def _create_risk_target(self, data: pd.DataFrame) -> pd.Series:
        """Create risk level target for classification"""
        logger.info("Creating risk level target...")
        
        # Simplified risk calculation
        risk_factors = []
        
        # Price risk
        price_percentiles = data['price'].quantile([0.2, 0.4, 0.6, 0.8])
        price_risk = pd.cut(data['price'], 
                           bins=[0] + list(price_percentiles) + [float('inf')],
                           labels=[1, 2, 3, 4, 5], include_lowest=True)
        risk_factors.append(price_risk.astype(int))
        
        # Location risk
        location_risk = data['city'].str.split('_').str[0].map({
            'islamabad': 2, 'lahore': 3, 'karachi': 4, 'rawalpindi': 3
        }).fillna(3)
        risk_factors.append(location_risk)
        
        # Property type risk
        type_risk = data['type'].map({
            'house': 3, 'flat': 2, 'plot': 4, 'shop': 4,
            'office': 3, 'building': 5, 'factory': 5, 'other': 3
        }).fillna(3)
        risk_factors.append(type_risk)
        
        # Calculate average risk level
        risk_level = pd.concat(risk_factors, axis=1).mean(axis=1).round().astype(int)
        risk_level = risk_level.clip(1, 5)
        
        logger.info(f"Risk levels distribution: {risk_level.value_counts().sort_index().to_dict()}")
        return risk_level
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for ML training (simplified)"""
        features = data.copy()
        
        # Basic price features
        area_marla_safe = features['area_marla'].replace(0, np.nan)
        features['price_per_marla'] = features['price'] / area_marla_safe
        features['price_per_sqft'] = features['price'] / (area_marla_safe * 272.25)
        
        # Basic features
        features['log_price'] = np.log(features['price'] + 1)
        features['log_area'] = np.log(area_marla_safe + 1)
        
        # Location features
        features['city_main'] = features['city'].str.split('_').str[0]
        features['city_category'] = features['city'].str.split('_').str[1]
        
        # Property type features
        features['property_category'] = features['type'].map({
            'plot': 'Land', 'house': 'Residential', 'flat': 'Residential',
            'shop': 'Commercial', 'office': 'Commercial', 'building': 'Commercial',
            'factory': 'Industrial', 'other': 'Other'
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
        logger.info("Handling missing values...")
        
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
        """Encode categorical features"""
        logger.info("Encoding categorical features...")
        
        categorical_cols = ['city_main', 'city_category', 'property_category', 'type', 'purpose']
        
        for col in categorical_cols:
            if col in data.columns:
                le = LabelEncoder()
                data[f'{col}_encoded'] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
        
        return data
    
    def _select_and_scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Select and scale features for ML models"""
        logger.info("Selecting and scaling features...")
        
        # Select basic features (removed log_price to prevent data leakage)
        feature_cols = [
            'price_per_marla', 'price_per_sqft', 'area_marla',
            'city_main_encoded', 'city_category_encoded', 'property_category_encoded',
            'type_encoded', 'purpose_encoded',
            'has_bedrooms', 'has_bathrooms', 'has_description', 'has_area',
            'price_to_area_ratio', 'is_affordable', 'is_premium',
            'log_area'  # Removed log_price to prevent data leakage
        ]
        
        # Filter to only available columns
        available_features = [col for col in feature_cols if col in data.columns]
        
        # Create feature matrix
        X = data[available_features].copy()
        
        # Handle infinity and extreme values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for each column
        for col in X.columns:
            if X[col].isnull().sum() > 0:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create final feature dataframe
        features_df = pd.DataFrame(X_scaled, columns=available_features, index=data.index)
        
        # Add target variables
        features_df['roi_target'] = data['roi_target']
        features_df['risk_level'] = data['risk_level']
        features_df['log_price'] = data['log_price']
        
        self.feature_columns = available_features
        
        logger.info(f"Final feature matrix shape: {features_df.shape}")
        return features_df
    
    def train_all_models(self, test_size: float = 0.2, random_state: int = 42):
        """Train all three models quickly without grid search"""
        logger.info("Starting fast model training...")
        
        # Prepare training data
        training_data = self.prepare_training_data()
        
        # Split features and targets
        X = training_data[self.feature_columns]
        y_roi = training_data['roi_target']
        y_risk = training_data['risk_level']
        y_price = training_data['log_price']
        
        # Split data
        X_train, X_test, y_roi_train, y_roi_test, y_risk_train, y_risk_test, y_price_train, y_price_test = train_test_split(
            X, y_roi, y_risk, y_price, test_size=test_size, random_state=random_state, stratify=y_risk
        )
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # 1. Train ROI Regressor (fast)
        logger.info("Training ROI Regressor...")
        self.roi_regressor = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=random_state
        )
        
        self.roi_regressor.fit(X_train, y_roi_train)
        roi_score = self.roi_regressor.score(X_test, y_roi_test)
        roi_rmse = np.sqrt(mean_squared_error(y_roi_test, self.roi_regressor.predict(X_test)))
        logger.info(f"ROI Regressor R² score: {roi_score:.4f}")
        logger.info(f"ROI Regressor RMSE: {roi_rmse:.4f}")
        
        # 2. Train Risk Classifier (fast)
        logger.info("Training Risk Classifier...")
        self.risk_classifier = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
        )
        
        self.risk_classifier.fit(X_train, y_risk_train)
        risk_score = self.risk_classifier.score(X_test, y_risk_test)
        logger.info(f"Risk Classifier accuracy: {risk_score:.4f}")
        
        # 3. Train Price Regressor (fast)
        logger.info("Training Price Regressor...")
        self.price_regressor = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1
        )
        
        self.price_regressor.fit(X_train, y_price_train)
        price_score = self.price_regressor.score(X_test, y_price_test)
        price_rmse = np.sqrt(mean_squared_error(y_price_test, self.price_regressor.predict(X_test)))
        logger.info(f"Price Regressor R² score: {price_score:.4f}")
        logger.info(f"Price Regressor RMSE: {price_rmse:.4f}")
        
        # Cross-validation scores
        roi_cv_scores = cross_val_score(self.roi_regressor, X, y_roi, cv=5)
        risk_cv_scores = cross_val_score(self.risk_classifier, X, y_risk, cv=5)
        price_cv_scores = cross_val_score(self.price_regressor, X, y_price, cv=5)
        
        logger.info(f"ROI Regressor CV R²: {roi_cv_scores.mean():.4f} (+/- {roi_cv_scores.std() * 2:.4f})")
        logger.info(f"Risk Classifier CV accuracy: {risk_cv_scores.mean():.4f} (+/- {risk_cv_scores.std() * 2:.4f})")
        logger.info(f"Price Regressor CV R²: {price_cv_scores.mean():.4f} (+/- {price_cv_scores.std() * 2:.4f})")
        
        self.trained = True
        logger.info("All models trained successfully!")
    
    def save_models(self, model_dir: str = "ml_models"):
        """Save all trained models and preprocessing objects"""
        if not self.trained:
            logger.error("Models not trained yet!")
            return
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models
        joblib.dump(self.roi_regressor, f"{model_dir}/roi_regressor.pkl")
        joblib.dump(self.risk_classifier, f"{model_dir}/risk_classifier.pkl")
        joblib.dump(self.price_regressor, f"{model_dir}/price_regressor.pkl")
        joblib.dump(self.scaler, f"{model_dir}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{model_dir}/label_encoders.pkl")
        joblib.dump(self.feature_columns, f"{model_dir}/feature_columns.pkl")
        
        logger.info(f"All models saved to {model_dir}/")

# Main execution
if __name__ == "__main__":
    try:
        print("🚀 Starting Fast ML Model Training...")
        print("="*60)
        
        # Initialize trainer
        trainer = FastMLTrainer()
        
        # Train all models
        trainer.train_all_models()
        
        # Save models
        trainer.save_models()
        
        print("="*60)
        print("✅ Fast ML Models Training Completed!")
        print("📁 Models saved to ml_models/ directory")
        print("🎯 ROI, Risk, and Price models ready for predictions")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("Please check your data and try again.")