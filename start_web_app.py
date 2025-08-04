#!/usr/bin/env python3
"""
Startup script for SmartReal Estate Pro Web Application
"""

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import joblib
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("📁 Checking data files...")
    
    required_files = [
        "ml_ready_data.csv",
        "ml_models/roi_regressor.pkl",
        "ml_models/risk_classifier.pkl",
        "ml_models/price_regressor.pkl",
        "ml_models/scaler.pkl",
        "ml_models/label_encoders.pkl",
        "ml_models/feature_columns.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all ML models are trained and data files are present.")
        return False
    else:
        print("✅ All required data files found")
        return True

def start_web_app():
    """Start the web application"""
    print("🚀 Starting SmartReal Estate Pro Web Application...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check data files
    if not check_data_files():
        return False
    
    print("\n🎯 Starting Flask server...")
    print("📱 The web interface will be available at: http://localhost:5000")
    print("🔄 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False
    
    return True

def open_browser():
    """Open the web application in the default browser"""
    try:
        time.sleep(3)  # Wait for server to start
        webbrowser.open("http://localhost:5000")
        print("🌐 Opening web application in your browser...")
    except Exception as e:
        print(f"⚠️ Could not open browser automatically: {e}")
        print("Please manually open: http://localhost:5000")

if __name__ == "__main__":
    print("🏠 SmartReal Estate Pro - Web Application Startup")
    print("=" * 60)
    
    # Start the application
    success = start_web_app()
    
    if success:
        print("\n✅ Web application started successfully!")
        print("🎉 Enjoy using SmartReal Estate Pro!")
    else:
        print("\n❌ Failed to start web application")
        print("Please check the error messages above and try again.") 