# 🏠 SmartReal Estate Pro - AI-Powered Investment Analyzer

> **Intelligent Real Estate Investment Analysis Platform**  
> *Developed at AIM Lab under the supervision of Saria Qamar*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.8%25-brightgreen.svg)](accuracy.md)

## 🎯 Project Overview

**SmartReal Estate Pro** is an advanced AI-powered real estate investment analysis platform that helps investors make data-driven decisions. The system analyzes property data from major Pakistani cities and provides comprehensive investment insights including ROI predictions, risk assessment, and market analysis.

### ✨ Key Features

- 🎯 **AI-Powered ROI Prediction** (85.2% accuracy)
- 🛡️ **Risk Assessment** (96.8% accuracy) 
- 💰 **Price Prediction** (95.7% accuracy)
- 📊 **Investment Recommendations**
- 🏙️ **Multi-City Analysis** (Islamabad, Karachi, Lahore, Rawalpindi)
- 🌐 **Modern Web Interface**
- 📱 **Responsive Design**

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Git
git --version
```

### 1. Clone the Repository

```bash
git clone https://github.com/Zenvila/smart-real-estate-investor.git
cd House-Price-Prediction-main
```

### 2. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
conda activate real-estate-ai
```

### 3. Run the Application

```bash
# Start the web application
python app.py

# Or use the start script
python start_web_app.py
```

### 4. Access the Application

Open your browser and navigate to: **http://localhost:5000**

## 📁 Project Structure

```
House-Price-Prediction-main/
├── 📁 ml_models/              # Trained ML models
├── 📁 static/                 # Web assets
│   ├── css/style.css         # Styling
│   └── js/script.js          # Frontend logic
├── 📁 templates/              # HTML templates
├── 📄 app.py                  # Flask web application
├── 📄 filter_agent.py         # Data filtering system
├── 📄 prediction_agent.py     # ML prediction engine
├── 📄 ml_model_trainer.py     # Model training script
├── 📄 requirements.txt        # Python dependencies
├── 📄 accuracy.md             # Model performance report
└── 📄 README.md              # This file
```

## 🛠️ Installation Guide

### Option 1: Standard Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/House-Price-Prediction-main.git
cd House-Price-Prediction-main

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
```



## 🎮 How to Use

### 1. **Search Properties**
- Select your preferred location
- Set budget range (min/max)
- Choose property type
- Click "Search Properties"

### 2. **Analyze Results**
- View ROI predictions for each property
- Check risk assessment levels
- Review investment recommendations
- Compare multiple properties

### 3. **Detailed Analysis**
- Click on any property for detailed view
- See comprehensive investment metrics
- Get personalized recommendations

## 🤖 AI Models

### ROI Prediction Model
- **Algorithm**: Gradient Boosting Regressor
- **Accuracy**: 85.2% R² Score
- **Features**: 16 engineered features
- **Output**: ROI percentage (3.2% - 18.7%)

### Risk Assessment Model
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 96.8%
- **Features**: Market stability indicators
- **Output**: Risk levels (1-5 scale)

### Price Prediction Model
- **Algorithm**: Gradient Boosting Regressor
- **Accuracy**: 95.7% R² Score
- **Features**: Property and market features
- **Output**: Market value predictions

## 📊 Data Sources

- **Properties**: 57,979 real estate listings
- **Cities**: Islamabad, Karachi, Lahore, Rawalpindi
- **Property Types**: Houses, Flats, Plots, Commercial
- **Price Range**: PKR 500K - 500M
- **Data Freshness**: Regularly updated

## 🔧 API Endpoints

### Web Interface
- `GET /` - Main application page
- `GET /api/available_options` - Get filter options
- `POST /api/search` - Search properties
- `GET /api/property/<id>` - Get property details

### Example API Usage

```python
import requests

# Search properties
response = requests.post('http://localhost:5000/api/search', json={
    'location': 'Lahore',
    'min_budget': 1000000,
    'max_budget': 5000000
})

# Get available options
options = requests.get('http://localhost:5000/api/available_options')
```

## 🧪 Testing

```bash
# Run all tests
python test_web_app.py

# Test specific components
python -m pytest tests/

# Performance testing
python benchmark_models.py
```

## 📈 Performance Metrics

| Component | Metric | Performance |
|-----------|--------|-------------|
| **ROI Model** | R² Score | 85.2% |
| **Risk Model** | Accuracy | 96.8% |
| **Price Model** | R² Score | 95.7% |
| **Web App** | Response Time | <500ms |
| **Data Processing** | Throughput | 1000+ properties/sec |



## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Credits

### Development Team
- **Haris** - AI Engineer
- **Munhim** - ML Engineer
- **Saria Qamar** - Project Supervisor

### Research Lab
- **AIM Lab** - Artificial Intelligence & Machine Learning Laboratory
- **Supervision**: Saria Qamar
- **Research Focus**: Real Estate Investment Analysis



## 🔄 Updates

### Latest Version: v2.1.0
- ✅ Enhanced ROI prediction accuracy
- ✅ Improved risk assessment model
- ✅ Modern web interface
- ✅ Real-time data processing
- ✅ Mobile-responsive design

### Upcoming Features
- 🔄 Real-time market data integration
- 🔄 Advanced portfolio analysis
- 🔄 Mobile application
- 🔄 API rate limiting
- 🔄 Multi-language support

---

**Built with dedication at AIM Lab**  
 *Developed by Haris & Munhim*
 *Supervised by Saria Qamar* 
---

