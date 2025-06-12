# US Open 2025 Prediction System - Complete Overview

## 🏆 What This System Does

This is a comprehensive machine learning system that predicts the 2025 US Open golf tournament outcomes using historical data from the DataGolf API. It implements the **Method B** approach you described - using historical data to train models that can predict tournament performance.

## 🎯 Key Features

### Data Collection
- **Historical US Open Results**: 2017-2024 tournament data
- **Major Championship Data**: All majors for broader context  
- **Recent PGA Tour Data**: Last 3 years for current form
- **Player Skills**: Current strokes-gained ratings
- **Player Rankings**: DataGolf rankings

### Advanced Feature Engineering
- **Recent Form**: Performance over last 4, 8, 16 tournaments
- **Course History**: US Open and similar course performance
- **Major Championship Experience**: Performance in major championships
- **Skill Trends**: Improving/declining performance patterns
- **Current Skills**: Strokes-gained in putting, approach, off-the-tee, etc.

### Machine Learning Models
- **XGBoost**: Primary model (usually best performance)
- **Random Forest**: Baseline model
- **Multiple Targets**: Win, Top 5, Top 10, Top 20 predictions

### Comprehensive Analysis
- **Feature Importance**: Which factors matter most
- **Model Performance**: Cross-validation and metrics
- **Interactive Visualizations**: Jupyter notebook analysis
- **Detailed Reports**: Human-readable prediction summaries

## 📊 Prediction Outputs

The system generates predictions for:
- **Win Probability**: Chance of winning the tournament
- **Top 5 Probability**: Chance of finishing in top 5
- **Top 10 Probability**: Chance of finishing in top 10  
- **Top 20 Probability**: Chance of finishing in top 20

## 🚀 Quick Start

### 1. Setup (5 minutes)
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API key in .env file
DATAGOLF_API_KEY=be1e0f4c0d741ab978b3fded7e8c
```

### 2. Run Complete System
```bash
# Full pipeline: collect data → process → train models → predict
python main.py --step full
```

### 3. View Results
- **Predictions**: `data/predictions/us_open_2025_predictions_YYYYMMDD_HHMMSS.csv`
- **Report**: `data/predictions/us_open_2025_report_YYYYMMDD_HHMMSS.txt`
- **Analysis**: `notebooks/us_open_analysis.ipynb`

## 📁 Project Structure

```
usopenprediction/
├── src/
│   ├── data_collection/        # DataGolf API integration
│   │   ├── datagolf_client.py  # API client
│   │   └── collect_historical_data.py  # Data collection
│   ├── preprocessing/          # Data cleaning & features
│   │   ├── data_cleaner.py     # Data cleaning
│   │   └── feature_engineering.py  # Feature creation
│   ├── modeling/              # Machine learning
│   │   └── tournament_predictor.py  # ML models
│   └── prediction/            # US Open predictions
│       └── us_open_2025_predictor.py  # Main predictor
├── data/
│   ├── raw/                   # Raw API data
│   ├── processed/             # Cleaned data
│   └── predictions/           # Model outputs
├── notebooks/                 # Jupyter analysis
├── config/                    # Configuration
└── main.py                    # Main execution script
```

## 🔬 Technical Approach

### Data Pipeline
1. **Collection**: Fetch historical data via DataGolf API
2. **Cleaning**: Handle missing values, outliers, data quality
3. **Feature Engineering**: Create predictive features from raw data
4. **Model Training**: Train ML models on historical outcomes
5. **Prediction**: Apply models to current field

### Feature Categories
- **Form Features**: Recent tournament performance (4, 8, 16 events)
- **Course Features**: Historical performance at US Open/similar courses
- **Major Features**: Performance in major championships
- **Skill Features**: Current strokes-gained ratings
- **Trend Features**: Improving/declining performance patterns

### Model Architecture
- **Ensemble Approach**: Multiple models for robustness
- **Cross-Validation**: 5-fold CV for model selection
- **Feature Importance**: Understand what drives predictions
- **Multiple Targets**: Different prediction objectives

## 📈 Example Output

```
TOP WIN CANDIDATES:
------------------------------
 1. Scottie Scheffler        8.2%
 2. Rory McIlroy            6.1%
 3. Jon Rahm                5.4%
 4. Viktor Hovland          4.8%
 5. Xander Schauffele       4.2%

TOP 10 CANDIDATES:
------------------------------
 1. Scottie Scheffler       45.2%
 2. Rory McIlroy           38.1%
 3. Jon Rahm               35.4%
 4. Viktor Hovland         32.8%
 5. Xander Schauffele      30.2%
```

## 🛠️ System Components

### 1. DataGolf API Client (`datagolf_client.py`)
- Handles all API interactions
- Rate limiting and error handling
- Multiple endpoint support

### 2. Data Collection (`collect_historical_data.py`)
- Systematic historical data gathering
- US Open, majors, and PGA Tour data
- Player information and current stats

### 3. Data Cleaning (`data_cleaner.py`)
- Missing value handling
- Outlier detection and removal
- Data quality validation

### 4. Feature Engineering (`feature_engineering.py`)
- Form-based features
- Course history features
- Major championship features
- Skill trend analysis

### 5. Tournament Predictor (`tournament_predictor.py`)
- Multiple ML algorithms
- Cross-validation and evaluation
- Feature importance analysis

### 6. US Open Predictor (`us_open_2025_predictor.py`)
- Complete prediction pipeline
- Report generation
- Results visualization

## 🎯 Methodology Validation

The system implements best practices for sports prediction:

1. **Historical Validation**: Models trained on 7+ years of data
2. **Cross-Validation**: Robust model evaluation
3. **Feature Engineering**: Golf-specific predictive features
4. **Ensemble Methods**: Multiple models for stability
5. **Interpretability**: Feature importance and model explanation

## 🔧 Configuration

Key settings in `config/model_config.py`:
- **Data Collection**: Years, tournaments, rate limits
- **Feature Engineering**: Lookback periods, trend windows
- **Model Training**: Hyperparameters, validation settings
- **Prediction**: Targets, confidence thresholds

## 📊 Performance Metrics

The system tracks:
- **Accuracy**: Classification accuracy for binary targets
- **AUC**: Area under ROC curve for probability predictions
- **Cross-Validation**: 5-fold CV scores
- **Feature Importance**: Which factors matter most

## 🚀 Advanced Usage

### Custom Analysis
```python
from src.prediction.us_open_2025_predictor import USOpen2025Predictor

predictor = USOpen2025Predictor()
predictions, report = predictor.run_full_prediction()
```

### Jupyter Notebook Analysis
```bash
jupyter notebook notebooks/us_open_analysis.ipynb
```

### Individual Components
```python
# Just collect data
python main.py --step collect

# Just process data  
python main.py --step process

# Just make predictions
python main.py --step predict
```

## 🎯 Next Steps

1. **Run the System**: Execute the full pipeline
2. **Analyze Results**: Review predictions and feature importance
3. **Compare with Odds**: Validate against sportsbook odds
4. **Track Performance**: Monitor actual tournament results
5. **Refine Models**: Improve based on performance

## 🏆 Success Metrics

The system is successful if it:
- ✅ Collects comprehensive historical data
- ✅ Generates reasonable probability predictions
- ✅ Identifies key predictive factors
- ✅ Provides actionable insights
- ✅ Outperforms simple baselines

This system represents a professional-grade approach to golf tournament prediction, combining domain expertise with modern machine learning techniques.
