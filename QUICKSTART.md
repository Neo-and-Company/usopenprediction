# US Open 2025 Prediction System - Quick Start Guide

## Overview

This system uses machine learning to predict the 2025 US Open golf tournament outcomes based on historical data from the DataGolf API.

## Prerequisites

1. **DataGolf API Access**: You need a DataGolf API key (Scratch Plus membership required)
2. **Python 3.8+**: Make sure you have Python installed
3. **Git**: For cloning the repository

## Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up API Key
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your DataGolf API key
# DATAGOLF_API_KEY=your_api_key_here
```

### 3. Run the Complete Pipeline
```bash
python main.py --step full
```

This will:
- Collect historical tournament data
- Process and clean the data
- Train machine learning models
- Generate 2025 US Open predictions

## Step-by-Step Execution

If you prefer to run each step individually:

### Step 1: Collect Data
```bash
python main.py --step collect
```
This downloads historical data from DataGolf API (takes 10-15 minutes).

### Step 2: Process Data
```bash
python main.py --step process
```
This cleans and prepares the data for modeling (takes 2-3 minutes).

### Step 3: Generate Predictions
```bash
python main.py --step predict
```
This trains models and generates predictions (takes 5-10 minutes).

## Understanding the Output

### Prediction Files
Results are saved in `data/predictions/`:
- `us_open_2025_predictions_YYYYMMDD_HHMMSS.csv`: Detailed predictions for each player
- `us_open_2025_report_YYYYMMDD_HHMMSS.txt`: Human-readable summary report

### Key Columns in Predictions
- `won_prob_xgboost`: Probability of winning the tournament
- `top_5_prob_xgboost`: Probability of finishing in top 5
- `top_10_prob_xgboost`: Probability of finishing in top 10
- `top_20_prob_xgboost`: Probability of finishing in top 20

## Advanced Analysis

### Jupyter Notebook
For detailed analysis and visualizations:
```bash
jupyter notebook notebooks/us_open_analysis.ipynb
```

### Custom Configuration
Edit `config/model_config.py` to customize:
- Model parameters
- Feature engineering settings
- Data collection preferences

## Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   ValueError: DataGolf API key is required
   ```
   **Solution**: Make sure your API key is correctly set in the `.env` file.

2. **No Data Collected**
   ```
   No US Open data collected
   ```
   **Solution**: Check your API key and internet connection. DataGolf may have rate limits.

3. **Memory Issues**
   ```
   MemoryError during model training
   ```
   **Solution**: Reduce the number of years in `config/model_config.py` or use a machine with more RAM.

### Getting Help

1. Check the console output for detailed error messages
2. Look at the log files in the data directories
3. Verify your DataGolf API subscription is active

## Understanding the Methodology

### Data Sources
- **Historical US Open Results**: 2017-2024 tournament data
- **Major Championship Data**: All majors for broader context
- **Recent PGA Tour Data**: Last 3 years for current form
- **Player Skills**: Current strokes-gained ratings
- **Player Rankings**: DataGolf rankings

### Features Used
- **Recent Form**: Average performance over last 4, 8, 16 tournaments
- **Course History**: Performance at US Open and similar courses
- **Major Championship Experience**: Performance in major championships
- **Skill Trends**: Improving/declining performance patterns
- **Current Skills**: Strokes-gained in putting, approach, off-the-tee, etc.

### Models
- **XGBoost**: Primary model (usually best performance)
- **LightGBM**: Secondary model
- **Random Forest**: Baseline model

### Targets Predicted
- **Win**: Probability of winning the tournament
- **Top 5**: Probability of finishing in top 5
- **Top 10**: Probability of finishing in top 10
- **Top 20**: Probability of finishing in top 20

## Example Output

```
TOP WIN CANDIDATES:
------------------------------
 1. Scottie Scheffler        8.2%
 2. Rory McIlroy            6.1%
 3. Jon Rahm                5.4%
 4. Viktor Hovland          4.8%
 5. Xander Schauffele       4.2%
```

## Next Steps

1. **Analyze Results**: Use the Jupyter notebook for detailed analysis
2. **Compare with Odds**: Compare predictions with sportsbook odds
3. **Track Performance**: Save predictions and compare with actual results
4. **Refine Models**: Adjust parameters based on performance

## Data Update Schedule

- **Weekly**: Update current rankings and skills
- **After Major Events**: Retrain models with new data
- **Pre-Tournament**: Final predictions with latest field information

## Tips for Best Results

1. **Run Close to Tournament**: Data is most accurate closer to the event
2. **Check Field Updates**: Monitor withdrawals and late additions
3. **Consider Course Conditions**: Weather and course setup can affect predictions
4. **Use Multiple Targets**: Don't just focus on win predictions
5. **Combine with Expert Analysis**: Use predictions alongside golf expertise
