# US Open Prediction System

A comprehensive machine learning system for predicting US Open golf tournament outcomes using DataGolf API data.

## Overview

This system implements a data-driven approach to predict the 2025 US Open by:

1. **Historical Data Collection**: Gathering tournament results, player statistics, and performance metrics
2. **Feature Engineering**: Creating predictive features from raw golf data
3. **Model Training**: Training ML models on historical tournament outcomes
4. **Prediction**: Applying trained models to predict 2025 US Open field performance

## Project Structure

```
├── src/
│   ├── data_collection/     # DataGolf API data collection
│   ├── preprocessing/       # Data cleaning and feature engineering
│   ├── modeling/           # Machine learning models
│   └── prediction/         # 2025 US Open predictions
├── data/
│   ├── raw/               # Raw API data
│   ├── processed/         # Cleaned and engineered data
│   └── predictions/       # Model outputs
├── notebooks/             # Jupyter notebooks for analysis
└── config/               # Configuration files
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your DataGolf API key:
```bash
cp .env.example .env
# Edit .env with your API key
```

3. Run the data collection:
```bash
python src/data_collection/collect_historical_data.py
```

## Usage

See the notebooks in the `notebooks/` directory for detailed analysis and model training examples.

## Data Sources

- **DataGolf API**: Historical tournament data, player statistics, strokes-gained metrics
- **Tournament Results**: Major championship outcomes and field compositions
- **Player Skills**: Current skill ratings and rankings
