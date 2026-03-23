# cm3070-Final-Project
Explainable AI Financial Advisor Bot - CM3070 Final Project

Project Overview
The Financial Advisor Bot is a full machine learning pipeline that:

Predicts short-term (3-day) price direction for 9 financial assets using Logistic Regression and XGBoost classifiers
Explains predictions using SHAP (SHapley Additive exPlanations) to identify the most influential technical indicators
Optimises trading strategies using a Genetic Algorithm (GA) that evolves entry thresholds, stop-loss, take-profit, and holding period parameters
Presents all results through an interactive Flask web dashboard

The system was designed as a decision-support tool for retail investors, combining predictive accuracy with transparency and practical strategy recommendations.

System Architecture
The system follows a modular, loosely-coupled pipeline. Each stage serialises its outputs as .pkl files, so computationally expensive stages (training, GA optimisation) only need to run once.

data_collection.py
       │
       ▼
  data/raw/*.csv
       │
       ▼
preprocessing.py  ──────────────────────────────────────────┐
       │                                                     │
       ▼                                                     │
train_models.py + models.py                                  │
       │                                                     │
       ▼                                                     │
results/all_assets_results_3d.pkl                            │
       │                                                     │
       ├──────────────────────┐                              │
       ▼                      ▼                              │
explainability.py     genetic_algorithm.py ◄────────────────┘
       │                      │
       ▼                      ▼
results/figures/        results/ga_results.pkl
[ASSET]/shap_summary.png
       │                      │
       └──────────┬───────────┘
                  ▼
              app.py  +  templates/index.html
                  │
                  ▼
        http://localhost:5000

Features
Machine Learning

Binary classification (UP / DOWN) over a 3-day prediction horizon
Logistic Regression with grid search hyperparameter tuning (C, solver, class weighting)
XGBoost with grid search over 8 hyperparameters (depth, estimators, learning rate, regularisation, etc.)
TimeSeriesSplit cross-validation (5 folds) to prevent data leakage
Empirical probability threshold tuning — optimises F1 on the final CV fold rather than assuming 0.50
Best model per asset selected by AUC-ROC

Feature Engineering (33 features)

RSI (14-day) + 3 lagged RSI values (1d, 2d, 5d)
EMA (20, 50, 200-day), EMA difference, above-EMA binary flags
MACD line, signal line, MACD diff + lagged MACD diff
Bollinger Bands (high, low, mid, width)
ATR (Average True Range)
OBV (On-Balance Volume), volume ratio + lags
Daily returns, 5-day and 20-day returns
10-day and 20-day rolling volatility (momentum)
Distance from 20-day rolling high/low

Genetic Algorithm

Chromosome: 4 genes — entry threshold (0.50–0.90), stop-loss (1–10%), take-profit (2–25%), holding days (1–10)
Fitness function: Annualised Sharpe ratio (minimum 3 trades required)
Operators: Tournament selection (k=5), uniform crossover (rate=0.80), Gaussian mutation (rate=0.15, σ=0.05)
Elitism: Top 3 chromosomes carried forward each generation
Configuration: 50 population, 100 generations, seed=42

Explainability

SHAP TreeExplainer for XGBoost models
SHAP LinearExplainer for Logistic Regression models
Global summary plots (beeswarm) saved per asset
Top-3 feature contributions returned live in the web dashboard

Web Dashboard (Flask)

Asset selector with live prediction fetch
Displays: prediction direction, probability, confidence score, model used, optimal threshold
SHAP feature explanation panel (top 3 contributors with values and direction)
GA strategy panel (entry threshold, stop-loss, take-profit, holding days, Sharpe, win rate, total return vs buy-and-hold)
/health, /all_predictions, /asset_info/<asset> API endpoints

1. Clone the repository
git clone https://github.com/your-username/financial-advisor-bot.git
cd financial-advisor-bot

2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

Usage
Run the stages in order. Stages 1–4 only need to be run once — their outputs are saved to disk and reused by the dashboard.
Stage 1 — Download Data
bashpython src/data_collection.py
Downloads historical OHLCV data for all 9 assets from Yahoo Finance (from 2015-01-01) and saves to data/raw/.

Stage 2 — Train Models
bashpython src/train_models.py
Runs the full preprocessing and training pipeline across all 9 assets with hyperparameter tuning and threshold optimisation enabled. Saves trained models and results to results/all_assets_results_3d.pkl.

⏱ This takes approximately 45 minutes with full grid search enabled. To run faster without tuning:
Edit train_models.py and set tune_hyperparams=False, use_threshold_tuning=False.


Stage 3 — Generate SHAP Explanations
bashpython src/explainability.py
Generates SHAP summary plots for each asset and saves them to results/figures/[ASSET]/shap_summary.png.

Stage 4 — Run Genetic Algorithm Optimisation
bashpython src/genetic_algorithm.py
Runs GA strategy optimisation for all 9 assets (50 population, 100 generations each). Saves results to results/ga_results.pkl and plots to results/figures/.

⏱ Takes approximately 10–20 minutes depending on hardware.


Stage 5 — Launch the Dashboard
bashpython app.py
Starts the Flask development server. Open your browser at:
http://localhost:5000

Pipeline Walkthrough
How a prediction is made (live, per asset)

The latest raw CSV for the selected asset is loaded and preprocessed
The best model (by AUC) is retrieved from all_assets_results_3d.pkl
The most recent row of test features is extracted
model.predict_proba() returns the probability of an upward move
The optimal threshold (stored per model per asset) determines the direction label
SHAP explains the top 3 contributing features for that prediction
GA strategy parameters are loaded from ga_results.pkl
All results are returned as JSON to the dashboard frontend

How the GA strategy is generated (offline, per asset)

Test-period closing prices and model probabilities are extracted
A population of 50 random chromosomes is initialised
Each chromosome is evaluated via run_backtest() — simulating trades on test-period data
The Sharpe ratio of the resulting capital series is assigned as fitness
Tournament selection, crossover, and mutation produce the next generation
After 100 generations, the best chromosome is saved as the recommended strategy