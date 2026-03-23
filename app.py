from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
import shap
from pathlib import Path
import sys

#flask web app for fa bot 

#add src to Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.preprocessing import FinancialPreprocessor
from src.genetic_algorithm import Chromosome, GAResult


app = Flask(__name__)

#Loading models and GA results
MODELS_PATH = "results/all_assets_results_3d.pkl"
GA_PATH = "results/ga_results.pkl"

with open(MODELS_PATH, 'rb') as f:
    all_models = pickle.load(f)  #assets:{models, results, optimal_thresholds, feature_names}

with open(GA_PATH, 'rb') as f:
    ga_results = pickle.load(f) #asset: GAResult

#t supported assets - used for input validation for all endpts
ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BTC", "ETH"]

def get_best_model(asset_name):
    #get best performing model for assset
    asset_results = all_models[asset_name]
    results = asset_results["results"]

    lr_auc = results.get("Logistic Regression", {}).get("test_auc", 0)
    xgb_auc = results.get("XGBoost", {}).get("test_auc", 0)

    if xgb_auc > lr_auc:
        return "XGBoost", asset_results["models"]["XGBoost"]
    else:
        return "Logistic Regression", asset_results["models"]["Logistic Regression"]
    
def get_latest_features(asset_name):
    #get the most recent features for prediction
    filepath = f"data/raw/{asset_name}.csv"
    preprocessor = FinancialPreprocessor(prediction_horizon=3)
    data = preprocessor.process_asset(filepath)

    #get the last row of test data(most recent)
    X_test = data["X_test"]
    feature_names = data["feature_names"]
#returns a dataframe for predict_proba
    return X_test.iloc[-1:], feature_names

def generate_shap_explanation(asset_name, model_name, model, features, feature_names):
    #generate shap explainability for a prediction
    #returns top 3 features with thier shap values 
    try:
        if "XBGoost" in model_name or "xgb" in str(type(model)).lower():  #select the appropriate shap explainer
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, features) 
        
        shap_values = explainer(features)

#extract shap value array for positive class
        if hasattr(shap_values, 'values'):    
            if len(shap_values.values.shape) > 2:
                features_importance = shap_values.values[0, :, 1]
            else:
                features_importance = shap_values.values[0]
        else:
            features_importance = shap_values[0]
        
        top_indices = np.argsort(np.abs(features_importance))[-3:][::-1]

        top_features = []
        for idx in top_indices:
            top_features.append({
                'name' : feature_names[idx],  
                'value': float(features.iloc[0, idx]),  #actual feature value 
                'shap_value': float(features_importance[idx])  #postitve puse=hes UP, negative = pushed down
            })
        return top_features
    
    except Exception as e:
        print(f"SHAP explanation error for {asset_name}: {e}")
#returns a safe fallback so the dashboard can still redner 
        return [
            {'name': 'RSI', 'value':0.0, 'shap_value':0.0},
            {'name': 'EMA_diff', 'value':0.0, 'shap_value':0.0},
            {'name': 'MACD', 'value':0.0, 'shap_value':0.0}
        ]

def get_ga_strategy(asset_name):
    #get optimized ga trading strategy for asset

    try:
        ga_result = ga_results[asset_name]
        best = ga_result.best_chromosome  #best evolved chromosome 

        return {
            'entry_threshold': float(best.entry_threshold),
            'stop_loss': float(best.stop_loss_pct) * 100,  #convert to %
            'take_profit': float(best.take_profit_pct) * 100,
            'holding_days': int(best.holding_days),
            'sharpe_ratio': float(best.fitness),  #fintess = sharpe ratio
            'win_rate': float(best.win_rate) * 100,
            'total_return': float(best.total_return) * 100,
            'buyhold_sharpe': float(ga_result.benchmark_sharpe),
            'buyhold_return': float(ga_result.benchmark_return) * 100,
        }
    except Exception as e:
        print(f"GA strategy error for {asset_name}: {e}")
        #return safe deafult values so the dashboard dosent crash 
        return {
            'entry_threshold': 0.5,
            'stop_loss': 5.0,
            'take_profit': 10.0,
            'holding_days': 5,
            'shapre_ratio': 0.0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'buyhold_shapre': 0.0,
            'buyhold_return': 0.0
        }
    
#routes 

#serves as the main dashboard HTML page. 


@app.route('/')
def index():
    "main dashboard page"
    return render_template('index.html', assets = ASSETS)

@app.route('/predict/<asset_name>')
def predict(asset_name):
    if asset_name not in ASSETS:
        return jsonify({'error': 'Invalid asset'}), 400
    #select best model for this asset 
    try:
        model_name, model = get_best_model(asset_name)
        #get the most recent feature row from preprocessed test set
        features, feature_names = get_latest_features(asset_name)
        #retrieve the optimal classification threshold stores during training
        model_threshold = all_models[asset_name].get('optimal_thresholds', {}).get(model_name, 0.5)
        #generate the predicted probability for thr UP class
        probability = model.predict_proba(features)[0,1]
        prediction = "UP" if probability >= model_threshold else "DOWN"
        #compute shap values for the top 3 contributing features 
        shap_features = generate_shap_explanation(asset_name, model_name, model, features, feature_names)
        #retrieve ga trategy 
        ga_strategy = get_ga_strategy(asset_name)
        display_threshold = ga_strategy['entry_threshold']
        #retrieve stores test-set performance metrics for this model
        results = all_models[asset_name]['results'][model_name]

        response = {
            'asset': asset_name,
            'prediction': prediction,
            'probability': float(probability),
            'confidence': float(abs(probability - 0.5) * 200),
            'model_used': model_name,
            'optimal_threshold': float(display_threshold),
            'model_threshold':float(model_threshold),
            'prediction_horizon': '3-day',
            'model_performance': {
                'test_f1': float(results.get('test_f1', 0.0)),
                'test_accuracy': float(results.get('test_accuracy', 0.0)),
                'test_auc': float(results.get('test_auc', 0.0))
            },
            'shap_explanations': shap_features,
            'ga_strategy': ga_strategy
        }

        return jsonify(response)
    
    except Exception as e:
        print(f"Prediction error for {asset_name}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/all_predictions')
def all_predictions():
    results = {}

    for asset in ASSETS:
        try:
            model_name, model = get_best_model(asset)

            features, _ = get_latest_features(asset)

            optimal_threshold = all_models[asset].get('optimal_thresholds', {}).get(model_name, 0.5)

            probability = model.predict_proba(features)[0,1]
            prediction = "UP" if probability >= optimal_threshold else "DOWN"

            results[asset] = {
                'prediction': prediction,
                'probability': float(probability),
                'confidence': float(abs(probability - 0.5) * 200),
                'model': model_name
            }
        except Exception as e:
            print(f"Error predicting {asset}: {e}")
            results[asset] = {
                'prediction': 'ERROR',
                'probability': 0.0,
                'confidence': 0.0,
                'model': 'N/A'
            }
    return jsonify(results)

@app.route('/asset_info/<asset_name>')
def asset_info(asset_name):
    if asset_name not in ASSETS:
        return jsonify({'error': 'Invalid asset'}), 400
    try:
        asset_data = all_models[asset_name]
        results = asset_data['results']

        info = {
            'asset': asset_name,
            'models': {},
            'best_model':None,
            'feature_count': len(asset_data.get('feature_names', []))
        }

        best_f1 = 0
        for model_name in ['Logistic Regression', 'XGBoost']:
            model_results = results[model_name]
            info['models'][model_name] = {
                'test_f1': float(model_results.get('test_f1', 0.0)),
                'test_accuracy': float(model_results.get('test_accuracy', 0.0)),
                'test_auc': float(model_results.get('test_auc', 0.0)),
                'test_precision': float(model_results.get('test_precision', 0.0)),
                'test_recall': float(model_results.get('test_recall', 0.0)),
                'cv_mean': float(model_results.get('cv_mean', 0.0)),
                'cv_std': float(model_results.get('cv_std', 0.0))
            }

            if model_results.get('test_f1', 0.0) > best_f1:
                best_f1 = model_results.get('test_f1', 0.0)
                info['best_model'] = model_name

        return jsonify(info)
    
    except Exception as e:
        print(f"Asset info error for {asset_name} : {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(all_models),
        'ga_results_loaded': len(ga_results),
        'assets': ASSETS
    })
#entry point

if __name__ == '__main__':
    print("="*60)
    print("Financial Advisor Bot - Dashboard")
    print("="*60)
    print(f"Models loaded: {len(all_models)} assets")
    print(f"GA results loaded: {len(ga_results)} assets")
    print(f"Prediction horizon: 3-day")
    print("="*60)
    print("/nStarting server at http//localhost:5000")
    print("Press Ctrl+C to stop")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)