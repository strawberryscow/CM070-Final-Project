from pathlib import Path
import pickle
import sys

#add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root /'src'))

from preprocessing import FinancialPreprocessor
from models import FinancialMLModels

def train_all_assets(tune_hyperparams=True, prediction_horizon=3, use_threshold_tuning=True):
    
    assets = {
        "AAPL": "data/raw/AAPL.csv",
        "MSFT": "data/raw/MSFT.csv",
        "GOOGL": "data/raw/GOOGL.csv",
        "AMZN": "data/raw/AMZN.csv",
        "TSLA": "data/raw/TSLA.csv",
        "NVDA": "data/raw/NVDA.csv",
        "META": "data/raw/META.csv",
        "BTC": "data/raw/BTC.csv",
        "ETH": "data/raw/ETH.csv"
    }

    all_results = {}

    print("="*80)
    print("Financial Advisor Bot - Model Training")
    print("="*80)
    print(f"Assets to process: {len(assets)}")
    print(f"Prediction horizon: {prediction_horizon} day")
    print(f"Hyperparameter tuning: {'ON' if tune_hyperparams else 'OFF'}")
    print(f"Threshold tuning: {'ON' if use_threshold_tuning else 'OFF'}")
    print("="*80)

    for idx, (asset_name, filepath) in enumerate(assets.items(), start=1):
        print(f"\n{"="*80}")
        print(f"[{idx}/{len(assets)}] Processing: {asset_name}")
        print(f"{'='*80}")
        
        try:
            #Data Preprocessing
            preprocessor = FinancialPreprocessor()
            data = preprocessor.process_asset(filepath)

            # Train models
            ml_models = FinancialMLModels(random_state=42)

            #Logistic Regression
            lr_model = ml_models.train_logistic_regression(
                data["X_train"], 
                data["y_train"],
                tune_hyperparams=tune_hyperparams
            )

            ml_models.evaluate_model(
                lr_model, 
                data["X_train"], 
                data["y_train"], 
                data["X_test"],
                data["y_test"],
                "Logistic Regression",
                use_threshold_tuning=use_threshold_tuning
            )

            #xboost
            xgb_model = ml_models.train_xgboost(
                data["X_train"], 
                data["y_train"],
                tune_hyperparams=tune_hyperparams
            )

            ml_models.evaluate_model(
                xgb_model, 
                data["X_train"], 
                data["y_train"], 
                data["X_test"],
                data["y_test"],
                "XGBoost",
                use_threshold_tuning=use_threshold_tuning
            )

            #comapre 
            ml_models.compare_models()

             #Plot
            save_dir = f"results/figures/{asset_name}"
            ml_models.plot_results(save_dir)


             #Save models and results
            all_results[asset_name] = {
                "models": ml_models.models,
                "results": ml_models.results,
                "optimal_thresholds": getattr(ml_models, 'optimal_thresholds', {}),
                "feature_names": data["feature_names"]
            }

            print(f"\n {asset_name} processing complete.")
        
        except Exception as e:
            print(f"\n Error processing {asset_name}: {e}")
            continue

    print("\n" + "="*80)
    print("Saving all assets results.")
    print("="*80) 
    
    if prediction_horizon == 1:
        consolidates_path = "results/all_assets_results.pkl"
    else:
        consolidates_path = f"results/all_assets_results_{prediction_horizon}d.pkl"
    
    Path(consolidates_path).parent.mkdir(parents=True, exist_ok=True)

    with open(consolidates_path, "wb") as f:
        pickle.dump(all_results, f)

    print(f"All assets results saved and processed")
    print(f" Consolidated results path: {consolidates_path}")
    print(f" Figures: results/figures/[ASSET]/")

    print(f"\n{'='*80}")
    print(f"Summary ({prediction_horizon} day prediction)")
    print(f"{'='*80}")
    print(f"\n{'Asset':<10} {'LR F1':<10} {'XGB F1':<10} {'Best Model':<15}")
    print("-"*80)

    for asset_name, asset_data in all_results.items():
        results = asset_data.get("results", {})
        lr_f1 = results.get("Logistic Regression", {}).get("test_f1", 0)
        xgb_f1 = results.get("XGBoost", {}).get("test_f1", 0)
        best = "XGBoost" if xgb_f1 > lr_f1 else "Logistic Regression"
        
        print(f"{asset_name:<10} {lr_f1:<10.4f} {xgb_f1:<10.4f} {best:<15}")

    print(f"\n{'='*80}")
    print(f"Training complete for all assets ({prediction_horizon} day prediction).")
    print(f"{'='*80}")

    avg_lr_f1 = np.mean([results["results"]["Logistic Regression"]["test_f1"] 
                        for results in all_results.values()])
    avg_xgb_f1 = np.mean([results["results"]["XGBoost"]["test_f1"] 
                        for results in all_results.values()])

    print(f"\nAverage F1 Scores:")
    print(f"Logistic Regression: {avg_lr_f1:.4f}")
    print(f"XGBoost: {avg_xgb_f1:.4f}")
    print(f" Overall Average: {(avg_lr_f1 + avg_xgb_f1) / 2:.4f}")

if __name__ == "__main__":
    import numpy as np

    train_all_assets(
        tune_hyperparams=True,
        prediction_horizon=3,
        use_threshold_tuning=True
    )



    