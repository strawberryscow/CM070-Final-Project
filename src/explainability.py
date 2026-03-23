import shap
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import ollama

#add src to  path 
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from preprocessing import FinancialPreprocessor

RESULTS_PATH = "results/all_assets_results_3d.pkl"

def get_best_model(asset_results):
    results = asset_results["results"]

    lr_auc = results.get("Logistic Regression", {}).get("test_auc", 0)
    xgb_auc = results.get("XGBoost", {}).get("test_auc", 0)
    
    if xgb_auc > lr_auc:
        return "XGBoost"
    else:
        return "Logistic Regression"
    
def generate_shap(asset_name, asset_results):
    
    print(f"\nGnerating SHAP for {asset_name}")

    filepath = f"data/raw/{asset_name}.csv"

    preprocessor = FinancialPreprocessor()
    data = preprocessor.process_asset(filepath)

    x_test = data["X_test"]
    feature_names = data["feature_names"]

    model_name = get_best_model(asset_results)
    model = asset_results["models"][model_name]

    print(f"Using model: {model_name}")

    if model_name == "XGBoost":
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, x_test)

    shap_values = explainer(x_test)

    save_dir = f"results/figures/{asset_name}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.summary_plot(
        shap_values, 
        x_test, 
        feature_names=feature_names, 
        show=False
    )

    path = f"{save_dir}/shap_summary.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"saved: {path}")

def main():
    if not Path(RESULTS_PATH).exists():
        print("Error: run  training first")
        return
    
    with open(RESULTS_PATH, "rb") as f:
        all_results = pickle.load(f)

    print(f"Loaded results for {len(all_results)} assets")

    for asset_name, asset_results in all_results.items():
        try:
            generate_shap(asset_name, asset_results)
        except Exception as e:
            print(f"Error generating SHAP for {asset_name}: {e}")

if __name__ == "__main__":
    main()