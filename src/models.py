import numpy as np
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, f1_score)

import xgboost as xgb
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class FinancialMLModels:
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.optimal_thresholds = {}
        
        #logistic regression model
    def train_logistic_regression(self, X_train, y_train, tune_hyperparams=False):

        print("\n" + "="*60)
        print("Training Logistic Regression Model...")
        print("="*60)

        if tune_hyperparams:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'class_weight': ['balanced'],
                'max_iter': [1000],
                'solver': ['liblinear']
            }

            tscv = TimeSeriesSplit(n_splits=5)

            lr = LogisticRegression(random_state=self.random_state)
            grid_search = GridSearchCV(
                lr, param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
            
            print("Running Grid Search for Hyperparameter Tuning...")
            grid_search.fit(X_train, y_train)

            print(f"Best Hyperparameters: {grid_search.best_params_}")
            print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

            model = grid_search.best_estimator_

        else:
            model = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=self.random_state)
            model.fit(X_train, y_train)

        self.models['Logistic Regression'] = model
        return model
    
    def train_xgboost(self, X_train, y_train, tune_hyperparams=False):

        print("\n" + "="*60)
        print("Training XGBoost Model...")
        print("="*60)

        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos

        print(f"Class imbalance handling: scale_pos_weight={scale_pos_weight:.2f}")

        if tune_hyperparams:
            param_grid = {
                'max_depth': [3, 5],
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'scale_pos_weight': [scale_pos_weight],
                'min_child_weight': [5, 10],
                'subsample': [0.6, 0.8],
                'colsample_bytree': [0.6, 0.8],
                'reg_alpha': [0.1, 0.5],
                'reg_lambda': [1.0, 2.0]
            }

            tscv = TimeSeriesSplit(n_splits=5)

            xgb_model = xgb.XGBClassifier(
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )

            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=tscv, scoring='f1', n_jobs=-1, verbose=1)
            
            print("Running Grid Search for Hyperparameter Tuning...")
            grid_search.fit(X_train, y_train)

            print(f"\nBest Hyperparameters:")
            for param, value in grid_search.best_params_.items():
                print(f" {param}: {value}")
            print(f"Best CV F1 Score: {grid_search.best_score_:.4f}")

            model = grid_search.best_estimator_

        else:
            model = xgb.XGBClassifier(
                max_depth=3,
                n_estimators=50,
                learning_rate=0.08,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0
            )
            model.fit(X_train, y_train)

        self.models['XGBoost'] = model
        return model
    
    def optimize_threshold(self, model, X_val, y_val, model_name):
        print(f"\nOptimizing classification threshold for {model_name}...")
        y_proba = model.predict_proba(X_val)[:, 1]
        best_threshold = 0.5
        best_f1 = 0.0

        thresholds = np.arange(0.30, 0.71, 0.02)

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        print(f"Optimal Threshold: {best_threshold:.2f} (F1 Score: {best_f1:.4f})")  

        self.optimal_thresholds[model_name] = best_threshold
        return best_threshold
    
    def evaluate_model(self, model, X_train, y_train, X_test, y_test, model_name, use_threshold_tuning=True):

        print("\n" + "="*60)
        print(f"Evaluating Model: {model_name}")
        print("="*60)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

        if use_threshold_tuning:
            tscv = TimeSeriesSplit(n_splits=5)

            for train_idx, val_idx in tscv.split(X_train):
                X_val_fold = X_train.iloc[val_idx]
                y_val_fold = y_train.iloc[val_idx]
            
            optimal_threshold = self.optimize_threshold(model, X_val_fold, y_val_fold, model_name)
        else:
            optimal_threshold = 0.5
            self.optimal_thresholds[model_name] = 0.5

        y_train_pred = (y_train_proba >= optimal_threshold).astype(int)
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

        print(f"\nProbability distribution (Test set):")
        print(f" Min: {np.min(y_test_proba):.3f}, Max:{np.max(y_test_proba):.3f}")
        print(f" Max: {np.max(y_test_proba):.3f}, Median: {np.median(y_test_proba):.3f}")
        print(f"Threshold: {optimal_threshold:.2f}")
        print(f" Prediction UP={sum(y_test_pred==1)}, DOWN={sum(y_test_pred==0)}")


        results = {
            'model_name': model_name,
            'optimal_threshold': optimal_threshold,
            'train_accuracy': accuracy_score(y_train, y_train_pred),
            'test_accuracy': accuracy_score(y_test, y_test_pred),
            'test_precision': precision_score(y_test, y_test_pred, zero_division=0),
            'test_recall': recall_score(y_test, y_test_pred),
            'test_f1': f1_score(y_test, y_test_pred, zero_division=0),
            'test_auc': roc_auc_score(y_test, y_test_proba),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'y_test_proba': y_test_proba,
            'y_test_pred': y_test_pred,
            'y_test': y_test
        }

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):

            X_tr = X_train.iloc[train_idx].values
            X_val = X_train.iloc[val_idx].values
            y_tr = y_train.iloc[train_idx].values
            y_val = y_train.iloc[val_idx].values

            if 'XGBoost' in model_name or 'xgb' in str(type(model)).lower():
                params = model.get_params()
                model_cv = xgb.XGBClassifier(**params)

            else:
                from sklearn.base import clone
                model_cv = clone(model)

            model_cv.fit(X_tr, y_tr)
            y_val_proba = model_cv.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_proba >= optimal_threshold).astype(int)
            cv_scores.append(f1_score(y_val, y_val_pred, zero_division=0))

            # X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            # y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            # cv_scores.append(f1_score(y_val, y_val_pred, zero_division=0))

        results['cv_scores'] = cv_scores
        results['cv_mean'] = np.mean(cv_scores)
        results['cv_std'] = np.std(cv_scores)

        print(f"Train Accuracy: {results['train_accuracy']:.4f}")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"\nTest Set Metrics:")
        print(f" Precision: {results['test_precision']:.4f}")
        print(f" Recall: {results['test_recall']:.4f}")
        print(f" F1 Score: {results['test_f1']:.4f}")
        print(f" AUC-ROC: {results['test_auc']:.4f}")

        print(f"\n5-Fold CV F1 Score (with threshold {optimal_threshold:.2f}):")
        for i, score in enumerate(cv_scores, 1):
            print(f" Fold {i}: {score:.4f}")
        print(f" Mean: {results['cv_mean']:.4f} +- {results['cv_std']:.4f}")

        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])

        #classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_test_pred, 
                                    target_names=['DOWN (0)', 'UP (1)'], zero_division=0))
            
        self.results[model_name] = results
        return results
        
    def compare_models(self):

            if len(self.results) < 2:
                print("At least two models are required for comparison.")
                return
            
            print("\n" + "="*60)
            print("COMPARING MODELS")
            print("="*60)

            compasrison_data = []
            for name, results in self.results.items():
                compasrison_data.append({
                    'Model': name,
                    'Threshold': f"{results.get('optimal_threshold', 0.5):.2f}",
                    'Test Accuracy': f"{results['test_accuracy']:.4f}",
                    'Test Precision': f"{results['test_precision']:.4f}",
                    'Test Recall': f"{results['test_recall']:.4f}",
                    'Test_F1 Score': f"{results['test_f1']:.4f}",
                    'Test AUC-ROC': f"{results['test_auc']:.4f}",
                    'CV Mean F1': f"{results['cv_mean']:.4f}",
                    'CV Std F1': f"{results['cv_std']:.4f}"
                })

                df_comparison = pd.DataFrame(compasrison_data)
                print("\n" + df_comparison.to_string(index=False))

                if 'Logistic Regression' in self.results and 'XGBoost' in self.results:
                    lr_scores = self.results['Logistic Regression']['cv_scores']
                    xgb_scores = self.results['XGBoost']['cv_scores']

                    t_stat, p_value = stats.ttest_rel(lr_scores, xgb_scores)

                    print(f"\n{'='*60}")
                    print("Statistical Significance Test")
                    print(f"{'='*60}")
                    print(f"T-test between Logistic Regression and XGBoost CV F1 Scores:")
                    print(f" T-statistic: {t_stat:.4f}")
                    print(f" P-value: {p_value:.4f}")

                    if p_value < 0.05:
                        print(" Result: The difference in model performance is statistically significant (p < 0.05).")
                    if t_stat < 0:
                        print(" XGBoost outperforms Logistic Regression.")
                    else:
                        print(" Logistic Regression outperforms XGBoost.")
                else:
                    print(" Result: No statistically significant difference in model performance (p >= 0.05).") 

    def plot_results(self, save_dir='results/figures'):
            import matplotlib
            matplotlib.use('Agg')  #use non-gui backend
            Path(save_dir).mkdir(parents=True, exist_ok=True)

            #model comparison bar chart
            fig, ax = plt.subplots(figsize=(10, 6))
            metrices = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            x = np.arange(len(metrices))
            width = 0.35

            lr_scores = [self.results['Logistic Regression'][metric] for metric in metrices]
            xgb_scores = [self.results['XGBoost'][metric] for metric in metrices]

            ax.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='skyblue')
            ax.bar(x + width/2, xgb_scores, width, label='XGBoost', color='salmon')

            ax.set_ylabel('Scores', fontsize=12)
            ax.set_xlabel('Model Performance Metrics', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig(f"{save_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
            print(f" Saved: {save_dir}/model_comparison.png")
            plt.close()

            # confusion_matrix

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            for idx, (name, results) in enumerate(self.results.items()):
                cm = results['confusion_matrix']
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                            xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
                axes[idx].set_title(f"{name}", fontsize=14, fontweight='bold')
                axes[idx].set_ylabel('True Label')
                axes[idx].set_xlabel('Predicted Label')

            # save and close after the loop, not inside it
            plt.tight_layout()
            plt.savefig(f"{save_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
            print(f" Saved: {save_dir}/confusion_matrices.png")
            plt.close()

            #roc curves
            fig, ax = plt.subplots(figsize=(8, 6))

            for name, results in self.results.items():
                y_test = list(self.results.values())[0]['y_test']
                fpr, tpr, _ = roc_curve(y_test, results['y_test_proba'])
                auc = results['test_auc']

                color = 'blue' if name == 'Logistic Regression' else 'orange'
                ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", color=color, linewidth=1)

            ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)', linewidth=1)
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10, loc='lower right')
            ax.grid(alpha=0.3)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])

            plt.tight_layout()
            plt.savefig(f"{save_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
            print(f" Saved: {save_dir}/roc_curves.png")
            plt.close()
        
    def save_results(self, filepath='results/model_results.pkl'):
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)

            save_data = {
                'models': self.models,
                'results': self.results
            }

            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"\nSaved model results to {filepath}")
        
if __name__ == "__main__":
    from preprocessing import FinancialPreprocessor

    preprocessor = FinancialPreprocessor()
    data = preprocessor.process_asset('data/raw/AAPL.csv')

    ml_models = FinancialMLModels(random_state=42)

    lr_model = ml_models.train_logistic_regression(
        data['X_train'], data['y_train'], tune_hyperparams=True)
    
    ml_models.evaluate_model(
        lr_model, data['X_train'], data['y_train'],
        data['X_test'], data['y_test'], 'Logistic Regression')
    
    xgb_model = ml_models.train_xgboost(
        data['X_train'], data['y_train'], tune_hyperparams=True
    )

    #comare models
    ml_models.comapre_models()

    #plot results
    ml_models.plot_results()

    #save results
    ml_models.save_results()

    print("\n" + "="*60)
    print("All tasks completed.")
    print("="*60)

