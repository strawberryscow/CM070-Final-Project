import pandas as pd
import numpy as np
import ta   #technical analysis lib
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

class FinancialPreprocessor:
    def __init__(self, prediction_horizon=3):
        self.feature_names = None
        self.scaler = StandardScaler()
        self.prediction_horizon = prediction_horizon

    def load_data(self, file_path):
        #load stock/crypto data from csv
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        })
        expected_columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']
        df = df[expected_columns]

        df = df[pd.to_numeric(df["Close"], errors="coerce").notna()]
        df = df.sort_values(by='Date').reset_index(drop=True)
        return df  
    
    #Feature engineering
    
    def engineer_features(self, df):
        df = df.copy()

        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        print("Engineering features...")

        # Technical indicators
        #measures overbought and oversold
        df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

        #EMAs
        df["EMA_20"] = close.ewm(span=20, adjust=False).mean()
        df["EMA_50"] = close.ewm(span=50, adjust=False).mean()
        df["EMA_200"] = close.ewm(span=200, adjust=False).mean()
        #ema difference captures momentum direction
        df["EMA_diff"] = df["EMA_20"] - df["EMA_50"]

        # MACD
        macd = ta.trend.MACD(close)
        df["MACD"] = macd.macd()
        df["MACD_signal"] = macd.macd_signal()
        df["MACD_diff"] = macd.macd_diff()

        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close)
        df["BB_high"] = bollinger.bollinger_hband()
        df["BB_low"] = bollinger.bollinger_lband()
        df["BB_mid"] = bollinger.bollinger_mavg()
        df["BB_width"] = (df["BB_high"] - df["BB_low"]) / df["BB_mid"]

        #ATR
        df["ATR"] = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()

        #OBV
        df["OBV"] = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        #Returns / momentum
        df["Returns"] = close.pct_change()
        df["Return_5d"] = close.pct_change(periods=5)
        df["Return_20d"] = close.pct_change(periods=20)

        #Volume features
        df["Volume_20d_MA"] = volume.rolling(window=20).mean()
        df["Volume_ratio"] = volume / df["Volume_20d_MA"]

        df["RSI_lag1"] = df["RSI"].shift(1)
        df["RSI_lag2"] = df["RSI"].shift(2)
        df["RSI_lag5"] = df["RSI"].shift(5)

        df["MACD_lag1"] = df["MACD"].shift(1)
        df["MACD_diff_lag1"] = df["MACD_diff"].shift(1)

        df["Volume_ratio_lag1"] = df["Volume_ratio"].shift(1)
        df["Volume_ratio_lag2"] = df["Volume_ratio"].shift(2)

        df["Above_EMA20"] = (close > df["EMA_20"]).astype(int)
        df["Above_EMA50"] = (close > df["EMA_50"]).astype(int)
        df["Above_EMA200"] = (close > df["EMA_200"]).astype(int)

        df["Momentum_10d"] = df["Returns"].rolling(10).std()
        df["Momentum_20d"] = df["Returns"].rolling(20).std()

        df["Dist_from_20d_high"] = close / close.rolling(20).max() - 1
        df["Dist_from_20d_low"] = close / close.rolling(20).min() - 1

        

        print(f"Features engineered: {df.shape[1] - 6}")

        return df
    
    #target variable creation
    def create_target(self, df):
        df = df.copy()

        df['Target'] = (df["Close"].shift(-self.prediction_horizon) > df["Close"]).astype(int)
        df = df.iloc[:-self.prediction_horizon]

        n_up = (df["Target"] == 1).sum()
        n_down = (df["Target"] == 0).sum()
        total = len(df)

        print(f"\nTarget distribution:")
        print(f" UP(1): {n_up} ({(n_up/total)*100:.1f}%)")
        print(f" DOWN(0): {n_down} ({(n_down/total)*100:.1f}%)")
        print(f" Imbalance ratio: {max(n_up, n_down) / min(n_up, n_down):.2f}:1")
        return df
    

    def prepare_features(self, df):

        exclude_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Target"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        df_clean = df.dropna().copy()

        print("\nData after cleaning:")
        print(f" Rows: {len(df)} -> {len(df_clean)}")
        print(f" Features used: {len(feature_cols)}")

        X = df_clean[feature_cols]
        y = df_clean["Target"]
        dates = df_clean["Date"]

        self.feature_names = feature_cols
        return X, y, dates
    
    #time aware train test split
    def split_data(self, X, y, dates, test_size=0.2):
        split_index = int(len(X) * (1 - test_size))

        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]
        train_dates = dates.iloc[:split_index]
        test_dates = dates.iloc[split_index:]

        print(f"\nTrain/Test split:")
        print(f" Train: {len(X_train)} samples")
        print(f" Test: {len(X_test)} samples")

        return X_train, y_train, train_dates, X_test, y_test, test_dates
    
    #feature scaling
    
    def scale_features(self, X_train, X_test):
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names, index=X_test.index)

        print("Feature scaling applied.")
        return X_train_scaled, X_test_scaled
    
    #full preprocessing pipeline

    def process_asset(self, filepath, test_size=0.2):

        print(f"\nProcessing {filepath}")

        df = self.load_data(filepath)
        df = self.engineer_features(df)
        df = self.create_target(df)
        
        X, y, dates = self.prepare_features(df)
        X_train, y_train, train_dates, X_test, y_test, test_dates = self.split_data(X, y, dates, test_size)
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        return {
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "train_dates": train_dates,
            "test_dates": test_dates,
            "feature_names": self.feature_names,
            "scaler": self.scaler
        }
