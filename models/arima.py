import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pmdarima as pm
from sklearn.metrics import mean_absolute_error 

class Arima:
    def __init__(self, df_path: str, pred_len: int=7):
        self.df_path = df_path
        self.pred_len = pred_len
        self.data = pd.read_csv(df_path)
        self.data['traffic'] = pd.to_datetime(self.data['traffic'])
        self.data.set_index('date', inplace=True)
        self.model = None
        self.results = None

    def train(self):

        if len(self.data) < self.pred_len:
            raise ValueError("Not enough data to train the model. Ensure that the dataset has more rows than the prediction length.")
        ts = self.data['traffic'][:-self.pred_len]
        self.model = pm.auto_arima(
            ts, 
            seasonal=True, 
            m=7,
            n_jobs=2,
            trace=True, 
            suppress_warnings=True
            )
        
    def predict(self):
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        self.results = self.model.predict(n_periods=self.pred_len)
        return self.results

    def calculate_mae(self):
        if self.results is None:
            raise ValueError("No predictions available. Call predict() first.")
        
        ts = self.data['traffic'][-self.pred_len:]
        mae = mean_absolute_error(ts, self.results)
        return mae
if __name__ == "__main__":
    # Example usage
    mod = Arima(df_path="../data/vtv.vn.csv", pred_len=7)
    mod.train()
    predictions = mod.predict()
    print(predictions)
    mae = mod.calculate_mae()
    print(f"Mean Absolute Error: {mae}")