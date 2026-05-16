import xgboost as xgb

class XGBoostModel:
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42
        }
        params.update(kwargs)
        self.model = xgb.XGBRegressor(**params)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
