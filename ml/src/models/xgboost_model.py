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
        
    def fit(self, X, y, eval_set=None):
        if eval_set:
            self.model.set_params(early_stopping_rounds=20)
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)
