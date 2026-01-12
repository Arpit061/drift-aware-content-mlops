import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class ModelAgent:
    def __init__(self, model_path="models/model_v1.pkl"):
        self.model_path = model_path

    def train(self, features):
        X = features.drop("traffic", axis=1)
        y = features["traffic"]

        model = LinearRegression()
        model.fit(X, y)

        joblib.dump(model, self.model_path)

        return model

    def predict(self, features):
        X = features.drop("traffic", axis=1)
        y_true = features["traffic"]

        model = joblib.load(self.model_path)
        y_pred = model.predict(X)

        error = mean_absolute_error(y_true, y_pred)

        return y_pred, y_true, error
