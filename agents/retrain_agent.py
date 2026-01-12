import os
import json
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

class RetrainAgent:
    def __init__(self, models_dir="models", metrics_path="models/metrics.json"):
        self.models_dir = models_dir
        self.metrics_path = metrics_path

    def _get_next_version(self):
        existing = [f for f in os.listdir(self.models_dir) if f.startswith("model_v")]
        versions = [int(f.split("_v")[1].split(".")[0]) for f in existing]
        return max(versions) + 1 if versions else 1

    def retrain(self, features):
        X = features.drop("traffic", axis=1)
        y = features["traffic"]

        # Train new model
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        new_error = mean_absolute_error(y, y_pred)

        # Load previous metrics
        if os.path.exists(self.metrics_path):
            with open(self.metrics_path, "r") as f:
                metrics = json.load(f)
                old_error = metrics.get("active_error", float("inf"))
                active_model = metrics.get("active_model")
        else:
            metrics = {}
            old_error = float("inf")
            active_model = None

        # Compare and decide
        if new_error < old_error:
            version = self._get_next_version()
            model_name = f"model_v{version}.pkl"
            model_path = os.path.join(self.models_dir, model_name)

            joblib.dump(model, model_path)

            metrics["active_model"] = model_name
            metrics["active_error"] = new_error

            with open(self.metrics_path, "w") as f:
                json.dump(metrics, f, indent=4)

            return f"New model promoted: {model_name}"

        return "Old model kept"
