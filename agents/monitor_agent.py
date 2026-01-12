import numpy as np

class MonitorAgent:
    def __init__(self, traffic_drop_threshold=-0.15, error_increase_threshold=0.10):
        self.traffic_drop_threshold = traffic_drop_threshold
        self.error_increase_threshold = error_increase_threshold
        self.last_error = None

    def check_drift(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate average traffic
        current_avg = np.mean(y_true)
           
        half = len(y_true) // 2
        previous_avg = np.mean(y_true[:half])

        traffic_change = (current_avg - previous_avg) / previous_avg

        # MAE calculation
        error = np.mean(np.abs(y_true - y_pred))

        drift = False

        # Rule 1: Traffic drop
        if traffic_change < self.traffic_drop_threshold:
            drift = True

        # Rule 2: Error increase
        if self.last_error is not None:
            if error > self.last_error * (1 + self.error_increase_threshold):
                drift = True

        # Store for next pipeline run
        self.last_error = error

        return drift, traffic_change, error
