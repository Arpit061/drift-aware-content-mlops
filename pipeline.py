from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.model_agent import ModelAgent
from agents.monitor_agent import MonitorAgent
from agents.retrain_agent import RetrainAgent

def main():
    data_agent = DataAgent()
    df = data_agent.load_data()

    feature_agent = FeatureAgent()
    features = feature_agent.create_features(df)

    model_agent = ModelAgent()

    try:
        y_pred, y_true, error = model_agent.predict(features)
    except:
        print("No model found. Training first model...")
        model_agent.train(features)
        y_pred, y_true, error = model_agent.predict(features)

    monitor = MonitorAgent()
    drift, traffic_change, error = monitor.check_drift(y_true.values, y_pred)

    print("Traffic change:", traffic_change)
    print("Error:", error)
    print("Drift:", drift)

    if drift:
        retrain_agent = RetrainAgent()
        print(retrain_agent.retrain(features))

if __name__ == "__main__":
    main()
