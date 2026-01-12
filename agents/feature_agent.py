import pandas as pd

class FeatureAgent:
    def __init__(self, output_path="data/processed/features.csv"):
        self.output_path = output_path

    def create_features(self, df):
        df = df.copy()

        # CTR = clicks / impressions
        df["ctr"] = df["clicks"] / df["impressions"]

        # Traffic trend per page
        df["traffic_trend"] = df.groupby("page_id")["traffic"].diff()

        # Fill missing trend values with 0
        df["traffic_trend"] = df["traffic_trend"].fillna(0)

        # Encode topic (simple label encoding)
        df["topic_code"] = df["topic"].astype("category").cat.codes

        # Select features for ML
        features = df[[
            "impressions",
            "clicks",
            "ctr",
            "avg_time_on_page",
            "traffic_trend",
            "topic_code",
            "traffic"   # target
        ]]

        # Save processed features
        features.to_csv(self.output_path, index=False)

        return features
