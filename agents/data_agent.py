##Only clean, ordered data enters the ML pipeline

import pandas as pd

class DataAgent:
    def __init__(self, data_path="data/raw/content_traffic.csv"):
        self.data_path = data_path

    def load_data(self):
        # Load raw CSV
        df = pd.read_csv(self.data_path)

        # Basic cleaning
        df = df.dropna()
        df = df.sort_values(["page_id", "week"])

        return df
