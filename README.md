# Content-intelligence-mlops

This project implements a multi-agent machine learning system that predicts content traffic, detects performance drift, and retrains models automatically when traffic patterns change.

This project uses the *Daily Website Visitors dataset by Bob Nau* , a publicly available real-world website analytics dataset. It contains several years of daily website activity, including page loads, unique visits, first-time visits, and returning visits. The raw daily data was aggregated into weekly time windows and transformed into machine-learning features such as impressions, clicks, traffic, and engagement, allowing the drift-aware ML pipeline to be tested on realistic user behavior and traffic fluctuations rather than synthetic data.

## Features
- Real-world website traffic ingestion
- Feature engineering (CTR, trends, engagement)
- Traffic prediction model
- Drift detection based on traffic and error
- Automated retraining
- Model versioning

## Tech Stack
- Python
- Pandas
- Scikit-learn

## How to Run

pip install -r requirements.txt  
python pipeline.py
