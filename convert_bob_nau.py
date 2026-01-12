import pandas as pd

df = pd.read_csv("data/raw/daily_website_visitors.csv")

# Convert Date
df["Date"] = pd.to_datetime(df["Date"])

# Remove commas from numeric columns
cols = ["Page.Loads", "Unique.Visits", "Returning.Visits", "First.Time.Visits"]
for c in cols:
    df[c] = df[c].astype(str).str.replace(",", "")
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Drop rows where traffic is missing
df = df.dropna(subset=["Unique.Visits"])

# Create week
df["week"] = df["Date"].dt.to_period("W").astype(str)

# Aggregate daily → weekly
weekly = df.groupby("week").agg({
    "Page.Loads": "sum",
    "Unique.Visits": "sum",
    "Returning.Visits": "sum",
    "First.Time.Visits": "sum"
}).reset_index()

# Build ML Ops columns
weekly["page_id"] = 1
weekly["topic"] = "Website"
weekly["publish_date"] = "2025-01-01"

weekly["impressions"] = weekly["Page.Loads"]
weekly["clicks"] = weekly["Returning.Visits"]
weekly["traffic"] = weekly["Unique.Visits"]
weekly["avg_time_on_page"] = weekly["First.Time.Visits"] * 0.5

final = weekly[[
    "page_id",
    "topic",
    "publish_date",
    "week",
    "impressions",
    "clicks",
    "traffic",
    "avg_time_on_page"
]]

final.to_csv("data/raw/content_traffic.csv", index=False)

print("Bob Nau real website data successfully converted.")
