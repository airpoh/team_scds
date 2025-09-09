# soe_model.py
import pandas as pd

def load_and_preprocess(file_path="combined_dataset_with_spam.csv"):
    """
    Load dataset and preprocess engagement values.
    """
    df = pd.read_csv(file_path)
    df["publishedAt"] = pd.to_datetime(df["publishedAt"], errors="coerce")
    df["year_week"] = df["publishedAt"].dt.strftime("%Y-%U")
    df["engagement"] = df["likeCount"].fillna(0) + 1
    return df

def compute_soe(df):
    """
    Compute Share of Engagement (SoE) on weekly basis.
    """
    weekly = (
        df.groupby(["videoId", "year_week"], as_index=False)
          .agg(total_engagement=("engagement", "sum"))
    )
    weekly["SoE"] = weekly["total_engagement"] / weekly["total_engagement"].max()
    return weekly
