# components/influencers.py
import streamlit as st
import pandas as pd
import plotly.express as px

def render(influencers_df: pd.DataFrame,
           communities_df: pd.DataFrame | None = None):
    st.subheader("📣 Influencers & Communities (PageRank • Betweenness • Louvain)")

    if influencers_df is None or len(influencers_df) == 0:
        st.info("No influencer data available.")
        return

    # ---- Top influencers table ----
    top_n = st.slider("Top N influencers", 10, 200, 50, 10)
    cols = [
        "user_id", "rank_score", "pagerank", "betweenness",
        "mentions_received", "mentions_made", "comments_count",
        "followers", "verified", "tier", "community_id",
    ]
    cols = [c for c in cols if c in influencers_df.columns]
    st.dataframe(influencers_df[cols].head(top_n), use_container_width=True)

    # ---- Communities bubble chart ----
    if communities_df is None or len(communities_df) == 0:
        return

    st.divider()
    st.markdown("**Community Sizes & Activity**")

    bubble = communities_df.copy()

    # Flexible rename → normalize to Community / Users / Comments
    rename_map = {}
    if "users" in bubble.columns:
        rename_map["users"] = "Users"
    elif "unique_users" in bubble.columns:
        rename_map["unique_users"] = "Users"
    if "total_comments" in bubble.columns:
        rename_map["total_comments"] = "Comments"
    if "community_id" in bubble.columns:
        rename_map["community_id"] = "Community"
    bubble = bubble.rename(columns=rename_map)

    # Ensure required columns exist as Series (not scalars)
    if "Community" not in bubble.columns:
        bubble["Community"] = bubble.index.astype(str)
    if "Users" not in bubble.columns:
        bubble["Users"] = 0
    if "Comments" not in bubble.columns:
        bubble["Comments"] = 0

    # Coerce numerics safely and clip to renderable values
    bubble["Users"] = (
        pd.to_numeric(bubble["Users"], errors="coerce")
          .fillna(0).astype(int).clip(lower=1)
    )
    bubble["Comments"] = (
        pd.to_numeric(bubble["Comments"], errors="coerce")
          .fillna(0).astype(int).clip(lower=1)
    )

    # If everything is tiny/zero, show a hint instead of a blank chart
    if (bubble["Users"] <= 1).all() and (bubble["Comments"] <= 1).all():
        st.info("Communities were detected but have no measurable activity yet (all zeros). "
                "Check that user_id exists and comments_count/total_comments were computed.")
        with st.expander("Debug view"):
            show_cols = [c for c in ["Community","Users","Comments","top_users",
                                     "sentiment_health","avg_quality"] if c in bubble.columns]
            st.dataframe(bubble[show_cols].head(20), use_container_width=True)
        return

    # Limit bubbles for readability
    top_k = st.slider("Communities to show (by comments)", 5, 200, 30, 5)
    bubble = bubble.sort_values("Comments", ascending=False).head(top_k)

    hover_cols = [c for c in ["top_users","sentiment_health","avg_quality",
                              "positive_ratio","spam_health"] if c in bubble.columns]

    fig = px.scatter(
        bubble,
        x="Community",
        y="Users",
        size="Comments",
        color="Community",
        hover_data=hover_cols if hover_cols else None,
        title="Communities (bubble size = total comments; y = unique users)",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show community details & exemplar comments"):
        preferred = ["community_id","Community","users","Users","total_comments","Comments",
                     "sample_comments","top_users","sentiment_health","avg_quality",
                     "spam_health","positive_ratio","negative_ratio"]
        show_cols = [c for c in preferred if c in communities_df.columns]
        if not show_cols:
            show_cols = list(communities_df.columns)[:8]
        st.dataframe(communities_df[show_cols], use_container_width=True)