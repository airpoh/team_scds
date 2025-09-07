import streamlit as st, pandas as pd
import plotly.express as px

def render(cats_out: pd.DataFrame):
    st.subheader("🧩 Categories & Sub-Topics")
    min_score = st.slider("Min category confidence", 0.0, 1.0, 0.0, 0.05)
    view = cats_out[cats_out["category_score"] >= min_score]
    st.dataframe(view, width="stretch")  # was use_container_width=True

    st.divider()
    # pandas < 2.1 compatible way:
    cc = (view["category"]
          .value_counts()
          .rename_axis("category")
          .reset_index(name="count"))
    st.plotly_chart(px.bar(cc, x="category", y="count", title="Comments by Category"),
                    width="stretch")      # was use_container_width=True

    chosen = st.selectbox("Filter subtopic by category",
                          ["(all)"] + sorted(view["category"].unique().tolist()))
    vv = view if chosen == "(all)" else view[view["category"] == chosen]
    sc = (vv["subtopic"]
          .value_counts()
          .rename_axis("subtopic")
          .reset_index(name="count"))
    st.plotly_chart(px.bar(sc, x="subtopic", y="count", title="Top Subtopics"),
                    width="stretch")      # was use_container_width=True
