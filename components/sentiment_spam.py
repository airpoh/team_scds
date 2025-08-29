import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Breakdown")
        st.bar_chart(df["sentiment"].value_counts())
        with st.expander("Examples"):
            for lab in ["positive","neutral","negative"]:
                st.markdown(f"**{lab.title()}**")
                st.write(df[df["sentiment"]==lab]["comment"].head(3).tolist())

    with col2:
        st.subheader("Detected Spam (top 10)")
        rate = float((df["spamness"] > 0.6).mean()) if len(df) else 0.0
        st.metric("Spam Catch (proxy)", f"{rate:.2%}")
        st.dataframe(
            df.sort_values("spamness", ascending=False)[["comment","spamness"]].head(10),
            use_container_width=True
        )
