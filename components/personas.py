import streamlit as st, pandas as pd
import plotly.express as px

def render(personas_df: pd.DataFrame):
    st.subheader(" Personas")
    st.dataframe(personas_df, width="stretch")  # was use_container_width=True
    pc = (personas_df["persona_label"]
          .value_counts()
          .rename_axis("persona")
          .reset_index(name="users"))
    st.plotly_chart(px.bar(pc, x="persona", y="users", title="Users per Persona"),
                    width="stretch")      # was use_container_width=True