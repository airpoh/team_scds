import streamlit as st
def init_state():
    if "tau" not in st.session_state:
        st.session_state.tau = 0.60
    if "spam_max" not in st.session_state:
        st.session_state.spam_max = 1.00
