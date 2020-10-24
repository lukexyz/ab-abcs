import streamlit as st
import streamlit.components.v1 as stc 


def main():
    st.beta_set_page_config(
        page_title="AB Testing",       # String or None. Strings get appended with "• Streamlit". 
        page_icon="🎲",                # String, anything supported by st.image, or None.
        layout="centered",             # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto")  # Can be "auto", "expanded", "collapsed"

    # ================== Using st.beta_columns ================== #
    col1, col2 = st.beta_columns([1, 1]) # first column 4x the size of second

    with col2: 
        st.header("📺 Video Stream")

    with col1:
        st.header("📺 Video Stream")

    st.text("Below columns")

if __name__ == '__main__':
    main()
