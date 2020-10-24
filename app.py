import streamlit as st
import streamlit.components.v1 as stc 
import time, random
import numpy as np
import pandas as pd
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue


@st.cache(persist=True) # persist cache on disk
def create_df(size):
    df = pd.DataFrame(np.random.randn(size[0], size[1]),
        columns=(f'col{i}' for i in range(size[1])))
    return df


def main():
    st.beta_set_page_config(
        page_title="AB Testing",       # String or None. Strings get appended with "• Streamlit". 
        page_icon="🎲",                # String, anything supported by st.image, or None.
        layout="centered",             # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto")  # Can be "auto", "expanded", "collapsed"

    nav = 1
    part1, part2, part3 = st.beta_columns([1, 1, 1])
    with part1:  # excuse hacky whitespace (alt+255) alignment 
        if st.button('💠 Part I: Probability          '):      nav = 1
    with part2:
        if st.button('💠 Part II: Error                    '): nav = 2
    with part3:
        if st.button('💠 Part III: P-values             '):    nav = 2
    st.markdown('---')

    # ================== Using st.beta_columns ================== #
    col1, col2 = st.beta_columns([1, 1]) # first column 4x the size of second

    with col1: 
        st.header("📺 Design A")
        x = st.slider('True conversion rate',0., 1., 0.41)
        x

    with col2:
        st.header("📺 Design B")
        y = st.slider('True conversion rate',0., 1., 0.48)
        y

    st.text("Below columns")

    # ========================== altair ============================== #
    df1 = create_df(size=(1,5))
    my_table = st.table(df1)

    if st.button('add rows'):
        df2 = pd.DataFrame(np.random.randn(3, 5),
                    columns=(f'col{i}' for i in range(5)))
        my_table.add_rows(df2)


    st.line_chart(df1)


    

if __name__ == '__main__':
    main()
