import streamlit as st
import streamlit.components.v1 as stc 
import time
from random import random
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
        a_conversion = st.slider('True Conversion Rate',0., 1., 0.41)

    with col2:
        st.header("📺 Design B")
        b_conversion = st.slider('True Conversion Rate',0., 1., 0.48)

    run = st.checkbox('Run')

    # Setup empty
    dx = pd.DataFrame([[a_conversion, b_conversion] for x in range(10)], columns=["squared", "cubed"])
    dx.index.name = "x"
    data = dx.reset_index().melt('x')
    lines = alt.Chart(data).mark_line().encode(
        x='x',
        y='value',
        color='variable')
    line_plot = st.altair_chart(lines, use_container_width=True)


    n_samples = st.number_input('Samples', min_value=0, max_value=1000, value=100)
    n_experiments = st.number_input('Experiments (how many times to run?)', min_value=0, max_value=1000, value=10)

    res_a, res_b = [], []

    if run: 
        for i in range(n_experiments):
            A = [random() for x in range(n_samples)]
            B = [random() for x in range(n_samples)]
            df = pd.DataFrame()
            df['A'] = pd.Series(A)
            df['A_conv'] = (df['A']>(1-a_conversion)).astype(int)
            df['B'] = pd.Series(B)
            df['B_conv'] = (df['B']>(1-b_conversion)).astype(int)
            res_a.append(df.A_conv.mean())
            res_b.append(df.B_conv.mean())

            dx = pd.DataFrame()
            dx[f'A_res'] = pd.Series(res_a)
            dx[f'B_res'] = pd.Series(res_b)

            dx.index.name = "x"
            dx = dx.reset_index().melt('x') # nice shape for altair
            lines = alt.Chart(dx).mark_line().encode(
                x='x',
                y='value',
                color='variable')
            
            line_plot.altair_chart(lines, use_container_width=True)
            wait_period = 2 / n_experiments
            time.sleep(wait_period) 

    

if __name__ == '__main__':
    main()
