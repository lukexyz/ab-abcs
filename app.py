import streamlit as st
import streamlit.components.v1 as stc 
import time
from random import random
import numpy as np
import pandas as pd
import altair as alt
from altair import Chart, X, Y, Axis, SortField, OpacityValue

# 2020-10-25 edit@ from st.annotated_text import annotated_text
from annotated_text import annotated_text  

import st_state


def main():
    st.beta_set_page_config(
        page_title="AB Testing",       # String or None. Strings get appended with "• Streamlit". 
        page_icon="🎲",                # String, anything supported by st.image, or None.
        layout="centered",             # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto")  # Can be "auto", "expanded", "collapsed"



    # ======================= Nav Bar  ==================== #
    nav = 1
    part1, part2, part3 = st.beta_columns([1, 1, 1])

    pages = ['💠 Part I: Probability          ', 
             '💠 Part II: Error                    ',
             '💠 Part III: P-values             ']

    pages[nav] = pages[nav]+' Selected'

    with part1:  # excuse hacky whitespace (alt+255) for alignment 
        if st.button(pages[0]): nav = 0
    with part2:
        if st.button(pages[1]): nav = 1
    with part3:
        if st.button(pages[2]): nav = 2
    st.markdown('---')

    if nav == 0:
        st.text('part I')
        
        state = st_state._get_state()
        pages = {
            "Dashboard": st_state.page_dashboard,
            "Settings": st_state.page_settings,
        }

        st.sidebar.title(":floppy_disk: Page states")
        page = st.sidebar.radio("Select your page", tuple(pages.keys()))

        # Display the selected page with the session state
        pages[page](state)

        annotated_text("This ",
            ("is", "verb", "#8ef"),
            " some ",
            ("annotated", "adj", "#faa"),
            ("text", "noun", "#afa"),
            " for those of ",
            ("you", "pronoun", "#fea"),
            " who ",
            ("like", "verb", "#8ef"),
            " this sort of ",
            ("thing", "noun", "#afa"))


    elif nav == 1: ######## PART II ############

        # ================== AB Test Sliders  ================== #
        col1, col2 = st.beta_columns([1, 1]) # first column 4x the size of second

        with col1: 
            st.header("📺 Variation A")
            a_conversion = st.slider('True Conversion Rate',0., 1., 0.41)

        with col2:
            st.header("📺 Variation B")
            b_conversion = st.slider('True Conversion Rate',0., 1., 0.48)
        st.write('')
        st.write('')

        # ============== Setup placeholder chart =============== #
        dx = pd.DataFrame([[a_conversion, b_conversion] for x in range(10)], columns=["A_Conv", "B_Conv"])
        dx.index.name = "x"
        y_max = max([a_conversion,b_conversion])+0.1
        y_min = max(0, min([a_conversion,b_conversion])-0.15)
        data = dx.reset_index().melt('x')

        lines = alt.Chart(data).mark_line().encode(
            x=alt.X('x', title='Iteration', axis=alt.Axis(tickMinStep=1)),
            y=alt.Y('value', title='Conversion', scale=alt.Scale(domain=[y_min, y_max])),
            color=alt.Color('variable', title=''))
        
        labels = lines.mark_text(align='left', baseline='middle', dx=3).encode(
                    alt.X('x:Q', aggregate='max'),
                    text='value:Q')

        line_plot = st.altair_chart(lines+labels, use_container_width=True)

        # ==================== User inputs ==================== #
        n_samples = st.number_input('Samples', min_value=0, max_value=5001, value=500)
        n_experiments = st.number_input('Iterations (how many times to run the experiment?)', min_value=0, max_value=1000, value=20)
        run = st.checkbox('Run')
    
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
                dx[f'A_Conv'] = pd.Series(res_a)
                dx[f'B_Conv'] = pd.Series(res_b)
                d_res = dx.copy()

                dx.index.name = "x"
                dx = dx.reset_index().melt('x') # nice shape for altair

                base = alt.Chart(dx)

                lines = alt.Chart(dx).mark_line().encode(
                    x=alt.X('x', title='Iterations', axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y('value', title='Conversion', scale=alt.Scale(domain=[y_min, y_max])),
                    color=alt.Color('variable', title=''))
                
                rule = base.mark_rule(strokeDash=[5,3]).encode(
                    y='average(value)',
                    color=alt.Color('variable'),
                    opacity=alt.value(0.4),
                    size=alt.value(2))

                # UPTO:
                # labels = lines.mark_text(align='left', baseline='middle', dx=3).encode(
                #     alt.X('x:Q', aggregate='max'),
                #     text='value:Q')
                labels = lines.mark_text(align='left', baseline='middle', dx=3).encode()


                line_plot.altair_chart(lines + rule + labels, use_container_width=True)
                if n_experiments < 20: wait_period = 0.05
                else: wait_period = 1 / n_experiments
                time.sleep(wait_period) 

            st.text(f"Experiment failure: {d_res[d_res['B_Conv'] < d_res['A_Conv']].shape[0]}/{n_experiments} (false positives)")
    
    elif nav == 3: ######## PART III ############    
        st.text('part 3')


    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()

if __name__ == '__main__':
    main()
