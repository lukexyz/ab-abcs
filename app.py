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

    # load state object
    state = st_state._get_state()

    # ==================== Nav Bar  ==================== #
    if state.nav is None: state.nav = 1
    nav = state.nav
    part1, part2, part3 = st.beta_columns([1, 1, 1])

    pages = ['Part I: Probability     ', 
             'Part II: Error          ',
             'Part III: P-values      ']

    pages[nav] = '⚪ ' + pages[nav]

    with part1: 
        if st.button(pages[0]): state.nav = 0
    with part2:
        if st.button(pages[1]): state.nav = 1
    with part3:
        if st.button(pages[2]): state.nav = 2
    st.markdown('---')

    
    if nav == 0:  ############ PART I ############
        
        st.header('How do we know when an event has happened?')

        conversion_rate = st.number_input('True Conversion Rate', value=0.2)
        n_samples = st.number_input('People (n samples)', value=100)

        # ============== Setup placeholder chart =============== #
        res = []
        df = pd.DataFrame()
        df['A'] = pd.Series(res)
        df['conv'] = (df['A']>(1-conversion_rate)).astype(int)
        df = df.sort_values('conv')
        df['converted'] = df['conv'].map({1:'Yes', 0:'No'})

        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X('converted'),
            y=alt.Y('A')
        ).properties(width=300, height=300)

        hist = alt.Chart(df).mark_bar(size=40).encode(
            alt.X('count()'),
            alt.Y("conv"),
            
        ).properties(width=300, height=300)

        scatter_plot = st.altair_chart(scatter | hist, use_container_width=True)
        run_p1 = st.checkbox('Run')

        if run_p1:
            for i in range(n_samples):
                res.append(random())
                df = pd.DataFrame()
                df['A'] = pd.Series(res)
                df['conv'] = (df['A']>(1-conversion_rate)).astype(int)
                df = df.sort_values('conv')
                df['converted'] = df['conv'].map({1:'Yes', 0:'No'})

                scatter = alt.Chart(df.reset_index()).mark_circle(size=60).encode(
                    x=alt.X('index'),
                    y=alt.Y('A'),
                    color=alt.Color('converted', title='', legend=None)
                ).properties(width=300, height=300)

                x_max = max(df.converted.value_counts().values) 

                hist = alt.Chart(df).mark_bar(size=40).encode(
                    alt.X('count()', scale=alt.Scale(domain=[0, x_max], clamp=True)),
                    alt.Y("conv"),
                    color=alt.Color('converted', title='', legend=None)
                ).properties(width=300, height=300)

                text = hist.mark_text(
                    align='left', fontSize=12, 
                    baseline='middle',
                    dx = 3,
                    color='black'  # Nudges text to right so it doesn't appear on top of the bar
                ).encode(
                    x='count():Q',
                    text=alt.Text('count()', format='.0f')
                )

                scatter_plot.altair_chart(scatter | (hist + text), use_container_width=False)

                if n_samples < 20: wait_period = 0.20
                else: wait_period = 1 / n_samples
                time.sleep(wait_period)

            results_yes = df[df.converted=='Yes']
            results_no = df[df.converted=='No']
            result_text_1 = f'Conversion Rate = {df.conv.mean():0.2f}'

        if df.shape[0] >1:
            annotated_text("Simulation Observation 👩‍🔬  ", 
                        (result_text_1, f"{len(results_yes)}/{df.shape[0]} converted", "#fea"))

        

    elif nav == 1: ############ PART II ############


        # ================== AB Test Sliders  ================== #
        col1, col2 = st.beta_columns([1, 1]) # first column 1x the size of second

        with col1: 
            st.header("📺 Variation A")   
            a_conversion = st.slider('True Conversion Rate',0., 1., 0.20)

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
        n_samples = st.number_input('Samples', min_value=0, max_value=5001, value=200)
        n_experiments = st.number_input('Iterations (how many times to run the experiment?)', min_value=0, max_value=1000, value=20)
        run_p2 = st.checkbox('Run')
    
        res_a, res_b = [], []

        if run_p2: 
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
                    color=alt.Color('variable', title=''),
                    tooltip = [alt.Tooltip('x:N'), alt.Tooltip('value:N')]
                    )
                
                rule = base.mark_rule(strokeDash=[5,3]).encode(
                    y='average(value)',
                    color=alt.Color('variable'),
                    opacity=alt.value(0.4),
                    size=alt.value(2)
                    )
                
                hover = alt.selection_single(
                    fields=["x"],
                    nearest=True,
                    on="mouseover",
                    empty="none",
                    clear="mouseout"
                )

                tooltips = alt.Chart(dx).transform_pivot(
                    "x", "value", groupby=["x"]
                ).mark_rule().encode(
                    x='x:Q',
                    opacity=alt.condition(hover, alt.value(0.3), alt.value(0)),
                    tooltip=["x:Q", "value"]
                ).add_selection(hover)

                labels = lines.mark_text(align='left', baseline='middle', dx=3).encode()

                line_plot.altair_chart(lines + rule + labels + tooltips, use_container_width=True)
                if n_experiments < 20: wait_period = 0.05
                else: wait_period = 1 / n_experiments
                time.sleep(wait_period) 

            results_text_2 = f"{d_res[d_res['B_Conv'] < d_res['A_Conv']].shape[0]}/{n_experiments}"

            if df.shape[0] >1:
                annotated_text("Experiment Results 👨‍🔬 ", 
                            (results_text_2,
                            "False positives", "#fea"))

            st.text(f"Simulation failures: {d_res[d_res['B_Conv'] < d_res['A_Conv']].shape[0]}/{n_experiments} (false positives)")
    
    elif nav == 2: ######## PART III ############    
        st.text('part 3')

    # Mandatory to avoid rollbacks with widgets, must be called at the end of app
    state.sync()

if __name__ == '__main__':
    main()
