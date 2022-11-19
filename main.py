import streamlit as st
import pandas as pd
import numpy as np
import itertools
from concurrent import futures
from tool import getError, getSolution

st.set_page_config(page_title='СА ЛР2', 
                   page_icon='📈',
                   layout='wide',
                   menu_items={
                       'About': 'Лабораторна робота №2 з системного аналізу'
                   })

st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
    }
    </style>
    """, unsafe_allow_html=True)

st.title('Select parameters and run: ')
params, main, addon = st.columns(3)
main.header('General input/output info:')
IN = main.file_uploader('Input file name', type=['csv', 'txt'], key='input_file')
output_name = main.text_input('Output file name', value='output', key='output_file')

params.header('Mandatory input parameters:')
dim = params.number_input('Dimension of Y', value=4, step=1, key='dim')
dim_1 = params.number_input('Dimension of X1', value=2, step=1, key='dim_1')
dim_2 = params.number_input('Dimension of X2', value=2, step=1, key='dim_2')
dim_3 = params.number_input('Dimension of X3', value=3, step=1, key='dim_3')
degree_1 = params.number_input('Degree for X1', value=13, step=1, key='degree_1')
degree_2 = params.number_input('Degree for X2', value=11, step=1, key='degree_2')
degree_3 = params.number_input('Degree for X3', value=7, step=1, key='degree_3')
use_type = params.radio('Polynomial type used: ', ['Chebyshev', 'Legendre', 'Laguerre', 'Hermite'])

addon.header('Additional input parameters:')
init_weight = addon.radio('Weights initialization: ', ['Mean', 'Normalized'])
lambdas = addon.checkbox('Fond lambdas from equations: ')
# norme = addon.radio('Plot normalized gra: ', ['Mean', 'Normalized'])addon.checkbox('Графіки для нормованих значень')

if addon.button('RUN', key='run'):
    try:
        input_file = IN.getvalue().decode()
        input_file = input_file_text.replace(',', '.').replace(' ', '\t')
        
        params = {
            'dimensions': [dim_1, dim_2, dim_3, dim],
            'input_file': input_file_text,
            'output_file': output_name + '.csv',
            'degrees': [degree_1, degree_2, degree_3],
            'weights': init_weight,
            'poly_type': use_type,
            'lambda': lambdas
        }
        with st.spinner('...'):
            solver, solution, degrees = getSolution(params, pbar_container=addon, max_deg=15)
        if degrees != params['degrees']:
            col3.write(f'**Підібрані степені поліномів:**  \nX1 — {degrees[0]}  \nX2 — {degrees[1]}  \nX3 — {degrees[2]}')

        error_cols = st.columns(2)
        for ind, info in enumerate(solver.show_streamlit()[-2:]):
            error_cols[ind].subheader(info[0])
            error_cols[ind].dataframe(info[1])

#         if normed_plots:
#             Y_values = solution._solution.Y
#             F_values = solution._solution.F
#         else:
        Y_values = solution._solution.Y_
        F_values = solution._solution.F_

        cols = Y_values.shape[1]
        
        st.subheader('Result: ')
        plot_cols = st.columns(cols)

        for n in range(plot_n_cols):
            df = pd.DataFrame(
                np.array([Y_values[:, n], F_values[:, n]]).T,
                columns=[f'Y{n+1}', f'F{n+1}']
            )
            plot_cols[n].write(f'Component {n+1}')
            plot_cols[n].line_chart(df)
            plot_cols[n].write(f'Error of component {n+1}')
            df = pd.DataFrame(
                np.abs(Y_values[:, n] - F_values[:, n]).T,
                columns=[f'E{n+1}']
            )
            plot_cols[n].line_chart(df)

        matrices = solver.show_streamlit()[:-2]
#         if normed_plots:
#             st.subheader(matrices[1][0])
#             st.dataframe(matrices[1][1])
#         else:
        st.subheader(matrices[0][0])
        st.dataframe(matrices[0][1])

        st.write(solution.process_final())

        matr_cols = st.columns(3)
        for ind, info in enumerate(matrices[2:5]):
            matr_cols[ind].subheader(info[0])
            matr_cols[ind].dataframe(info[1])
    except:
        st.write("Something went wrong, check inputs")
            