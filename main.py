import streamlit as st
import pandas as pd
import numpy as np
import itertools
from concurrent import futures
from time import time
from stqdm import stqdm
from solve import Solve
from poly import Builder

def print_stats(method_name, func_runtimes):
    if method_name not in func_runtimes:
        print("{!r} wasn't profiled, nothing to display.".format(method_name))
    else:
        runtimes = func_runtimes[method_name]
        total_runtime = sum(runtimes)
        average = total_runtime / len(runtimes)
        print('function: {!r}'.format(method_name))
        print(f'\trun times: {len(runtimes)}')
        # print('  total run time: {}'.format(total_runtime))
        print(f'\taverage run time: {average:.7f}')

def getError(params):
    params_new = params[-1].copy()
    params_new['degrees'] = [*(params[:-1])]
    solver = Solve(params_new)
    func_runtimes = solver.prepare()
    normed_error = min(solver.norm_error)
    return (params_new['degrees'], normed_error, func_runtimes)

def getSolution(params, pbar_container, max_deg=15):
    if params['degrees'][0] == 0:
        x1_range = list(range(1, max_deg+1))
    else:
        x1_range = [params['degrees'][0]]
    
    if params['degrees'][1] == 0:
        x2_range = list(range(1, max_deg+1))
    else:
        x2_range = [params['degrees'][1]]
    
    if params['degrees'][2] == 0:
        x3_range = list(range(1, max_deg+1))
    else:
        x3_range = [params['degrees'][2]]

    ranges = list(itertools.product(x1_range, x2_range, x3_range, [params]))
    tick = time()
    if len(ranges) > 1:
        with futures.ThreadPoolExecutor() as pool:
            results = list(stqdm(
                pool.map(getError, ranges), 
                total=len(ranges), 
                st_container=pbar_container,
                desc='**Підбір степенів**',
                backend=True, frontend=True))

        results.sort(key=lambda t: t[1])
    else:
        results = [getError(ranges[0])]
    # func_runtimes = {key: [] for key in results[-1][-1].keys()}
    # for key in func_runtimes:
    #     for res in results:
    #         func_runtimes[key] += res[-1][key]

    final_params = params.copy()
    final_params['degrees'] = results[0][0]
    solver = Solve(final_params)
    solver.prepare()
    tock = time()
    
    print('\n--- BEGIN DEBUG INFO ---')
    # for func in func_runtimes:
    #     print_stats(func, func_runtimes)

    print(f'TOTAL RUNTIME: {tock-tick:.3f} sec\n\n')

    solver.save_to_file()
    solution = PolynomialBuilder(solver)
    
    return solver, solution, final_params['degrees']
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
# col_sep = col1.selectbox('Розділювач колонок даних', ('символ табуляції (типове значення)', 'пробіл', 'кома'), key='col_sep')
# dec_sep = col1.selectbox('Розділювач дробової частини', ('крапка (типове значення)', 'кома'), key='dec_sep')
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

if col3.button('RUN', key='run'):
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

        st.write(solution.get_results())

        matr_cols = st.columns(3)
        for ind, info in enumerate(matrices[2:5]):
            matr_cols[ind].subheader(info[0])
            matr_cols[ind].dataframe(info[1])

        with open(params['output_file'], 'rb') as fout:
            col3.download_button(
                label='Download',
                data=fout,
                file_name=params['output_file']
#                 mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )