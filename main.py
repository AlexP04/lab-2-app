#Python imports
import streamlit as st
import pandas as pd
import numpy as np
import sys

#Other packages
from tool import *
from poly import Builder

#Setting tab icons and name
st.set_page_config(page_title='Solver - 2', 
                   layout='wide')

#seting color theme 
st.markdown("""
    <style>
    .stProgress .st-ey {
        background-color: #5fe0de;
    }
    </style>
    """, unsafe_allow_html=True)

#Setting general title 
st.title('Solver')

#Dividing page into three parts (main and parameters input + output) 
main, dims, degs, add = st.columns(4)

#Setting main input header
main.header('Files')

#Declared variables for input/output files will be used
input_name = main.file_uploader('Input file name', type=['csv'], key='input_file')
output_name = main.text_input('Output file name', value='output', key='output_file')

#Setting header for dimension input 
dims.header('Input dimensionality')

#Declaring variables for dimensionality of data
dim = dims.number_input('Dimension of Y', value=4, step=1, key='dim')
dim_1 = dims.number_input('Dimension of X1', value=2, step=1, key='dim_1')
dim_2 = dims.number_input('Dimension of X2', value=2, step=1, key='dim_2')
dim_3 = dims.number_input('Dimension of X3', value=3, step=1, key='dim_3')

#Same for degrees
degs.header('Input polynoms degrees ')

#Declaring variables
degree_1 = degs.number_input('Degree for X1', value=13, step=1, key='degree_1')
degree_2 = degs.number_input('Degree for X2', value=11, step=1, key='degree_2')
degree_3 = degs.number_input('Degree for X3', value=7, step=1, key='degree_3')

#Additional input section, some specifications
add.header('Input other parameters ')
use_type = add.radio('Polynomial type used: ', ['Chebyshev', 'Legendre', 'Laguerre', 'Hermite'])
lambdas =  add.checkbox('Enable search for lambdas')
normalize = add.checkbox('Plot normalized plots ')

#Defining functionality of run button
if main.button('Run', key='run'):
    try:
        #try-block
        #Parsing file recieved
        input_file_text = str(input_name.getvalue().decode())
        input_file = input_file_text.replace(",",".").replace(';', '\t')
        
        #Storing parameters in convinient way
        params = {
            'dimensions': [dim_1, dim_2, dim_3, dim],
            'input_file': input_file,
            'output_file': output_name + '.csv',
            'degrees': [degree_1, degree_2, degree_3],
            'polynomial_type': use_type,
            'lambda': lambdas
        }
      
        
        #Processing of data using packages created previously
        with st.spinner('...'):
            solver, degrees = get_solution(params, pbar_container=main, max_deg=8) 
      
        solution = Builder(solver) 
        
        #Showing and plotting errors
        error_cols = st.columns(2)
    
        for ind, info in enumerate(solver.show()[-2:]):
            error_cols[ind].subheader(info[0])
            error_cols[ind].dataframe(info[1])
        
        #Saving results in variables
        if normalize:
            Y_values = solution._solution.Y
            final_values = solution._solution.final
        else:
            #Saving results in variables
            Y_values = solution._solution.Y_
            final_values = solution._solution.final_d
            
       
        cols = Y_values.shape[1]
        
        #Results section
        st.subheader('Results')
        
        #Defining layout of plots
        plot_cols = st.columns(cols)
        
        #Plotting residuals, components for each dimension of Y
        for n in range(cols):
            df = pd.DataFrame(
                np.array([Y_values[:, n], final_values[:, n]]).T,
                columns=[f'Y{n+1}', f'F{n+1}']
            )
            plot_cols[n].write(f'Component №{n+1}')
            plot_cols[n].line_chart(df)
            plot_cols[n].write(f'Сomponent\'s №{n+1} residual')
            
            df = pd.DataFrame(
                np.abs(Y_values[:, n] - final_values[:, n]).T,
                columns=[f'Error{n+1}']
            )
            plot_cols[n].line_chart(df)
        
        #Show polynoms
        matrices = solver.show()[:-2]
                                 
        if normalize:
            st.subheader(matrices[1][0])
            st.dataframe(matrices[1][1])
        else:
            st.subheader(matrices[0][0])
            st.dataframe(matrices[0][1])

        st.write(solution.get_results())

        matr_cols = st.columns(3)
        
        for ind, info in enumerate(matrices[2:5]):
            matr_cols[ind].subheader(info[0])
            matr_cols[ind].dataframe(info[1])
        
        #Downloading output button
        with open(params['output_file'], 'rb') as fout:
            main.download_button(
                label='Download output file',
                data=fout,
                file_name=params['output_file']
            )
            
    except Exception as ex:
        #except-block, if something goes wrong
        st.write("Exception :"+ str(sys.exc_info()) + ":: Check input and try again")