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
        try:
            input_file_text = str(input_name.getvalue().decode())
        except:
            #default variant
            input_file_text = """6,05;12,02;1,05;9,02;10;1;6,1;254,621;58,145;119,406;117,683
8,15;10,1;1,15;9,11;15,8;2,1;4,2;298,163;73,368;92,651;90,123
10,2;8,13;1,19;9,13;22,5;2,5;3,5;387,411;71,084;87,691;83,576
12,25;6,18;2,25;9,18;25,7;3,51;2,72;467,197;83,567;78,793;74,789
14,33;5,2;4,33;9,2;32,5;4,2;2,53;566,547;93,813;79,497;54,316
16,35;4,25;6,35;9,25;35;5,02;2,1;653,789;101,378;77,082;32,817
18,49;3,4;8,41;9,5;40,7;8,2;1,15;710,926;155,579;67,758;57,425
20,7;2,5;10,51;10,5;51,8;10,1;0,72;851,381;160,432;71,956;89,519
44826;2,7;12,61;11,6;65;12,8;0,54;987,364;176,283;91,123;121,374
18,45;3,7;14,7;13,7;82;14,4;0,15;1036,123;193,657;112,859;249,173
16,75;4,75;15,75;15,75;95,4;14,7;0,55;1292,341;278,624;153,717;384,136
14,8;5,78;17,8;17,78;102,8;15,5;1,76;1088,324;354,324;117,965;479,152
12,95;6,8;19,85;19,8;117;16,3;2,23;926,939;478,926;155,912;501,239
10,84;7,85;18,05;21,85;125,78;16,7;3,61;877,128;588,675;169,359;625,482
8,91;8,86;16,91;23,86;97;16,9;5,16;605,327;499,367;192,924;740,976
15,93;10,87;14,93;25,87;95,5;17,5;8,25;458,386;468,567;218,549;875,846
14,93;12,89;12,01;27,88;93,9;17,7;11,37;218,859;353,932;247,354;916,124
13,93;14,92;10,93;25,9;91,5;18,2;13,26;195,737;335,124;284,167;863,928
12,94;16,95;8,94;23,95;79,58;19,1;15,51;306,168;261,946;316,375;703,153
11,95;18,98;6,95;21,98;55,4;19,5;17,74;685,761;151,387;341,326;631,195
10,81;21;4,95;19,02;31,5;21;13,14;890,639;210,519;375,651;571,588
8,75;22,98;2,11;17,98;12,5;23,56;11,35;923,784;485,142;344,856;436,847
6,15;19,95;1,25;15,95;10,8;25,3;8,58;1031,438;688,125;348,314;441,842
5,2;18,9;3,2;13,92;8,5;28,7;6,74;1121,321;883,435;344,716;439,425
4,45;17,88;5,25;11,88;4,4;31,56;4,85;1291,845;972,834;329,942;322,147
7,33;15,87;8,33;9,87;2,5;27,1;6,21;1308,614;1080,562;349,316;235,954
8,35;13,86;11,35;7,86;5,3;24,7;9,52;1529,956;887,987;348,231;150,492
9,4;11,85;15,41;5,85;8,7;26,2;10,75;1730,129;688,951;347,987;254,897
10,5;9,78;17,5;3,78;11,2;23,7;8,1;1917,152;455,494;342,967;458,289
12,6;7,75;15,61;1,75;14,7;20,36;6,1;2278,654;211,209;132,856;672,164
14,7;5,71;13,7;3,7;17,8;17,7;4,15;2412,145;96,197;115,632;453,356
16,75;3,6;11,75;5,61;20,1;13,34;2,36;2186,243;77,325;93,135;227,168
18,8;2,5;9,8;7,5;40,52;11,72;1,35;1862,345;64,615;77,824;106,123
19,85;4,39;7,85;9,42;65,2;9,9;2,13;1632,879;52,534;63,453;82,659
17,91;6,25;5,91;11,26;80,76;7,74;4,57;1467,156;45,178;79,167;93,834
15,91;8,19;3,91;13,21;91,1;6,36;6,75;1270,531;36,176;80,836;91,345
13,93;10,18;2,93;11,18;109,5;5,7;9,26;1084,243;20,364;87,192;96,841
11,93;12,13;1,93;9,13;122,9;4,75;11,79;881,956;10,428;85,834;93,952
9,01;14,11;3,93;7,09;108,3;3,65;13,12;616,829;8,475;101,985;109,463
7,94;12,01;5,94;5,99;84,5;3,52;15,36;473,329;10,924;128,591;233,415
5,95;10,11;7,95;3,12;58,6;2,72;12,85;249,421;24,183;102,861;308,613
5,02;8,12;10;1,12;35,8;2,34;10,34;225,356;46,324;105,817;207,319
4,05;6,13;11,95;2,12;15,26;2,16;8,68;176,578;76,457;78,473;182,263
5,94;4,13;13,94;4,13;9,52;1,76;5,32;170,948;95,814;81,417;84,132
6,93;2,14;15,93;6,14;4,8;1,48;2,16;170,948;95,814;81,417;84,132
"""
            
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