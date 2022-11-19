import streamlit as st

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
