import streamlit as st
st.title('MPG ML Project')

displacement = st.number_input('Displacement', value = 300, placeholder = 'enter a value for displacement')
horsepower = st.number_input('horsepower', value = 130, placeholder = 'enter a value for horsepower')
Weight = st.number_input('Weight', value = 3000, placeholder = 'enter a value for weight')
acceleration = st.number_input('acceleration', value = 300, placeholder = 'enter a value for acceleration')
import pickle
loaded_model = pickle.load(open('mpg_regression.sav','rb'))