#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[4]:


from pycaret.regression import *
import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image
import os
import pickle


# ## Streamlit

# In[5]:


model = pickle.load(open('../model/box_office_model.pkl','rb'))
print(model)

def main():
    image = Image.open('../assets/Box_Office.jfif')
    st.image(image, use_column_width=False)


    add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'Batch')) #bruke batch for aa predikere paa alle bildene. 
    st.sidebar.info('This app is created to predict revenue for movies' )
    st.sidebar.success('DAT158')
    st.title('Box Office Predictions')
    

    if add_selectbox == 'Online': 
        budget = st.number_input('budget', min_value=0, max_value=100000000, value=1000000)
        popularity = st.number_input('popularity', min_value=0, max_value=100, value=0)
        runtime = st.number_input('runtime', min_value=0, max_value=500, value=0)
        
        inputs = [[budget,runtime,popularity]]
        #inputs_scaled = StandardScaler().fit_transform(inputs)

        if st.button('Predict'): 
            result = model.predict(inputs)
            print('result: ',result)
            result = result.flatten().astype(float)
            st.success('Predicted output: {}'.format(result))
            
        
        
    if add_selectbox == 'Batch': 
        fn = st.file_uploader("Upload csv file for predictions") #st.file_uploader('Upload csv file for predictions, type=["csv"]')
        if fn is not None: 
            input_df = pd.read_csv(fn)
            predictions = self.predict(input_df)
            st.write(predictions)
if __name__ =='__main__':
  main()
