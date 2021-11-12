#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[25]:


from pycaret.regression import *
import pandas as pd 
import numpy as np
import streamlit as st
from PIL import Image
import os


# ## Streamlit

# In[26]:


    
    """ def __init__(self):
        self.model = load_model('../model/box_office_model') 
        self.save_fn = '../path.csv' 
        
    def predict(self, input_data): 
        return predict_model(self.model, data=input_data)
    
    def store_prediction(self, output_df): 
        if os.path.exists(self.save_fn):
            save_df = pd.read_csv(self.save_fn)
            save_df = save_df.append(output_df, ignore_index=True)
            save_df.to_csv(self.save_fn, index=False)
            
        else: 
            output_df.to_csv(self.save_fn, index=False) 
            

    
    def run(self):
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
            
            output=''
            input_dict = {'budget':budget, 'popularity':popularity, 'runtime':runtime}
            input_df = pd.DataFrame(input_dict, index=[0])
        
            if st.button('Predict'): 
                output = self.predict(input_df)
                self.store_prediction(output)
                
                output = output['Label'][0]
                #output = str(output['Label'])
                
            
            st.success('Predicted output: {}'.format(output))
            
        if add_selectbox == 'Batch': 
            fn = st.file_uploader("Upload csv file for predictions") #st.file_uploader('Upload csv file for predictions, type=["csv"]')
            if fn is not None: 
                input_df = pd.read_csv(fn)
                predictions = self.predict(input_df)
                st.write(predictions)         """
            
# In[24]:



