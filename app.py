#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[7]:


import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import os
import pickle

#Open model created by the notebook
model = pickle.load(open('/model/box_office_model.pkl','rb'))

#create main page
def main():
    image = Image.open('/assets/Box_Office.jfif')
    st.image(image, use_column_width=False)


    add_selectbox = st.sidebar.selectbox('How would you like to predict?', ('Online', 'test'))
    st.sidebar.info('This app is created to predict revenue for movies' )
    st.sidebar.success('DAT158')
    st.title('Box Office Predictions')


    if add_selectbox == 'Online':
        budget = st.number_input('budget', min_value=0, max_value=1000000000, value=1000000)
        popularity = st.number_input('popularity', min_value=0., max_value=100., value=0., format="%.2f", step=1.)
        runtime = st.number_input('runtime', min_value=0., max_value=500., value=0., format="%.2f", step=1.)

        inputs = [[budget,runtime,popularity]]
        inputs_scaled = StandardScaler().fit_transform(inputs)

        if st.button('Predict'):
            result = model.predict(inputs)
            #format_result = "{:.2f}".format(float(result))
            print(result)
            st.success('Predicted output: €{:,.2f}'.format(float(result)))




#Start application
if __name__ =='__main__':
  main()
