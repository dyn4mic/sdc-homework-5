
import numpy as np

import streamlit as st
import ModelClient as model




def showGeneration(content,style):
  st.write('Generation: ')
  response=model.getGeneration(content,style,'localhost')
  st.image(np.array(response.json()['predictions']))

content=content=st.sidebar.selectbox('Select Content',model.content_images.keys())
style=style=st.sidebar.selectbox('Select Style',model.style_images.keys())
col1,col2,col3=st.columns(3)
with col1:
  st.write('Content: ',content)
  st.image(model.getContent(content).numpy())
with col2:
  st.write('Style: ',style)
  st.image(model.getStyle(style).numpy())
with col3:
  showGeneration(content,style)



