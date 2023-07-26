# cd C:\Users\LENOVO YOGA CORE I5\Documents\2023\1 Data Science Immersive\Projects\Capstone Projects\Streamlit
# conda prompt: streamlit run app.py


import streamlit as st

# This page works like a main entry page of the streamlit app
st.title('AI Squat Training Assistant Demonstration')

recorded_file = 'correct_demo.mp4'
sample_vid = st.empty()
sample_vid.video(recorded_file)
