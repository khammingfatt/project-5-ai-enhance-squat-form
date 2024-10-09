# C:\Users\LENOVO YOGA CORE I5\Documents\2023\1 Data Science Immersive\Projects\project-5-ai-enhance-squat-form
# conda prompt: streamlit run app.py

import streamlit as st

# This page works like a main entry page of the streamlit app
st.title('AI Squat Training Assistant Demonstration')

recorded_file = 'images/correct_demo.mp4'
output_file = 'images/squat_counter_output_demo.mp4'

col1, col2 = st.columns(2)

with col1:
    st.subheader("The AI Squat Counter computes your squat count for you.")
    output_vid = st.video(output_file)
    st.write("The probability shows the probability of your state.")

with col2:
    st.subheader("The AI Squat Enhancer give you feedback on your correct and incorrect squat.")
    sample_vid = st.video(recorded_file)
    st.write("When your squat is incorrect, it will display text to inform you how to improve it.")

