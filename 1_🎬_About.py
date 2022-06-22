import streamlit as st
from PIL import Image


st.title('Capstone Project General Assembly: Medium article clap predictor')

# Image
image = Image.open('assets/medium.jpeg')
st.image(image)

st.subheader(
    "This is the web app for my capstone project. Where you can visualize my results and EDA")
st.subheader(
    "You can also experiment with your own Medium articles and get article recommendations based on your content!")

st.balloons()

st.write(f'Link to the github repo: https://github.com/Arik-LT/Medium_Article_Analysis_and_Sucess_Predictor')
