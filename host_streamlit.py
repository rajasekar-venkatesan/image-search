import streamlit as st
from PIL import Image
from image_search import *

"""
# Image Search from Free Text

Given a free text query, search through the images directory to find the most relevant images
"""

query = st.text_input("Enter Search Query: ", "trees near beach")
num_results = st.slider("Num Search Results", 1, 5, 3, 1)

results = search(query, num_results)
for index, item in enumerate(results):
    st.write(f'Result {index+1}: {item[0]} : Similarity Score: {item[1]:.3f}')
    st.image(Image.open(item[2]))
