import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Image Source",  layout="wide",
                   initial_sidebar_state="auto")
with st.expander("Introduction"):
    st.write("The non deterministic source")
with st.expander("Theory"):
    st.subheader
with st.expander("Procedure"):
    st.subheader
with st.container():
    st.write("Simulations")
    # st.subheader
    st.expander("Classification of Digital Image")
    input_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    digital_Image = st.selectbox(
        "Digital Image", ("Binary Image", "Grayscale Image", "Color Image"))
    if input_image is not None:
        if digital_Image == "Binary Image":

            # Load the image in grayscale
            img = cv2.imread('image.jpg', 0)

            # Convert the image to binary
            ret, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

            # Display the original and binary images
            cv2.imshow('Original Image', img)
            cv2.imshow('Binary Image', binary_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


with st.expander("Quize"):
    st.subheader
with st.expander("References"):
    st.subheader
