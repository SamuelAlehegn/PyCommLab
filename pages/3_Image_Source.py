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
    with st.expander("Classification of Digital Image"):

        st.write("Image to Binary Image Converter")
        # Upload image
        uploaded_file = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read image
            image = cv2.imdecode(np.fromstring(
                uploaded_file.read(), np.uint8), 1)

            image_type = st.selectbox(
                "Select Image", ("Binary Image", "Grayscale Image", "Color Image"))
            if image_type == "Binary Image":

                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Convert to binary image using thresholding
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

                column_one, column_two = st.columns(2)
                with column_one:
                    # Display original and binary images
                    st.image(image, caption="Original Image",
                             use_column_width=True)
                with column_two:
                    st.image(binary, caption="Binary Image",
                             use_column_width=True)

            elif image_type == "Grayscale Image":
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                column_one, column_two = st.columns(2)
                with column_one:
                    st.image(image, caption="Original Image",
                             use_column_width=True)
                with column_two:
                    st.image(gray, caption="Grayscale Image",
                             use_column_width=True)
            else:

                color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                column_one, column_two = st.columns(2)

                # color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                with column_one:
                    st.image(image, caption="Original Image",
                             use_column_width=True)
                with column_two:
                    st.image(image, caption="Color Image",
                             use_column_width=True)
            


with st.expander("Quize"):
    st.subheader
with st.expander("References"):
    st.subheader
