from collections import defaultdict
from heapq import heappush, heappop, heapify
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2


st.set_page_config(page_title="Image Source",  layout="wide",
                   initial_sidebar_state="auto")
st.set_option('deprecation.showPyplotGlobalUse', False)



# ********************************************************


def huffman_encoding(data):
    # Calculate frequency of each character
    freq = defaultdict(int)
    for char in data:
        freq[char] += 1

    # Create a priority queue with tuples of (frequency, character)
    heap = [[freq[char], char] for char in freq]
    heapify(heap)

    # Combine nodes with lowest frequency until there is only one node left
    while len(heap) > 1:
        left = heappop(heap)
        right = heappop(heap)
        heappush(heap, [left[0] + right[0], left[1] + right[1]])

    # Create a dictionary with the Huffman code for each character
    huffman_code = {}

    def traverse(node, code):
        if len(node) == 1:
            huffman_code[node] = code
            return
        traverse(node[0], code + '0')
        traverse(node[1], code + '1')
    traverse(heap[0], '')

    # Encode the data using the Huffman code
    encoded_data = ''.join([huffman_code[char] for char in data])

    return encoded_data, huffman_code


def huffman_decoding(encoded_data, huffman_code):
    # Invert the Huffman code dictionary
    inv_huffman_code = {v: k for k, v in huffman_code.items()}

    # Decode the encoded data using the Huffman code
    decoded_data = ''
    i = 0
    while i < len(encoded_data):
        j = i + 1
        while encoded_data[i:j] not in inv_huffman_code:
            j += 1
        decoded_data += inv_huffman_code[encoded_data[i:j]]
        i = j

    return decoded_data

# st.title("Huffman Coding for Image Compression")


# Get input image file from user
image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if image_file is not None:
    # Read image data as bytes
    image_data = image_file.read()

    # Encode image data using Huffman coding
    encoded_data, huffman_code = huffman_encoding(image_data)

    # Decode encoded data back to the original image data
    decoded_data = huffman_decoding(encoded_data, huffman_code)

    # Display original and decoded images
    st.image(image_data, caption="Original Image")
    st.image(decoded_data, caption="Decoded Image")

# ********************************************************
# Run-length encoding (RLE)
# Function to encode an image using RLE


def rle_encode(image):
    pixels = image.load()
    width, height = image.size
    encoded = []
    for y in range(height):
        row = []
        count = 0
        color = pixels[0, y]
        for x in range(width):
            if pixels[x, y] == color:
                count += 1
            else:
                row.append((count, color))
                count = 1
                color = pixels[x, y]
        row.append((count, color))
        encoded.append(row)
    return encoded
# Function to decode an image using RLE


def rle_decode(encoded):
    image = Image.new("RGB", (len(encoded[0]), len(encoded)))
    pixels = image.load()
    for y in range(len(encoded)):
        x = 0
        for count, color in encoded[y]:
            for i in range(count):
                pixels[x, y] = color
                x += 1
    return image


with st.container():
    st.subheader("Introduction")
    with st.expander("Introduction"):
        st.write("Introduction")
with st.container():
    st.subheader("Theory")
    with st.expander("Theory"):
        st.write("Theory")
with st.container():
    st.subheader("Procedure")
    with st.expander("Procedure"):
        st.write("Procedure")
with st.container():
    st.subheader("Simulation")

    # st.subheader
    with st.expander("Classification of Digital Image"):

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
                st.write("Image to Binary Image Converter")

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
                st.write("Image to Grayscale Image Converter")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                column_one, column_two = st.columns(2)
                with column_one:
                    st.image(image, caption="Original Image",
                             use_column_width=True)
                with column_two:
                    st.image(gray, caption="Grayscale Image",
                             use_column_width=True)
            else:
                st.write("Image to Color Image Converter")
                color = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                column_one, column_two = st.columns(2)

                # color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                with column_one:
                    st.image(image, caption="Original Image",
                             use_column_width=True)
                with column_two:
                    st.image(color, caption="Color Image",
                             use_column_width=True)

# ********************************************************************************************
# Source Coding
    with st.expander("Source Coding"):
        st.subheader("Source Coding")
        uploaded_file = st.file_uploader(
            "Choose an image file", type=["jpg", "jpeg", "png"], key="source_coding")
        if uploaded_file is not None:
            types_of_source_coding = st.selectbox(
                "Select Source Coding Techniques ", ("1. Lossless Coding", "2. Lossy Coding", "3. Predictive Coding", "4. Transform Coding"))

            if types_of_source_coding == "1. Lossless Coding":
                st.write("Lossless Coding")
                lossless_coding = st.selectbox("Select Lossless Coding Techniques ", (
                    "1. Huffman Coding", "2. Arithmetic Coding", "3. Lempel-Ziv Coding", "4. Run Length Coding"))
                if lossless_coding == "1. Huffman Coding":
                    st.write("Huffman Coding")
                    image_data = image_file.read()

                    # Encode image data using Huffman coding
                    encoded_data, huffman_code = huffman_encoding(image_data)

                    # Decode encoded data back to the original image data
                    decoded_data = huffman_decoding(encoded_data, huffman_code)

                    # Display original and decoded images
                    st.image(image_data, caption="Original Image")
                    st.image(decoded_data, caption="Decoded Image")

                elif lossless_coding == "2. Arithmetic Coding":
                    st.write("Arithmetic Coding")
                elif lossless_coding == "3. Lempel-Ziv Coding":
                    st.write("Lempel-Ziv Coding")
                else:
                    st.write("Run Length Coding")
                    image = Image.open(uploaded_file)
                    # st.image(image, caption="Original Image",
                    #          use_column_width=True)
                    encoded = rle_encode(image)
                    st.code(str(encoded), language="python")
                    # decoded = rle_decode(encoded)
                    # st.image(decoded, caption="Decoded Image", use_column_width=True)

            elif types_of_source_coding == "2. Lossy Coding":
                st.write("Lossy Coding")
                lossy_coding = st.selectbox("Select Lossy Coding Techniques ", (
                    "1. Vector Quantization", "2. Discrete Cosine Transform", "3. Discrete Wavelet Transform"))
                if lossy_coding == "1. Vector Quantization":
                    st.write("Vector Quantization")
                elif lossy_coding == "2. Discrete Cosine Transform":
                    st.write("Discrete Cosine Transform")
                else:
                    st.write("Discrete Wavelet Transform")
            elif types_of_source_coding == "3. Predictive Coding":
                st.write("Predictive Coding")
                predictive_coding = st.selectbox(
                    "Select Predictive Coding Techniques ", ("1. Linear Prediction", "2. Non Linear Prediction"))
                if predictive_coding == "1. Linear Prediction":
                    st.write("Linear Prediction")
                else:
                    st.write("Non Linear Prediction")
            else:
                st.write("Transform Coding")
                transform_coding = st.selectbox("Select Transform Coding Techniques ", (
                    "1. Discrete Cosine Transform", "2. Discrete Wavelet Transform"))
                if transform_coding == "1. Discrete Cosine Transform":
                    st.write("Discrete Cosine Transform")
                else:
                    st.write("Discrete Wavelet Transform")

with st.container():
    st.subheader("Quize")
    with st.expander("Quize"):
        st.write("Quize")
with st.container():
    st.subheader("References")
    with st.expander("References"):
        st.write("References")
