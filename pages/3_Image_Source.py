from collections import defaultdict
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from scipy.special import comb


st.set_page_config(page_title="Image Source",  layout="wide",
                   initial_sidebar_state="auto")
st.set_option('deprecation.showPyplotGlobalUse', False)


# ********************************************************
def image_to_matrix(image):
    return np.asarray(image.convert('RGB'))


def matrix_to_image(matrix):
    return Image.fromarray(np.uint8(matrix))


def matrix_to_binary(matrix):
    return np.unpackbits(matrix.astype(np.uint8), axis=-1)


def binary_to_matrix(binary_matrix):
    return np.packbits(binary_matrix, axis=-1).astype(np.uint8)


def bpsk_modulation(binary_matrix):
    return 2*binary_matrix - 1


def bpsk_demodulation(modulated_matrix):
    return (modulated_matrix > 0).astype(np.uint8)


def qpsk_modulation(binary_matrix):
    binary_matrix = binary_matrix.reshape(-1, 2)
    symbols = np.zeros(binary_matrix.shape[0], dtype=np.complex128)
    symbols[np.logical_and(binary_matrix[:, 0] == 0,
                           binary_matrix[:, 1] == 0)] = 1 + 1j
    symbols[np.logical_and(binary_matrix[:, 0] == 0,
                           binary_matrix[:, 1] == 1)] = -1 + 1j
    symbols[np.logical_and(binary_matrix[:, 0] == 1,
                           binary_matrix[:, 1] == 0)] = -1 - 1j
    symbols[np.logical_and(binary_matrix[:, 0] == 1,
                           binary_matrix[:, 1] == 1)] = 1 - 1j
    return symbols


def qpsk_demodulation(modulated_matrix):
    binary_matrix = np.zeros((modulated_matrix.size * 2,), dtype=np.uint8)
    binary_matrix[::2] = np.real(modulated_matrix) < 0
    binary_matrix[1::2] = np.imag(modulated_matrix) < 0
    return binary_matrix


def awgn_channel(signal, snr_db):
    snr = 10**(snr_db/10)
    power = np.mean(np.abs(signal)**2)
    noise_power = power/snr
    noise_std = np.sqrt(noise_power)
    noise = noise_std*np.random.randn(*signal.shape)
    return signal + noise


# def rayleigh_channel(signal):
#     snr_db = 0.01
#     noise = np.random.normal(scale=np.sqrt(0.5), size=signal.shape)
#     signal_with_noise = signal + 10 ** (-snr_db / 20) * noise
#     return signal_with_noise


def rayleigh_channel_with_mitigate(signal, n_antennas):
    snr_db = 1
    h = np.random.rayleigh(1, size=(n_antennas,) + signal.shape)
    noise = np.random.normal(scale=np.sqrt(0.5), size=signal.shape)
    signal_with_noise = signal + 10 ** (-snr_db / 20) * noise
    return h * signal_with_noise


def selection_diversity(signal):
    return np.max(signal, axis=0)


def rayleigh_channel_without_mitigate(signal):
    h = np.sqrt(0.5)*(np.random.randn(*signal.shape))
    return h*signal


def rayleigh_channel_with_selection_diversity(signal, num_antennas):
    h = np.sqrt(0.5)*(np.random.randn(num_antennas, *signal.shape) +
                      1j*np.random.randn(num_antennas, *signal.shape))
    h_magnitude = np.abs(h)
    max_index = np.argmax(h_magnitude, axis=0)
    h_max = np.take_along_axis(h, np.expand_dims(
        max_index, axis=0), axis=0).squeeze(axis=0)
    return h_max * signal


def rician_channel(signal, K):
    # Generate Rician fading channel coefficients
    A = np.sqrt(K/(K+1))
    sigma = np.sqrt(1/(2*(K+1)))
    h = A + sigma*(np.random.randn(*signal.shape) +
                   1j*np.random.randn(*signal.shape))
    return h*signal


def hamming_encode(data):
    G = np.array([[1, 1, 0, 1],
                  [1, 0, 1, 1],
                  [1, 0, 0, 0],
                  [0, 1, 1, 1],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]).T
    return data.dot(G) % 2


def hamming_decode(codeword):
    H = np.array([[1, 0, 1, 0, 1, 0, 1],
                  [0, 1, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1]])
    syndrome = H.dot(codeword.T) % 2
    error = np.sum(2**np.arange(2, -1, -1)*syndrome.T, axis=1)
    corrected = codeword.copy()
    corrected[np.arange(corrected.shape[0]), error-1] ^= error != 0
    R = np.array([[0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1]])
    return R.dot(corrected.T).T


with st.container():
    st.subheader("Introduction")
    with st.expander("Introduction"):
        st.subheader("Introduction")
        st.write("Welcome to our image transmission simulation page! The main purpose of this website page is to provide an interactive and immersive experience for users who want to simulate the process of image transmission. Our goal is to help users understand the various components involved in image transmission, such as encoding, decoding, compression, and error correction, by providing a hands-on approach to learning. Through our simulation tools and step-by-step guides, users can explore how images are transmitted over networks and learn about the technical details that make it all possible.")
with st.container():
    st.subheader("Theory")
    with st.expander("Theory"):
        st.write("Image is a visual representation of an object, scene, or concept that can be captured and stored in various forms. In digital imaging, an image is typically represented as a two-dimensional array of pixels, each of which represents a unique color or shade of gray. The resolution of an image is determined by the number of pixels in the array, with higher resolutions resulting in sharper and more detailed images.")
        st.write("Images can be created and manipulated using a variety of techniques, including photography, digital art, and computer graphics. They are used for a wide range of purposes, including artistic expression, scientific visualization, and communication.")
        st.write("One of the most important aspects of digital imaging is the compression and transmission of images over networks. To reduce the file size of an image for storage and transmission, various compression techniques can be applied, such as lossy and lossless compression. Lossy compression techniques use algorithms to eliminate or reduce redundant information in an image, resulting in a smaller file size but with some loss of image quality. Lossless compression techniques, on the other hand, retain all of the original image data and achieve smaller file sizes through more efficient encoding.")
        st.write("During transmission, images can be subject to various types of noise and distortion, such as compression artifacts, transmission errors, and interference. To minimize the effects of these distortions, error-correcting codes and other techniques can be used to ensure that the transmitted image is as close as possible to the original.")
        st.write("Digital images can be classified based on various characteristics, such as their format, color space, and resolution. Here are some common classifications of digital images:")
        st.write("1. Format: Digital images can be stored in various file formats, such as JPEG, PNG, TIFF, and BMP. Each format has its own advantages and disadvantages in terms of file size, compression, and image quality.")
        st.write("2. Color space: Digital images can be represented using various color spaces, such as RGB, CMYK, and grayscale. RGB color space is commonly used for digital imaging and is based on the combination of red, green, and blue light to create colors. CMYK color space is used for printing and is based on the combination of cyan, magenta, yellow, and black ink to create colors. Grayscale images contain only shades of gray and are commonly used for black-and-white photography.")
        st.write("3. Resolution: Digital images can be classified based on their resolution, which is determined by the number of pixels in the image. High-resolution images contain more pixels and therefore have higher image quality and sharper details. Low-resolution images have fewer pixels and may appear blurry or pixelated when zoomed in.")
        st.write("4. Bit depth: Digital images can also be classified based on their bit depth, which determines the number of colors that can be represented in the image. Images with higher bit depth can represent more colors and have smoother tonal gradations, while images with lower bit depth may have visible color banding.")
        st.write("Binary, grayscale, and color are three common types of digital images that differ in the way they represent color information.")
        st.write("1. Binary images: Binary images are the simplest type of digital images, consisting of only two colors – usually black and white. In a binary image, each pixel is either black (representing a value of 0) or white (representing a value of 1). Binary images are commonly used in applications such as document scanning, OCR, and barcode recognition.")
        st.write("2. Grayscale images: Grayscale images contain shades of gray ranging from black to white. In a grayscale image, each pixel is represented by a single value that corresponds to its brightness, with values ranging from 0 (black) to 255 (white). Grayscale images are commonly used in applications such as medical imaging, scientific visualization, and black-and-white photography.")
        st.write("3. Color images: Color images contain multiple colors, typically represented in the RGB or CMYK color space. In an RGB color image, each pixel is represented by three values corresponding to its red, green, and blue color components. In a CMYK color image, each pixel is represented by four values corresponding to its cyan, magenta, yellow, and black color components. Color images are commonly used in applications such as digital art, graphic design, and photography.")
        st.write("Each type of digital image has its own unique characteristics and uses. Binary images are simple and efficient for applications that require only black and white information, while grayscale images provide more detail and tonal range for applications that require varying shades of gray. Color images provide the most information and flexibility, allowing for a wide range of colors and visual effects. Understanding the differences between these types of images is important for selecting the appropriate image format for a given application.")
        st.write("")
        st.write("Image transmission with BPSK and QPSK modulation techniques and AWGN, Rayleigh, and Rician channel models is a challenging process that involves several stages.")
        st.write("1. BPSK (Binary Phase-Shift Keying): BPSK is a digital modulation technique that modulates the carrier signal by shifting its phase by 180 degrees for each '1' in the binary data. BPSK is widely used in image transmission because it is simple to implement and provides good error performance. However, it is not suitable for transmitting high-speed data or images with complex color information.")
        st.write("2. QPSK (Quadrature Phase-Shift Keying): QPSK is another digital modulation technique that modulates the carrier signal by shifting its phase by 90 degrees for each pair of binary digits. QPSK is more complex than BPSK but can transmit data at a higher rate and is better suited for images with complex color information.")
        st.write("AWGN (Additive White Gaussian Noise): AWGN is a common channel model used in image transmission that simulates the effects of random noise in the communication channel. The noise is assumed to be Gaussian and has a uniform power spectral density. AWGN is widely used because it is simple to model and provides a good approximation of the noise in many real-world communication systems.")
        st.write("""The received signal in an AWGN channel can be modeled as:
                    y(t) = x(t) + n(t)
                    where y(t) is the received signal at time t, x(t) is the transmitted signal at time t, and n(t) is the Gaussian noise signal at time t with zero mean and variance σ².""")
        st.write("Rayleigh fading: Rayleigh fading is a channel model that simulates the effects of multipath propagation in wireless communication systems. It assumes that the signal experiences random amplitude and phase variations as it travels through the channel due to reflections from surrounding objects. Rayleigh fading is common in urban environments and can have a significant impact on image transmission performance.")
        st.write("""The received signal in a Rayleigh fading channel can be modeled as:
                    y(t) = h(t)x(t) + n(t)
                    where y(t) is the received signal at time t, x(t) is the transmitted signal at time t, h(t) is the complex channel gain at time t, and n(t) is the Gaussian noise signal at time t with zero mean and variance σ².
                    The channel gain h(t) is modeled as a complex Gaussian random variable with zero mean and variance σh². The magnitude of h(t) follows a Rayleigh distribution, and its phase follows a uniform distribution.""")
        st.write("Rician fading: Rician fading is a variation of Rayleigh fading that assumes a dominant line-of-sight path in addition to the reflected paths. This can occur in environments where there is a clear line of sight between the transmitter and receiver, such as in satellite communication systems. Rician fading can improve image transmission performance compared to Rayleigh fading, but it requires more complex modeling and processing.")
        st.write("""The received signal in a Rician fading channel can be modeled as:
                    y(t) = h(t)x(t) + n(t)
                    where y(t) is the received signal at time t, x(t) is the transmitted signal at time t, h(t) is the complex channel gain at time t, and n(t) is the Gaussian noise signal at time t with zero mean and variance σ².
                    The channel gain h(t) is modeled as a sum of a dominant line-of-sight component and a scattered component, both of which are modeled as complex Gaussian random variables with zero mean and variances σl² and σs², respectively. The magnitude of the channel gain follows a Rician distribution with a parameter K = σl²/σs².""")
        st.write("")
with st.container():
    st.subheader("Procedure")
    with st.expander("Procedure"):
        st.write("1. Load the sample image file and display it using Streamlit.")
        st.write(
            "2. Convert the image to a matrix of RGB values and a binary matrix.")
        st.write("3. Apply Hamming (7,4) encoding to the binary matrix.")
        st.write(
            "4. Prompt the user to select a modulation technique (BPSK or QPSK).")
        st.write("5. Apply the selected modulation technique to the encoded matrix.")
        st.write(
            "6. Prompt the user to select a channel model (AWGN, Rayleigh, or Rician).")
        st.write(
            "7. If AWGN is selected, add noise to the modulated signal using the selected SNR value.")
        st.write("8. If Rayleigh or Rician is selected, simulate a fading channel using the selected number of antennas and K factor.")
        st.write(
            "9. If Rayleigh is selected with selection diversity, apply selection diversity to the received signal.")
        st.write(
            "10. Demodulate the received signal using the selected modulation technique.")
        st.write("11. Apply Hamming (7,4) decoding to the demodulated matrix.")
        st.write("12. Convert the resulting binary matrix to a matrix of RGB values and display the demodulated image using Streamlit.")

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
    with st.expander("Image Transmission"):
        st.subheader(
            "Image Transmission with Modulation Techniques and Channel Models")
        image_file = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png"])
if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Original Image", use_column_width=True)

    # Convert image to matrix of RGB values
    matrix = image_to_matrix(image)
    binary_matrix = matrix_to_binary(matrix)
# Reshape binary matrix into blocks of length k=4
    k = 4
    n_blocks = int(np.ceil(binary_matrix.size/k))
    binary_matrix_padded = np.zeros(n_blocks*k, dtype=int)
    binary_matrix_padded[:binary_matrix.size] = binary_matrix.ravel()
    binary_matrix_padded = binary_matrix_padded.reshape(-1, k)

    # Apply Hamming (7,4) encoding
    codeword_matrix = hamming_encode(binary_matrix_padded)

    modulation_options = ["BPSK", "QPSK"]
    modulation = st.selectbox(
        "Select a modulation technique", modulation_options)

    if modulation == "BPSK":

        # Apply BPSK modulation
        modulated_matrix = bpsk_modulation(codeword_matrix)

    elif modulation == "QPSK":
        # Apply QPSK modulation
        modulated_matrix = qpsk_modulation(codeword_matrix)
    # elif modulation == "QFSK":
        # Apply QFSK modulation
        # modulated_matrix = qfsk_modulation(codeword_matrix)

    # Select SNR in dB
    snr_db = st.slider("Select SNR in dB", min_value=0,
                       max_value=30, step=1)

    # Select channel model
    n_antennas = 2

    # Select channel model
    channel_model_options = ["AWGN", "Rayleigh", "Rician"]
    channel_model = st.selectbox(
        "Select a channel model", channel_model_options)

    if channel_model == "AWGN":
        # Add AWGN to signal
        noisy_matrix = awgn_channel(modulated_matrix, snr_db)
    elif channel_model == "Rayleigh":
        mitigate = st.radio("Use Selection Diversity", ("No", "Yes"))
        if mitigate == "No":
            noisy_matrix = rayleigh_channel_without_mitigate(modulated_matrix)

        elif mitigate == "Yes":

            n_antennas = st.slider("Select number of antennas", min_value=2,
                                   max_value=4, step=1)

            # Add Rayleigh fading to signal
            noisy_matrix = rayleigh_channel_with_mitigate(
                modulated_matrix, n_antennas)
            # Apply selection diversity
            noisy_matrix = selection_diversity(noisy_matrix)

    elif channel_model == "Rician":
        # Select Rician K factor
        K = st.slider("Select Rician K factor", min_value=0,
                      max_value=10, step=1)
        # Add Rician fading to signal
        noisy_matrix = rician_channel(modulated_matrix, K)

    if modulation == "BPSK":

        # Apply BPSK demodulation
        demodulated_matrix = bpsk_demodulation(noisy_matrix)
        decoded_matrix = hamming_decode(demodulated_matrix)
        # Convert matrix of binary values to matrix of RGB values
        decoded_matrix = decoded_matrix[:, :binary_matrix.shape[1]]
        decoded_matrix = decoded_matrix.reshape(matrix.shape[:2] + (-1,))
        decoded_matrix = binary_to_matrix(decoded_matrix)
        demodulated_image = matrix_to_image(decoded_matrix)
    elif modulation == "QPSK":
        demodulated_matrix = qpsk_demodulation(noisy_matrix)
        demodulated_matrix = demodulated_matrix.reshape(-1, 7)
        decoded_matrix = hamming_decode(demodulated_matrix)
        decoded_matrix = decoded_matrix[:, :k].ravel()
        decoded_matrix = decoded_matrix[:binary_matrix.size]
        decoded_matrix = decoded_matrix.reshape(binary_matrix.shape)

        demodulated_image = matrix_to_image(binary_to_matrix(decoded_matrix))

    # elif modulation == "QFSK":

    # Apply Hamming (7,4) decoding

    # Display demodulated image
    st.image(demodulated_image, caption="Demodulated Image",
             use_column_width=True)


with st.container():
    st.subheader("Quiz")
    with st.expander("Quiz"):
        st.write("1. What is the difference between BPSK and QPSK modulation techniques? Under what circumstances would you choose one over the other for image transmission?")
        st.write("2. What is the effect of noise on image transmission performance? How can you simulate noise in a communication channel?")
        st.write("3. What is Rayleigh fading, and how does it affect image transmission performance? How can you model Rayleigh fading in a communication channel?")
        st.write("4. What is Rician fading, and how does it differ from Rayleigh fading? Under what circumstances would you expect to encounter Rician fading in a communication channel?")
        st.write("5. How can error correction codes be used to improve image transmission performance in the presence of noise and other channel impairments?")
        st.write("6. What is the impact of modulation and channel coding on transmission efficiency and bandwidth requirements? How can you optimize these parameters for a given image transmission system?")
        st.write("7. How can compression techniques be used to reduce the size of an image before transmission? What are the trade-offs between compression efficiency and image quality?")
        st.write("8. What are some real-world applications of image transmission with modulation techniques and channel models, and how do they differ from laboratory experiments? What are some of the practical challenges that arise in these applications?")
with st.container():
    st.subheader("References")
    with st.expander("References"):
        st.subheader("References")
        st.write("1. 'Digital Image Processing' by Rafael C. Gonzalez and Richard E. Woods - This textbook provides a comprehensive introduction to digital image processing, covering topics such as image enhancement, restoration, compression, and segmentation.")
        st.write(
            "2. Proakis, J. G., & Salehi, M. (2008). Digital Communications (5th ed.). New York, NY: McGraw-Hill.")
        st.write(
            "3. Haykin, S. (2009). Communication Systems (5th ed.). Hoboken, NJ: Wiley.")
        st.write("4. Sklar, B. (2001). Digital Communications: Fundamentals and Applications (2nd ed.). Upper Saddle River, NJ: Prentice Hall.")
