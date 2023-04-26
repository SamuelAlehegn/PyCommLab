import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Non Deterministic Source",  layout="wide",
                   initial_sidebar_state="auto")

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

    st.write("1. Analog to Digital Conversion")

    # column_one, column_two = st.columns(2)
    # with column_one:
    with st.expander("PULSE CODE MODULATION"):
        # INPUT
        signal_Type = st.selectbox("Signal Type", ("Sine", "Cosine"))
        amplitude = st.slider(
            "Amplitude", min_value=1, max_value=10, step=1)
        frequency = st.slider("Frequency", min_value=1,
                              max_value=10, step=1)
        theta = st.slider("Theta", min_value=0, max_value=360, step=1)
        sampling_frequency = st.slider(
            "Sampling Frequency", min_value=1, max_value=5, step=1)

        bit_rate = st.slider("Bit Rate", min_value=1,
                             max_value=100, step=1)
        bit_depth = st.slider(
            "Bit Depth", min_value=1, max_value=100, step=1)
        # bit_rate = st.slider("Bit Rate", min_value=1, max_value=100, step=1)
        # bit_depth = st.slider("Bit Depth", min_value=1, max_value=100, step=1)
        if st.button("Generate Signal"):
            if signal_Type == "Sine":
                t = np.arange(0, 2*np.pi, 0.01)
                x = amplitude*np.sin(2*np.pi*frequency*t+theta)
                fig = plt.plot(t, x)
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title("Sine Wave")
                plt.grid()
                st.pyplot()
            else:
                t = np.arange(0, 2*np.pi, 0.01)
                x = amplitude*np.cos(2*np.pi*frequency*t+theta)
                plt.plot(t, x)
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title("Cosine Wave")
                plt.grid()
                st.pyplot()

# Sampling
            st.write("I. Sampling")
            st.write("Sampling Frequency: ", sampling_frequency)
            sampling_period = 1/sampling_frequency
            st.write("Sampling Period: ", sampling_period)

            if signal_Type == "Sine":

                n = np.arange(0, 2*np.pi/sampling_period, 1)
                sampled_signal = amplitude * \
                    np.sin(2*np.pi*frequency*n*sampling_period + theta)
                plt.stem(n*sampling_period, sampled_signal)
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title("Sampled Signal")
                plt.grid()
                st.pyplot()
            else:

                n = np.arange(0, 2*np.pi, 1/sampling_frequency)
                plt.stem(n, x[::int(1/sampling_frequency)])
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title("Sampled Signal")
                plt.grid()
                st.pyplot()

            # Set the sampling frequency and duration
            # fs = 44100  # Hz
            # duration = 5  # seconds

            # # Generate a sine wave signal with a frequency of 440 Hz
            # freq = 440  # Hz
            # t = np.linspace(0, duration, int(fs * duration), endpoint=False)
            # signal = np.sin(2 * np.pi * freq * t)

            # # Plot the signal
            # plt.plot(t, signal)
            # plt.title('Sampled Signal')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            # plt.show()
# Quantization

            # Set the sampling frequency and duration
            # sampling_frequency = 44100  # Hz
            # duration = 5  # seconds

            # # Generate a sine wave signal with a frequency of 440 Hz
            # # freq = 440  # Hz
            # t = np.linspace(0, duration, int(
            #     sampling_frequency * duration), endpoint=False)
            # signal = np.sin(2 * np.pi * frequency * t)

            # # Quantize the signal using PCM with 8 bits
            # n_bits = 8
            # quantized_signal = np.round(signal * (2**(n_bits-1)-1))

            # # Plot the quantized signal
            # plt.plot(t, quantized_signal)
            # plt.title('Quantized Signal')
            # plt.xlabel('Time (s)')
            # plt.ylabel('Amplitude')
            # # plt.pyplot()
            # st.pyplot()

            st.write("II. Quantization")
            st.write("Bit Rate: ", bit_rate)
            st.write("Bit Depth: ", bit_depth)
            quantization_levels = 2**bit_depth
            st.write("Quantization Levels: ", quantization_levels)
            quantization_interval = 2*amplitude/quantization_levels
            st.write("Quantization Interval: ", quantization_interval)
            quantization_error = quantization_interval/2
            st.write("Quantization Error: ", quantization_error)
            quantized_signal = np.round(sampled_signal/quantization_interval)

            # quantized_signal = np.round(x * (2**(bit_depth-1)-1))

            # st.write("Quantized Signal: ", quantized_signal)
            plt.stem(n*sampling_period, quantized_signal)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Quantized Signal")
            plt.grid()
            st.pyplot()

# Encoding
            st.write("III. Encoding")

            encoded_signal = np.zeros_like(quantized_signal, dtype=int)
            for i in range(len(quantized_signal)):
                if i == 0:
                    encoded_signal[i] = quantized_signal[i]
                else:
                    encoded_signal[i] = quantized_signal[i] - \
                        quantized_signal[i-1]

            # Print the encoded signal
            print(encoded_signal)
            st.write("Encoded Signal: ", encoded_signal)

            # st.write("Encoding Bit Rate: ", bit_rate)
            # st.write("Encoding Bit Depth: ", bit_depth)
            # encoded_signal = np.binary_repr(quantized_signal, width=bit_depth)
            # st.write("Encoded Signal: ", encoded_signal)
            plt.stem(n*sampling_period, encoded_signal)
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title("Encoded Signal")
            plt.grid()
            st.pyplot()

    # with column_two:
    #     with st.expander("DELTA MODULATION"):
    #         st.subheader
    st.write("2. Convolution")
    with st.expander("Convolution"):
        st.subheader("Convolution")
        convolution_type = st.selectbox(
            "Convolution Type", ("Discrete Time", "Continuous Time"))
        if convolution_type == "Discrete Time":
            st.write("Discrete Time Convolution")
        else:
            st.write("Continuous Time Convolution")

    st.write("3. Modulation")
    with st.expander("Modulation"):
        st.write("Modulation")
    st.write("4. Filter")
    with st.expander("Filter"):
        st.write("Filter")


with st.container():
    st.subheader("Quize")
    with st.expander("Quize"):
        st.write("Quize")
with st.container():
    st.subheader("References")
    with st.expander("References"):
        st.write("References")
