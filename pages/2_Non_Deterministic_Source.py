import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Non Deterministic Source",  layout="wide",
                   initial_sidebar_state="auto")

with st.expander("Introduction"):
    st.write("The non deterministic source")
with st.expander("Theory"):
    st.subheader
with st.expander("Procedure"):
    st.subheader

with st.container():
    st.write("Simulations")
    st.write("1. Analog to Digital Conversion")

    column_one, column_two = st.columns(2)
    with column_one:
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
                # st.write("I. Sampling")
                # st.write("Sampling Frequency: ", sampling_frequency)
                # sampling_period = 1/sampling_frequency
                # st.write("Sampling Period: ", sampling_period)

                # if signal_Type == "Sine":

                #     n = np.arange(0, 2*np.pi/sampling_period, 1)
                #     sampled_signal = amplitude * \
                #         np.sin(2*np.pi*frequency*n*sampling_period + theta)
                #     plt.stem(n*sampling_period, sampled_signal)
                #     plt.xlabel("Time")
                #     plt.ylabel("Amplitude")
                #     plt.title("Sampled Signal")
                #     plt.grid()
                #     st.pyplot()
                # else:

                #     n = np.arange(0, 2*np.pi, 1/sampling_frequency)
                #     plt.stem(n, x[::int(1/sampling_frequency)])
                #     plt.xlabel("Time")
                #     plt.ylabel("Amplitude")
                #     plt.title("Sampled Signal")
                #     plt.grid()
                #     st.pyplot()

    with column_two:
        with st.expander("DELTA MODULATION"):
            st.subheader


with st.expander("Quize"):
    st.subheader
with st.expander("References"):
    st.subheader
