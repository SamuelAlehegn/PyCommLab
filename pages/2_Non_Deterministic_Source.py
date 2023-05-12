import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal


st.set_page_config(page_title="Non Deterministic Source",  layout="wide",
                   initial_sidebar_state="auto", )
st.set_option('deprecation.showPyplotGlobalUse', False)

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
                sampled_signal = amplitude * \
                    np.cos(2*np.pi*frequency*n*sampling_period + theta)
                # plt.stem(n, x[::int(1/sampling_frequency)])
                plt.xlabel("Time")
                plt.ylabel("Amplitude")
                plt.title("Sampled Signal")
                plt.grid()
                st.pyplot()

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
        st.subheader("Modulation")

        modulation_Type = st.selectbox(
            "Modulation Type", ("Amplitude modulation", "Angle modulation"))
        if modulation_Type == "Amplitude modulation":
            st.write("Amplitude Modulation")
            amplitude_modulation_type = st.selectbox("Select Amplitude Modulation Type", (
                "Double Sideband Suppressed Carrier(DSB SC)", "Double side-band full carrier", "Single sideband (SSB)", "Vestigial sideband modulation"))
            if amplitude_modulation_type == "Double Sideband Suppressed Carrier(DSB SC)":
                st.write("Double Sideband Suppressed Carrier(DSB SC)")

                # Define the carrier signal
                def carrier_signal(freq, duration, sampling_rate):
                    t = np.linspace(0, duration, sampling_rate *
                                    duration, endpoint=False)
                    return np.sin(2 * np.pi * freq * t)

                # Define the message signal
                def message_signal(freq, duration, sampling_rate):
                    t = np.linspace(0, duration, sampling_rate *
                                    duration, endpoint=False)
                    return np.sin(2 * np.pi * freq * t)

                # Define DSB-SC modulation function
                def dsb_sc_modulation(message, carrier):
                    return message * carrier

                # Streamlit interface
                st.title('Double Sideband Suppressed Carrier Modulation')

                # Get user inputs
                message_freq = st.number_input(
                    'Message signal frequency', value=10)
                carrier_freq = st.number_input(
                    'Carrier signal frequency', value=100)
                duration = st.number_input(
                    'Signal duration (seconds)', value=1)
                sampling_rate = st.number_input(
                    'Sampling rate (samples per second)', value=44100)

                # Generate the signals
                message = message_signal(message_freq, duration, sampling_rate)
                carrier = carrier_signal(carrier_freq, duration, sampling_rate)
                modulated_signal = dsb_sc_modulation(message, carrier)

                time_column, frequency_column = st.columns(2)

                with time_column:
                    # Time domain plots
                    fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8))
                    axs1[0].plot(message)
                    axs1[0].set_title('Message signal')
                    axs1[1].plot(carrier)
                    axs1[1].set_title('Carrier signal')
                    axs1[2].plot(modulated_signal)
                    axs1[2].set_title('Modulated signal')
                    plt.tight_layout()
                    st.pyplot(fig1)
                with frequency_column:
                    # Frequency domain plots
                    fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8))
                    f = np.fft.fftshift(np.fft.fftfreq(
                        len(message), 1/sampling_rate))

                    axs2[0].magnitude_spectrum(message, Fs=sampling_rate)
                    axs2[0].set_title('Message signal frequency spectrum')
                    axs2[1].magnitude_spectrum(carrier, Fs=sampling_rate)
                    axs2[1].set_title('Carrier signal frequency spectrum')
                    axs2[2].magnitude_spectrum(
                        modulated_signal, Fs=sampling_rate)
                    axs2[2].set_title('Modulated signal frequency spectrum')
                    plt.tight_layout()
                    st.pyplot(fig2)

            elif amplitude_modulation_type == "Double side-band full carrier":
                st.write("Double side-band full carrier")
                # Define the DSB-FC signal generation function

                def dsbfc_signal(fc, fm, m, duration, fs):
                    t = np.arange(0, duration, 1/fs)
                    carrier = np.cos(2*np.pi*fc*t)
                    message = np.cos(2*np.pi*fm*t)
                    dsbfc = (1 + m*message)*carrier
                    return t, dsbfc

                # Create Streamlit app
                st.title("DSB-FC Signal Generator")

                # User input for carrier frequency
                fc = st.number_input(
                    "Enter carrier frequency (Hz):", value=1000, step=100)

                # User input for message signal frequency
                fm = st.number_input(
                    "Enter message signal frequency (Hz):", value=100, step=10)

                # User input for modulation index
                m = st.number_input(
                    "Enter modulation index:", value=0.5, step=0.1)

                duration = st.number_input(
                    "Enter duration of signal (seconds):", value=1)

                fs = st.number_input(
                    "Enter sampling frequency (Hz):", value=10000, step=1000)

                t, dsbfc = dsbfc_signal(fc, fm, m, duration, fs)

                # Plot the time-domain signal
                fig1, ax1 = plt.subplots()
                ax1.plot(t, dsbfc)
                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Amplitude")
                ax1.set_title("DSB-FC Signal (Time Domain)")
                st.pyplot(fig1)

                # Plot the frequency-domain signal
                fig2, ax2 = plt.subplots()
                f = np.fft.fftfreq(len(dsbfc), 1/fs)
                Y = np.fft.fft(dsbfc)
                ax2.plot(f, np.abs(Y))
                ax2.set_xlabel("Frequency (Hz)")
                ax2.set_ylabel("Magnitude")
                ax2.set_title("DSB-FC Signal (Frequency Domain)")
                st.pyplot(fig2)

            elif amplitude_modulation_type == "Single sideband (SSB)":
                st.write("Single sideband (SSB)")
            elif amplitude_modulation_type == "Vestigial sideband modulation":
                st.write("Vestigial sideband modulation")

        elif modulation_Type == "Angle modulation":
            st.write("Angle modulation")
            angle_modulation_type = st.selectbox(
                "Select Angle Modulation", ("Frequency Modulation", "Phase Modulation"))
            if angle_modulation_type == "Frequency Modulation":
                st.write("Frequency Modulation")
            elif angle_modulation_type == "Phase Modulation":
                st.write("Phase Modulation")


st.write("4. Filter")
with st.expander("Filter"):
    st.write("Filter")

    def plot_graphs(x, y, xlabel, ylabel, title):
        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(x, y, 'b-')
        axs[0].set_xlabel(xlabel)
        axs[0].set_ylabel(ylabel)
        axs[0].set_title(title)

        Y = np.fft.fft(y)
        freq = np.fft.fftfreq(len(y), d=1 / 1000)
        axs[1].plot(freq, np.abs(Y), 'r')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Magnitude')
        axs[1].set_xlim([0, 500])
        axs[1].set_ylim([0, 1.2 * np.max(np.abs(Y))])
        axs[1].set_title('Frequency Domain')

        st.pyplot(fig)

    def lowpass_filter(cutoff_freq, x, y):
        b, a = signal.butter(5, cutoff_freq, 'low', fs=1000)
        filtered = signal.filtfilt(b, a, y)
        plot_graphs(x, y, 'Time (ms)', 'Amplitude', 'Low Pass Filter')
        plot_graphs(x, filtered, 'Time (ms)', 'Amplitude',
                    'Low Pass Filter (Filtered)')

    def highpass_filter(cutoff_freq, x, y):
        b, a = signal.butter(5, cutoff_freq, 'high', fs=1000)
        filtered = signal.filtfilt(b, a, y)
        plot_graphs(x, y, 'Time (ms)', 'Amplitude', 'High Pass Filter')
        plot_graphs(x, filtered, 'Time (ms)', 'Amplitude',
                    'High Pass Filter (Filtered)')

    def bandpass_filter(low_cutoff_freq, high_cutoff_freq, x, y):
        b, a = signal.butter(
            5, [low_cutoff_freq, high_cutoff_freq], 'band', fs=1000)
        filtered = signal.filtfilt(b, a, y)
        plot_graphs(x, y, 'Time (ms)', 'Amplitude', 'Band Pass Filter')
        plot_graphs(x, filtered, 'Time (ms)', 'Amplitude',
                    'Band Pass Filter (Filtered)')

    def bandreject_filter(low_cutoff_freq, high_cutoff_freq, x, y):
        b, a = signal.butter(
            5, [low_cutoff_freq, high_cutoff_freq], 'bandstop', fs=1000)
        filtered = signal.filtfilt(b, a, y)
        plot_graphs(x, y, 'Time (ms)', 'Amplitude', 'Band Reject Filter')
        plot_graphs(x, filtered, 'Time (ms)', 'Amplitude',
                    'Band Reject Filter (Filtered)')

    st.subheader('Filter Implementation in Python and Streamlit')

    filter_type = st.selectbox('Select a filter:', ('Low Pass Filter',
                                                    'High Pass Filter', 'Band Pass Filter', 'Band Reject Filter'))

    if filter_type == 'Low Pass Filter':
        cutoff_freq = st.slider('Cutoff Frequency (Hz)', 0, 500, 50)
        x = np.linspace(0, 1, 1000, endpoint=False)
        y = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 100 * x)
        lowpass_filter(cutoff_freq, x, y)

    elif filter_type == 'High Pass Filter':
        cutoff_freq = st.slider('Cutoff Frequency (Hz)', 0, 500, 50)
        x = np.linspace(0, 1, 1000, endpoint=False)
        y = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 100 * x)
        highpass_filter(cutoff_freq, x, y)

    elif filter_type == 'Band Pass Filter':
        low_cutoff_freq = st.slider(
            'Low Cutoff Frequency (Hz)', 0, 500, 50)
        high_cutoff_freq = st.slider(
            'High Cutoff Frequency (Hz)', low_cutoff_freq, 500, 100)
        x = np.linspace(0, 1, 1000, endpoint=False)
        y = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 100 * x)
        bandpass_filter(low_cutoff_freq, high_cutoff_freq, x, y)

    elif filter_type == 'Band Reject Filter':
        low_cutoff_freq = st.slider(
            'Low Cutoff Frequency (Hz)', 0, 500, 50)
        high_cutoff_freq = st.slider(
            'High Cutoff Frequency (Hz)', low_cutoff_freq, 500, 100)
        x = np.linspace(0, 1, 1000, endpoint=False)
        y = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 100 * x)
        bandreject_filter(low_cutoff_freq, high_cutoff_freq, x, y)

with st.container():
    st.subheader("Quize")
    with st.expander("Quize"):
        st.write("Quize")
with st.container():
    st.subheader("References")
    with st.expander("References"):
        st.write("References")
