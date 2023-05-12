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

            def generate_signal(fc, fm):
                t = np.linspace(0, 1, 1000)
                m = np.sin(2*np.pi*fm*t)
                return t, m

            def dsb_fc_modulation(t, m, fc):
                dsb_fc = m * np.cos(2*np.pi*fc*t)
                return dsb_fc

            def ssb_sc_modulation(t, m, fc):
                ssb_sc = signal.hilbert(m)
                ssb_sc_upper = np.real(ssb_sc * np.exp(1j*2*np.pi*fc*t))
                ssb_sc_lower = np.imag(ssb_sc * np.exp(1j*2*np.pi*fc*t))
                return ssb_sc_upper, ssb_sc_lower

            def ssb_fc_modulation(t, m, fc):
                ssb_fc = signal.hilbert(m)
                ssb_fc_upper = np.real(ssb_fc * np.exp(1j*2*np.pi*fc*t))
                ssb_fc_lower = -np.imag(ssb_fc * np.exp(1j*2*np.pi*fc*t))
                return ssb_fc_upper, ssb_fc_lower

            def vsb_modulation(t, m, fc):
                vsb = signal.firwin(101, 0.25)
                v = np.convolve(m, vsb, mode='same')
                vsb_upper = v * np.cos(2*np.pi*fc*t)
                vsb_lower = signal.hilbert(v) * np.sin(2*np.pi*fc*t)
                return vsb_upper, vsb_lower

            def plot_graphs(t, signal, spectrum, title):
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(title)
                axs[0].plot(t, signal)
                axs[0].set_title('Time-domain')
                axs[1].plot(spectrum[0], np.abs(spectrum[1]))
                axs[1].set_title('Frequency-domain')
                plt.tight_layout()
                return fig

            # # Define the carrier frequency and the message signal
            # fc = 1000  # carrier frequency
            # fm = 100  # message signal frequency
            # Accept user input for carrier and message signal frequencies
            fc = st.slider(
                'Carrier Frequency (Hz)', 100, 10000, 1000, 100)
            fm = st.slider('Message Frequency (Hz)', 10, 1000, 100, 10)

            # Generate the message signal
            t, m = generate_signal(fc, fm)

            # Show the message signal
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t, m)
            ax.set_title('Message Signal')
            plt.tight_layout()
            st.pyplot(fig)

            amplitude_modulation_type = st.selectbox(
                "Select Amplitude Modulation", ("Double-sideband a full-carrier", "Single-sideband suppressed-carrier", "Single-sideband full-carrier", "Vestigial sideband modulation"))
            if amplitude_modulation_type == "Double-sideband a full-carrier":
                # Double-sideband full carrier modulation
                dsb_fc = dsb_fc_modulation(t, m, fc)
                dsb_fc_spectrum = np.fft.fft(dsb_fc)
                dsb_fc_fig = plot_graphs(t, dsb_fc, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), dsb_fc_spectrum), 'DSB-FC')
                st.pyplot(dsb_fc_fig)

            elif amplitude_modulation_type == "Single-sideband suppressed-carrier":
                # Single-sideband suppressed carrier modulation
                ssb_sc_upper, ssb_sc_lower = ssb_sc_modulation(t, m, fc)
                ssb_sc_upper_spectrum = np.fft.fft(ssb_sc_upper)
                ssb_sc_lower_spectrum = np.fft.fft(ssb_sc_lower)
                ssb_sc_fig = plot_graphs(t, ssb_sc_upper, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), ssb_sc_upper_spectrum), 'SSB-SC (Upper Sideband)')
                ssb_sc_fig2 = plot_graphs(t, ssb_sc_lower, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), ssb_sc_lower_spectrum), 'SSB-SC (Lower Sideband)')
                st.pyplot(ssb_sc_fig)
                st.pyplot(ssb_sc_fig2)

            elif amplitude_modulation_type == "Single-sideband full-carrier":
                # Single-sideband full carrier modulation
                ssb_fc_upper, ssb_fc_lower = ssb_fc_modulation(t, m, fc)
                ssb_fc_upper_spectrum = np.fft.fft(ssb_fc_upper)
                ssb_fc_lower_spectrum = np.fft.fft(ssb_fc_lower)
                ssb_fc_fig = plot_graphs(t, ssb_fc_upper, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), ssb_fc_upper_spectrum), 'SSB-FC (Upper Sideband)')
                ssb_fc_fig2 = plot_graphs(t, ssb_fc_lower, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), ssb_fc_lower_spectrum), 'SSB-FC (Lower Sideband)')
                st.pyplot(ssb_fc_fig)
                st.pyplot(ssb_fc_fig2)

            elif amplitude_modulation_type == "Vestigial sideband modulation":
                # Vestigial sideband modulation
                vsb_upper, vsb_lower = vsb_modulation(t, m, fc)
                vsb_upper_spectrum = np.fft.fft(vsb_upper)
                vsb_lower_spectrum = np.fft.fft(vsb_lower)
                vsb_fig = plot_graphs(t, vsb_upper, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), vsb_upper_spectrum), 'VSB (Upper Sideband)')
                vsb_fig2 = plot_graphs(t, vsb_lower, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), vsb_lower_spectrum), 'VSB (Lower Sideband)')
                st.pyplot(vsb_fig)
                st.pyplot(vsb_fig2)

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
