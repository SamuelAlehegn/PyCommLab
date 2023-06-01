import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
# import scipy.signal as signal
from scipy.signal import convolve
from scipy import signal

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

        st.subheader("Pulse Code Modulation")
        t = np.linspace(0, 1, 1020)
        f = 5
        signal = np.sin(2 * np.pi * f * t)
        # Sampling
        sampling_freq = st.slider("Sampling Frequency", 20, 200, 100)
        sampling_period = 1 / sampling_freq
        samples = signal[::int(sampling_period * 1002)]  # Modified line
        # st.write("Sampled Signal:", samples)

        # Quantization
        num_bits = st.slider("Number of Quantization Bits", 1, 8, 4)
        quantization_levels = 2**num_bits
        quantization_step = (
            np.max(samples) - np.min(samples)) / quantization_levels
        quantized_samples = np.round(
            samples / quantization_step) * quantization_step
        st.write("Quantized Signal:", quantized_samples)

        # Encoding
        encoded_samples = np.uint8(
            (quantized_samples - np.min(quantized_samples)) / quantization_step)
        st.write("Encoded Signal:", encoded_samples)

        # Decoding
        decoded_samples = encoded_samples * \
            quantization_step + np.min(quantized_samples)

        # Reconstruction
        reconstructed_signal = np.repeat(decoded_samples, int(
            sampling_period * 1002))  # Modified line

        # Plot the results
        fig, ax = plt.subplots(4, 1, figsize=(8, 10))
        ax[0].plot(t, signal)
        ax[0].set_title("Original Signal")
        ax[1].stem(samples, use_line_collection=True)
        ax[1].set_title("Sampled Signal")
        ax[2].stem(quantized_samples, use_line_collection=True)
        ax[2].set_title("Quantized Signal")
        ax[3].stem(encoded_samples, use_line_collection=True)
        ax[3].set_title("Encoded Signal")
        st.pyplot(fig)

        # Plot the reconstructed signal
        fig, ax = plt.subplots(1, 1)
        ax.plot(t, signal, label="Original Signal")
        ax.plot(t, reconstructed_signal, label="Reconstructed Signal")
        ax.legend()
        ax.set_title("Reconstructed Signal vs Original Signal")
        st.pyplot(fig)

    st.write("2. Convolution")
    with st.expander("Convolution"):
        st.subheader("Convolution")

        def continuous_time_convolution():
            st.title("Continuous-Time Convolution")

            # Define the input signals
            t = np.linspace(-10, 10, 1000)
            x = st.selectbox(
                "Input Signal", ["exp(-t^2)", "sin(t)", "cos(t)", "rect(t)", "tri(t)"])
            if x == "exp(-t^2)":
                x = np.exp(-t**2)
            elif x == "sin(t)":
                x = np.sin(t)
            elif x == "cos(t)":
                x = np.cos(t)
            elif x == "rect(t)":
                x = np.where(np.abs(t) < 1, 1, 0)
            elif x == "tri(t)":
                x = np.where(t < 0, 0, 1-t)
                x = np.where(t > 1, 0, x)

            h = st.selectbox("Impulse Response", [
                "exp(-t^2)", "sin(t)", "cos(t)", "rect(t)", "tri(t)"])
            if h == "exp(-t^2)":
                h = np.exp(-t**2)
            elif h == "sin(t)":
                h = np.sin(t)
            elif h == "cos(t)":
                h = np.cos(t)
            elif h == "rect(t)":
                h = np.where(np.abs(t) < 1, 1, 0)
            elif h == "tri(t)":
                h = np.where(t < 0, 0, 1-t)
                h = np.where(t > 1, 0, h)

            # Perform the convolution
            y = convolve(x, h, mode='same') / len(t)

            # Plot the signals and the convolution result
            fig, ax = plt.subplots(3, 1, figsize=(8, 8))

            ax[0].plot(t, x)
            ax[0].set_title("Input Signal x(t)")

            ax[1].plot(t, h)
            ax[1].set_title("Impulse Response h(t)")

            ax[2].plot(t, y)
            ax[2].set_title("Convolution Result y(t)")

            st.pyplot(fig)

        def discrete_time_convolution():
            st.title("Discrete-Time Convolution")

            # Define the input signals
            n = np.arange(-10, 11)
            x = st.selectbox(
                "Input Signal", ["exp(-n^2/10)", "sin(n)", "cos(n)", "rect(n)", "tri(n)"])
            if x == "exp(-n^2/10)":
                x = np.exp(-n**2/10)
            elif x == "sin(n)":
                x = np.sin(n)
            elif x == "cos(n)":
                x = np.cos(n)
            elif x == "rect(n)":
                x = np.where(np.abs(n) < 1, 1, 0)
            elif x == "tri(n)":
                x = np.where(n < 0, 0, 1-n)
                x = np.where(n > 1, 0, x)

            h = st.selectbox("Impulse Response", [
                "exp(-n^2/10)", "sin(n)", "cos(n)", "rect(n)", "tri(n)"])
            if h == "exp(-n^2/10)":
                h = np.exp(-n**2/10)
            elif h == "sin(n)":
                h = np.sin(n)
            elif h == "cos(n)":
                h = np.cos(n)
            elif h == "rect(n)":
                h = np.where(np.abs(n) < 1, 1, 0)
            elif h == "tri(n)":
                h = np.where(n < 0, 0, 1-n)
                h = np.where(n > 1, 0, h)

            # Perform the convolution
            y = np.convolve(x, h, mode='same') / len(n)

            # Plot the signals and the convolution result
            fig, ax = plt.subplots(3, 1, figsize=(8, 8))

            ax[0].stem(n, x)
            ax[0].set_title("Input Signal x[n]")

            ax[1].stem(n, h)
            ax[1].set_title("Impulse Response h[n]")

            ax[2].stem(n, y)
            ax[2].set_title("Convolution Result y[n]")

            st.pyplot(fig)

        convolution_type = st.selectbox(
            "Convolution Type", ("Discrete-Time Convolution", "Continuous-Time Convolution"))

        if convolution_type == "Continuous-Time Convolution":
            continuous_time_convolution()
        elif convolution_type == "Discrete-Time Convolution":
            discrete_time_convolution()

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

            def generate_signal(fm):
                t = np.linspace(0, 1, 1000)
                m = np.sin(2*np.pi*fm*t)
                return t, m

            def freq_modulation(t, m, fc, kf):
                freq_mod = np.cos(2*np.pi*fc*t + 2*np.pi*kf*np.cumsum(m))
                return freq_mod

            def phase_modulation(t, m, fc, kp):
                phase_mod = np.cos(2*np.pi*fc*t + kp*m)
                return phase_mod

            def plot_graphs(t, signal, spectrum, title):
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                fig.suptitle(title)
                axs[0].plot(t, signal)
                axs[0].set_title('Time-domain')
                axs[1].plot(spectrum[0], np.abs(spectrum[1]))
                axs[1].set_title('Frequency-domain')
                plt.tight_layout()
                return fig

            # Accept user input for message signal frequency and modulation parameters
            fm = st.slider('Message Frequency (Hz)', 10, 1000, 100, 10)
            fc = st.slider(
                'Carrier Frequency (Hz)', 100, 10000, 1000, 100)
            kf = st.slider(
                'Frequency Modulation Index', 0.1, 10.0, 2.0, 0.1)
            kp = st.slider(
                'Phase Modulation Index', 0.1, 10.0, 2.0, 0.1)

            # Generate the message signal
            t, m = generate_signal(fm)

            # Show the message signal
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(t, m)
            ax.set_title('Message Signal')
            plt.tight_layout()
            st.pyplot(fig)

            angle_modulation_type = st.selectbox(
                "Select Angle Modulation", ("Frequency Modulation", "Phase Modulation"))
            if angle_modulation_type == "Frequency Modulation":
                st.write("Frequency Modulation")
                # Implement frequency modulation
                freq_mod = freq_modulation(t, m, fc, kf)
                freq_mod_spectrum = np.fft.fft(freq_mod)
                freq_mod_fig = plot_graphs(t, freq_mod, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), freq_mod_spectrum), 'Frequency Modulation')
                st.pyplot(freq_mod_fig)
            elif angle_modulation_type == "Phase Modulation":
                st.write("Phase Modulation")

                # Implement phase modulation
                phase_mod = phase_modulation(t, m, fc, kp)
                phase_mod_spectrum = np.fft.fft(phase_mod)
                phase_mod_fig = plot_graphs(t, phase_mod, (np.fft.fftfreq(
                    t.shape[-1], 1/1000), phase_mod_spectrum), 'Phase Modulation')
                st.pyplot(phase_mod_fig)


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
