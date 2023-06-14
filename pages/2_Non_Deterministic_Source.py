import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
# import scipy.signal as signal
from scipy.signal import convolve
from scipy import signal

st.set_page_config(page_title="Non Deterministic Source",  layout="wide",
                   initial_sidebar_state="auto", )
st.set_option('deprecation.showPyplotGlobalUse', False)

with st.container():
    st.subheader("Introduction")
    with st.expander("Introduction"):
        st.write("The main purpose of this website page is to teach beginners about different communication system concepts. Our goal is to provide an accessible and comprehensive resource for people who are new to the field of communication systems and want to learn about topics like PCM, convolution, modulation, filter, and line coding. By breaking down these complex concepts into simple and easy-to-understand explanations, we hope to demystify the world of communication systems and help beginners develop a strong foundation in this area. So whether you're interested in pursuing a career in telecommunications, or you simply want to expand your knowledge of how information is transmitted over networks, this website page is the perfect starting point.")
with st.container():
    st.header("Theory")
    with st.expander("Pulse Code Modulation"):
        st.subheader("Pulse Code Modulation")
        st.write(
            "Signals can be divided into two main categories: analog signals and digital signals.")
        st.write("Analog signals are continuous signals that vary in amplitude and/or frequency over time. Examples of analog signals include sound waves, light waves, and voltage signals. In an analog signal, the amplitude or frequency of the signal varies smoothly and continuously over time, which means that it can take on an infinite number of values.")
        st.write("Digital signals, on the other hand, are discrete signals that take on a finite number of values. Digital signals are represented using binary code, which means that they consist of a series of 1s and 0s. Digital signals can be thought of as a series of on/off switches, where each switch represents a bit of information.")
        st.write("In the context of communication systems, analog signals are typically converted into digital signals using modulation techniques such as Pulse Code Modulation (PCM), while digital signals are converted back into analog signals using demodulation techniques.")
        st.write("Pulse Code Modulation (PCM) is a digital modulation technique used to convert analog signals into digital signals. In PCM, the analog signal is sampled at regular intervals, and each sample is then quantized to a specific level. The quantized samples are then encoded into a digital signal for transmission or storage.")
        st.write("The process of PCM involves three main steps: sampling, quantization, and coding. In the sampling step, the analog signal is converted into a sequence of discrete samples by measuring the amplitude of the signal at regular intervals. The sampling rate determines how often the signal is sampled, and it is typically expressed in Hz.")
        st.write("In the quantization step, each sample is assigned to a specific quantization level. The number of quantization levels determines the resolution of the system and is typically expressed in bits. For example, an 8-bit PCM system has 256 quantization levels, while a 16-bit PCM system has 65,536 levels.")
        st.write("The quantization process introduces quantization noise, which is the difference between the actual sample value and the quantized value. The magnitude of the quantization noise depends on the quantization step size, which is calculated as the difference between the maximum and minimum sample values divided by the number of quantization levels.")
        st.write("In the coding step, the quantized samples are encoded into a digital signal using a binary code. The most common coding scheme used in PCM is the two's complement, which allows both positive and negative values to be represented using the same binary code.")
        st.write("The encoded signal can be transmitted over a digital communication channel or stored on a digital storage medium. To reconstruct the original analog signal from the PCM signal, the decoding process is used, which involves decoding the binary code and then applying the inverse quantization and sampling processes.")
        st.write("PCM is widely used in digital audio and video applications, where it is used to convert analog signals into digital signals for storage or transmission. PCM provides high fidelity and accuracy, making it a suitable choice for applications where high-quality sound or images are required.")
        st.write("Aliasing is a phenomenon that can occur when an analog signal is sampled at a rate that is too low to accurately capture its frequency content. In PCM, aliasing can cause distortion, loss of information, and other artifacts in the reconstructed signal. To avoid aliasing, the sampling rate must be high enough to satisfy the Nyquist criterion, and anti-aliasing filters can be used to remove high-frequency components above the Nyquist frequency")
        st.write("The Nyquist frequency is the maximum frequency that can be accurately represented by a digital signal, and it is equal to half the sampling rate. In other words, the Nyquist frequency sets the upper limit for the frequency content that can be accurately captured by the digital signal. If the analog signal contains frequency components that are higher than the Nyquist frequency, those components will be aliased, or folded back, into the frequency range that can be accurately represented by the samples. Therefore, it is important to sample the analog signal at a rate that is at least twice the highest frequency component in order to avoid aliasing.")
        st.write("")

    with st.expander("Convolution"):
        st.subheader("Convolution")
        st.write("Convolution is a mathematical operation that is commonly used in digital signal processing to analyze and manipulate signals. In essence, convolution is a way of combining two signals to produce a third signal that represents how the first signal modifies the second signal as it passes through a system.")
        st.write("The basic idea behind convolution is to slide one signal (known as the input signal) over another signal (known as the impulse response or kernel) and calculate the integral of their product at each point. The result of this integration is the output signal, which represents the combined effect of the input signal and the impulse response.")
        st.write("Convolution is widely used in a variety of signal processing applications, including filtering, smoothing, and feature extraction. For example, convolution can be used to apply a filter to a signal in order to remove noise or to extract certain features from the signal.")
        st.write("In digital signal processing, convolution is typically performed using a discrete convolution operation, which involves multiplying corresponding samples of the input signal and the impulse response and summing the results. This operation can be implemented efficiently using the Fast Fourier Transform (FFT) algorithm, which allows convolution to be performed in the frequency domain.")
        st.write("")

    with st.expander("Modulation"):
        st.subheader("Modulation")
        st.write("Modulation is the process of changing the characteristics of a signal, known as the carrier signal, in order to transmit information. By modulating the carrier signal, we can encode information onto the signal and transmit it over a communication channel.")
        st.write("There are several types of modulation, including amplitude modulation (AM), frequency modulation (FM), phase modulation (PM), and more complex modulation schemes such as quadrature amplitude modulation (QAM) and orthogonal frequency-division multiplexing (OFDM).")
        st.write("In AM, the amplitude of the carrier signal is varied in proportion to the amplitude of the modulating signal, which contains the information to be transmitted. In FM, the frequency of the carrier signal is varied in proportion to the amplitude of the modulating signal. In PM, the phase of the carrier signal is varied in proportion to the amplitude of the modulating signal.")
        st.write("More complex modulation schemes, such as QAM and OFDM, use combinations of amplitude, frequency, and phase modulation to transmit multiple signals simultaneously over the same communication channel.")
        st.write("Modulation is used in a wide range of applications, including radio and television broadcasting, cellular communication, satellite communication, and digital audio and video transmission. In these applications, modulation allows information to be transmitted over long distances and through various types of media, while minimizing interference and signal degradation.")
        st.write("Modulation provides several benefits in the transmission and reception of information. Here are some of the key benefits of modulation:")
        st.write("Efficient use of bandwidth: By modulating the carrier signal, we can encode information onto the signal in a way that uses the available frequency spectrum efficiently. This means that multiple signals can be transmitted simultaneously over the same communication channel, without interfering with each other.")
        st.write("Resistance to noise and interference: Modulation can make the transmitted signal more resistant to noise and interference, which can degrade the signal quality. For example, in frequency modulation (FM), the signal is less affected by noise than in amplitude modulation (AM), which makes FM more suitable for high-quality audio transmission.")
        st.write("Flexibility: Modulation provides flexibility in terms of the type and amount of information that can be transmitted. Different modulation schemes can be used to transmit different types of information, such as voice, data, and video, and the amount of information that can be transmitted can be adjusted by changing the modulation parameters.")
        st.write("Compatibility: Modulation schemes are standardized, which means that devices from different manufacturers and different countries can communicate with each other as long as they use the same modulation scheme. This makes it easier to develop and use communication systems that are compatible with each other.")
        st.subheader("Amplitude Modulation (AM):")
        st.write("1. Double Sideband Full Carrier (DSB-FC) Modulation: In DSB-FC modulation, both upper and lower sidebands are transmitted along with the carrier signal. The advantage of DSB-FC modulation is that it is simple and easy to implement. The disadvantage is that it requires a lot of bandwidth and is susceptible to noise and interference. DSB-FC is mainly used in broadcasting.")
        st.write("2. Double Sideband Suppressed Carrier (DSB-SC) Modulation: In DSB-SC modulation, the carrier signal is suppressed, and only the upper and lower sidebands are transmitted. The advantage of DSB-SC modulation is that it requires less bandwidth than DSB-FC modulation. The disadvantage is that it requires a more complex demodulation process. DSB-SC is mainly used in radio communications.")
        st.write("3. Single Sideband (SSB) Modulation: In SSB modulation, only one of the sidebands is transmitted along with the carrier signal. The advantage of SSB modulation is that it requires less bandwidth than DSB-FC and DSB-SC modulation. The disadvantage is that it requires a more complex demodulation process. SSB is mainly used in amateur radio and military communications.")
        st.write("4. Vestigial Sideband (VSB) Modulation: In VSB modulation, one of the sidebands is suppressed, but a small portion of it is retained to allow for better demodulation. The advantage of VSB modulation is that it requires less bandwidth than DSB-FC modulation, but provides better performance than DSB-SC modulation. The disadvantage is that it requires a more complex demodulation process. VSB is mainly used in television broadcasting.")
        st.subheader("Angle Modulation:")
        st.write("1. Frequency Modulation (FM): In FM, the frequency of the carrier signal is varied in proportion to the amplitude of the modulating signal. The advantage of FM is that it is less susceptible to noise and interference than AM. The disadvantage is that it requires more bandwidth than AM. FM is mainly used in high-quality audio transmission, such as in radio broadcasting and music reproduction.")
        st.write("2. Frequency Modulation (FM): In FM, the frequency of the carrier signal is varied in proportion to the amplitude of the modulating signal. The advantage of FM is that it is less susceptible to noise and interference than AM. The disadvantage is that it requires more bandwidth than AM. FM is mainly used in high-quality audio transmission, such as in radio broadcasting and music reproduction.")
        st.write("")
        st.write("Modulation index: The modulation index is a parameter that determines the degree of modulation in an amplitude or angle modulation scheme. In AM, the modulation index is the ratio of the amplitude of the modulating signal to the amplitude of the carrier signal, while in FM and PM, the modulation index is the maximum deviation of the frequency or phase of the carrier signal from its unmodulated state. The modulation index affects the bandwidth and the quality of the transmitted signal.")
        st.write("Demodulation: Demodulation is the process of extracting the original modulating signal from the modulated carrier signal. Demodulation is necessary in order to recover the information that was encoded onto the carrier signal.")
        st.write("")
    with st.expander("Filter"):
        st.write("A filter is a fundamental tool in signal processing used to remove unwanted noise or frequency components from a signal. Filters can be implemented in both the time domain and the frequency domain. In the time domain, a filter operates on the samples of a signal in a sliding window fashion, while in the frequency domain, a filter operates on the Fourier transform of the signal.")
        st.write("Filters can be classified into two categories: infinite impulse response (IIR) filters and finite impulse response (FIR) filters. An IIR filter has feedback loops in its implementation, which allows it to have a much simpler design than an FIR filter. On the other hand, an FIR filter does not have feedback loops and is more stable than an IIR filter.")
        st.write("Filters can also be classified based on their frequency response")
        st.write("1. A low-pass filter is a type of filter that allows low-frequency components of a signal to pass through while attenuating high-frequency components. The magnitude response of a low-pass filter decreases as the frequency of the signal increases. Low-pass filters are commonly used in audio applications to eliminate high-frequency noise or to smooth out signals by removing high-frequency components. They are also used in control systems to eliminate high-frequency disturbances.")
        st.write("2. A high-pass filter, on the other hand, allows high-frequency components of a signal to pass through while attenuating low-frequency components. The magnitude response of a high-pass filter increases as the frequency of the signal increases. High-pass filters are commonly used in audio applications to eliminate low-frequency noise or to remove DC offset from a signal. They are also used in control systems to eliminate low-frequency disturbances.")
        st.write("3. A high-pass filter, on the other hand, allows high-frequency components of a signal to pass through while attenuating low-frequency components. The magnitude response of a high-pass filter increases as the frequency of the signal increases. High-pass filters are commonly used in audio applications to eliminate low-frequency noise or to remove DC offset from a signal. They are also used in control systems to eliminate low-frequency disturbances.")
        st.write("4. A band-stop filter, also known as a notch filter, attenuates a specific range of frequencies while allowing frequencies outside that range to pass through. The magnitude response of a band-stop filter typically has a dip in the stopband and a rapid rolloff in the passband. Band-stop filters are commonly used in audio applications to remove unwanted frequencies from a signal, such as hum or noise caused by electrical interference.")
    with st.expander("Line Coding"):
        st.subheader("Line Coding")
        st.write("Line coding is a technique used in digital communication systems to convert a sequence of digital bits into a digital signal that can be transmitted over a communication channel. The purpose of line coding is to ensure that the transmitted signal has desirable properties such as being immune to noise and having a predictable spectrum")
        st.write("We always come across different types of data such as text, numbers, graphical images, audio, and video. These all data are stored in computer memory in form of a sequence of bits. As shown below, line coding converts bit sequences into digital signals. ")
        st.image("file\img\LineCoding.png")
        st.info("Line Coding")
        st.write("We can roughly divide line coding schemes into five categories:")
        st.write("1. Unipolar (eg. NRZ scheme): positive voltage defines bit 1 and the zero voltage defines bit 0. Signal does not return to zero at the middle of the bit thus it is called NRZ. ")
        st.write(
            "2. Polar schemes – In polar schemes, the voltages are on the both sides of the axis.")
        # st.write("NRZ-L and NRZ-I – These are somewhat similar to unipolar NRZ scheme but here we use two levels of amplitude (voltages). For NRZ-L(NRZ-Level), the level of the voltage determines the value of the bit, typically binary 1 maps to logic-level high, and binary 0 maps to logic-level low, and for NRZ-I(NRZ-Invert), two-level signal has a transition at a boundary if the next bit that we are going to transmit is a logical 1, and does not have a transition if the next bit that we are going to transmit is a logical 0")
        st.write("NRZ-L(NRZ-Level), the level of the voltage determines the value of the bit, typically binary 1 maps to logic-level high, and binary 0 maps to logic-level low")
        st.write("NRZ-L(NRZ-Level), the level of the voltage determines the value of the bit, typically binary 1 maps to logic-level high, and binary 0 maps to logic-level low")
        st.write("NRZ-L(NRZ-Level), the level of the voltage determines the value of the bit, typically binary 1 maps to logic-level high, and binary 0 maps to logic-level low")
        st.write("Biphase (Manchester and Differential Manchester ) – Manchester encoding is somewhat combination of the RZ (transition at the middle of the bit) and NRZ-L schemes. The duration of the bit is divided into two halves. The voltage remains at one level during the first half and moves to the other level in the second half. The transition at the middle of the bit provides synchronization. Differential Manchester is somewhat combination of the RZ and NRZ-I schemes. There is always a transition at the middle of the bit but the bit values are determined at the beginning of the bit. If the next bit is 0, there is a transition, if the next bit is 1, there is no transition.")
        st.write("3. Bipolar schemes – In this scheme there are three voltage levels positive, negative, and zero. The voltage level for one data element is at zero, while the voltage level for the other element alternates between positive and negative.")
        st.write("Alternate Mark Inversion (AMI) – A neutral zero voltage represents binary 0. Binary 1’s are represented by alternating positive and negative voltages.")
        st.write("Pseudoternary – Bit 1 is encoded as a zero voltage and the bit 0 is encoded as alternating positive and negative voltages i.e., opposite of AMI scheme.")


with st.container():
    st.subheader("Procedure")
    with st.expander("PCM"):
        st.subheader("PCM")
        st.write("1. Adjust the number of quantization bits: Once the code is running, you will see a slider that allows you to adjust the number of quantization bits used in the PCM process. Move the slider to choose the number of quantization bits you want to use. The higher the number of quantization bits, the more accurate the reconstructed signal will be, but it will also require more bits to represent each sample.")
        st.write("2. Observe the plots: The code will generate two plots. The first plot shows the original signal, sampled signal, quantized signal, and encoded signal. The second plot shows the original signal overlaid with the reconstructed signal. Observe the plots to see how the signal changes as it goes through the PCM process.")
        st.write("3. Experiment with different parameters: You can experiment with different sampling rates, quantization levels, and number of quantization bits to observe the effects on the reconstructed signal.")

    with st.expander("Convlution"):
        st.subheader("Convlution")
        st.write("1. Select the convolution type: Once the code is running, you will see a dropdown menu that allows you to select the type of convolution you want to perform. Choose either 'Continuous-Time Convolution' or 'Discrete-Time Convolution'.")
        st.write("2. Select the input signal and impulse response: After selecting the convolution type, you will see two dropdown menus that allow you to select the input signal and impulse response for the chosen convolution type. Choose a signal and impulse response from the available options.")
        st.write("3. Observe the plots: The code will generate three plots. The first plot shows the input signal, the second plot shows the impulse response, and the third plot shows the convolution result. Observe the plots to see how the convolution works and how the output signal is generated from the input signal and impulse response.")
        st.write("4. Experiment with different parameters: You can experiment with different input signals and impulse responses to observe the effects on the convolution result. You can also try changing the length of the input signal or impulse response to see how it affects the convolution result.")
        st.write("")
    with st.expander("Modulation"):
        st.subheader("Modulation")
        st.write("1. Select the modulation type and frequencies: Once the code is running, you will see a dropdown menu that allows you to select the type of amplitude modulation you want to perform. Choose one of the available options. You will also see two sliders that allow you to select the carrier frequency and message signal frequency.")
        st.write("2. Observe the plots: The code will generate one or two plots, depending on the type of modulation selected. Each plot shows the time-domain and frequency-domain representation of the modulated signal(s). Observe the plots to see how the different types of modulation work and how they affect the signal.")
        st.write("3. Experiment with different parameters: You can experiment with different carrier and message signal frequencies to observe the effects on the modulated signal. You can also try changing the amplitude or frequency of the message signal to see how it affects the modulated signal.")
    with st.expander("Filter"):
        st.subheader("Filter")
        st.write("1. Select the filter type and parameters: Once the code is running, you will see a dropdown menu that allows you to select the type of filter you want to apply. Choose one of the available options. You will also see a slider that allows you to select the order of the filter. Depending on the type of filter selected, you may also see one or two additional sliders that allow you to select the cutoff frequencies.")
        st.write("2. Observe the plots: The code will generate two plots. The first plot shows the time-domain representation of the unfiltered and filtered signal. The second plot shows the frequency response of the filter. Observe the plots to see how the filter affects the signal.")
        st.write("3. Experiment with different parameters: You can experiment with different filter orders and cutoff frequencies to see how they affect the filtered signal.")

    with st.expander("Line Coding"):
        st.subheader("Line Coding")
        st.write("1. Enter the bit sequence: Once the code is running, you will see a text box where you can input a bit sequence. Enter a sequence of 0 and 1 (e.g., '01010101').")
        st.write("2. Select the line coding category and scheme: You will see a dropdown menu where you can select the line coding category (unipolar, polar, or bipolar) and a dropdown menu where you can select a line coding scheme within that category.")
        st.write("3. Observe the plot: The code will generate a plot that shows the signal generated by the selected line coding scheme. The top plot shows the signal, and the bottom plot shows the clock signal. Observe the plot to see how the line coding scheme encodes the bit sequence.")
        st.write("4. Enter a new bit sequence: You can enter a new bit sequence and select a new line coding scheme to see how it encodes the bit sequence.")


with st.container():
    st.subheader("Simulation")

    st.write("1. Analog to Digital Conversion")
    with st.expander("PULSE CODE MODULATION"):

        st.subheader("Pulse Code Modulation")
        t = np.linspace(0, 1, 1020)
        f = 5
        signal = np.sin(2 * np.pi * f * t)
        # Sampling
        sampling_freq = 100
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
        # st.write("Quantized Signal:", quantized_samples)

        # Encoding
        encoded_samples = np.uint8(
            (quantized_samples - np.min(quantized_samples)) / quantization_step)
        # st.write("Encoded Signal:", encoded_samples)

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
        ax[1].stem(samples)
        ax[1].set_title("Sampled Signal")
        ax[2].stem(quantized_samples)
        ax[2].set_title("Quantized Signal")
        ax[3].stem(encoded_samples)
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

        # modulation_Type = st.selectbox(
        #     "Modulation Type", ("Amplitude modulation", "Angle modulation"))
        # if modulation_Type == "Amplitude modulation":

        def generate_signal(fc, fm):
            t = np.linspace(0, 1, 1000)
            m = np.sin(2*np.pi*fm*t)
            return t, m

        def dsb_fc_modulation(t, m, fc):
            dsb_fc = m * np.cos(2*np.pi*fc*t)
            return dsb_fc

        def ssb_sc_modulation(t, m, fc):
            ssb_sc = sig.hilbert(m)
            ssb_sc_upper = np.real(ssb_sc * np.exp(1j*2*np.pi*fc*t))
            ssb_sc_lower = np.imag(ssb_sc * np.exp(1j*2*np.pi*fc*t))
            return ssb_sc_upper, ssb_sc_lower

        def ssb_fc_modulation(t, m, fc):
            ssb_fc = sig.hilbert(m)
            ssb_fc_upper = np.real(ssb_fc * np.exp(1j*2*np.pi*fc*t))
            ssb_fc_lower = -np.imag(ssb_fc * np.exp(1j*2*np.pi*fc*t))
            return ssb_fc_upper, ssb_fc_lower

        def vsb_modulation(t, m, fc):
            vsb = sig.firwin(101, 0.25)
            v = np.convolve(m, vsb, mode='same')
            vsb_upper = v * np.cos(2*np.pi*fc*t)
            vsb_lower = sig.hilbert(v) * np.sin(2*np.pi*fc*t)
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

        # elif modulation_Type == "Angle modulation":
        #     st.write("Angle modulation")

        #     def generate_signal(fm):
        #         t = np.linspace(0, 1, 1000)
        #         m = np.sin(2*np.pi*fm*t)
        #         return t, m

        #     def freq_modulation(t, m, fc, kf):
        #         freq_mod = np.cos(2*np.pi*fc*t + 2*np.pi*kf*np.cumsum(m))
        #         return freq_mod

        #     def phase_modulation(t, m, fc, kp):
        #         phase_mod = np.cos(2*np.pi*fc*t + kp*m)
        #         return phase_mod

        #     def plot_graphs(t, signal, spectrum, title):
        #         fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #         fig.suptitle(title)
        #         axs[0].plot(t, signal)
        #         axs[0].set_title('Time-domain')
        #         axs[1].plot(spectrum[0], np.abs(spectrum[1]))
        #         axs[1].set_title('Frequency-domain')
        #         plt.tight_layout()
        #         return fig

        #     # Accept user input for message signal frequency and modulation parameters
        #     fm = st.slider('Message Frequency (Hz)', 10, 1000, 100, 10)
        #     fc = st.slider(
        #         'Carrier Frequency (Hz)', 100, 10000, 1000, 100)
        #     kf = st.slider(
        #         'Frequency Modulation Index', 0.1, 10.0, 2.0, 0.1)
        #     kp = st.slider(
        #         'Phase Modulation Index', 0.1, 10.0, 2.0, 0.1)

        #     # Generate the message signal
        #     t, m = generate_signal(fm)

        #     # Show the message signal
        #     fig, ax = plt.subplots(figsize=(10, 5))
        #     ax.plot(t, m)
        #     ax.set_title('Message Signal')
        #     plt.tight_layout()
        #     st.pyplot(fig)

        #     angle_modulation_type = st.selectbox(
        #         "Select Angle Modulation", ("Frequency Modulation", "Phase Modulation"))
        #     if angle_modulation_type == "Frequency Modulation":
        #         st.write("Frequency Modulation")
        #         # Implement frequency modulation
        #         freq_mod = freq_modulation(t, m, fc, kf)
        #         freq_mod_spectrum = np.fft.fft(freq_mod)
        #         freq_mod_fig = plot_graphs(t, freq_mod, (np.fft.fftfreq(
        #             t.shape[-1], 1/1000), freq_mod_spectrum), 'Frequency Modulation')
        #         st.pyplot(freq_mod_fig)
        #     elif angle_modulation_type == "Phase Modulation":
        #         st.write("Phase Modulation")

        #         # Implement phase modulation
        #         phase_mod = phase_modulation(t, m, fc, kp)
        #         phase_mod_spectrum = np.fft.fft(phase_mod)
        #         phase_mod_fig = plot_graphs(t, phase_mod, (np.fft.fftfreq(
        #             t.shape[-1], 1/1000), phase_mod_spectrum), 'Phase Modulation')
        #         st.pyplot(phase_mod_fig)

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

        def apply_filter(filter_type, filter_order, cutoff_freqs, x, y):
            if filter_type == 'Low-pass':
                btype = 'lowpass'
            elif filter_type == 'High-pass':
                btype = 'highpass'
            elif filter_type == 'Band-pass':
                btype = 'bandpass'
            elif filter_type == 'Band-stop':
                btype = 'bandstop'
            b, a = sig.butter(filter_order, cutoff_freqs, btype, fs=1000)

            # Plot the frequency response of the filter
            w, h = sig.freqz(b, a)
            fig1, ax1 = plt.subplots()
            ax1.plot(w/np.pi/2 * 500, 20 * np.log10(abs(h)))
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Amplitude (dB)')
            ax1.set_title(f'{filter_type} Filter Frequency Response')

            st.pyplot(fig1)

            filtered = sig.filtfilt(b, a, y)

            plot_graphs(x, y, 'Time (ms)', 'Amplitude',
                        f'{filter_type} Unfiltered')
            plot_graphs(x, filtered, 'Time (ms)', 'Amplitude',
                        f'{filter_type} Filter (Filtered)')

        st.subheader('Filter Implementation in Python and Streamlit')

        filter_type = st.selectbox(
            'Select a filter:', ('Low-pass', 'High-pass', 'Band-pass', 'Band-stop'))
        filter_order = st.slider('Filter Order', 1, 10)

        if filter_type == 'Low-pass' or filter_type == 'High-pass':
            cutoff_freq = st.slider('Cutoff Frequency (Hz)', 0.01, 500.0)
        else:
            low_cutoff_freq = st.slider(
                'Low Cutoff Frequency (Hz)', 0.01, 500.0)
            high_cutoff_freq = st.slider(
                'High Cutoff Frequency (Hz)', low_cutoff_freq + 0.01, 500.0)

        x = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * 5 * x) + np.sin(2 * np.pi * 100 * x)

        if filter_type == 'Low-pass' or filter_type == 'High-pass':
            apply_filter(filter_type.capitalize(),
                         filter_order, cutoff_freq, x, y)
        else:
            apply_filter(filter_type.capitalize(), filter_order, [
                low_cutoff_freq, high_cutoff_freq], x, y)
    st.write("5. Line Coding")
    with st.expander("Line Coding"):
        st.subheader("Line Coding")

        def polar_nrz_l(bit_seq):
            bit_seq = [int(i) for i in bit_seq]
            bit_seq = [-1 if i == 0 else 1 for i in bit_seq]
            return bit_seq

        def unipolar_nrz_l(bit_seq):
            bit_seq = [int(i) for i in bit_seq]
            return bit_seq

        def polar_nrz_i(bit_seq):
            bit_seq = [int(i) for i in bit_seq]
            prev = -1
            new_bit_seq = []
            for i in range(len(bit_seq)):
                if bit_seq[i] == 0:
                    new_bit_seq.append(prev)
                else:
                    prev *= -1
                    new_bit_seq.append(prev)
            return new_bit_seq

        def manchester(bit_seq):
            bit_seq = [int(i) for i in bit_seq]
            new_bit_seq = []
            for i in range(len(bit_seq)):
                if bit_seq[i] == 0:
                    new_bit_seq.append(-1)
                    new_bit_seq.append(1)
                else:
                    new_bit_seq.append(1)
                    new_bit_seq.append(-1)
            return new_bit_seq

        def differential_manchester(bit_seq):
            bit_seq = [int(i) for i in bit_seq]
            prev = -1
            new_bit_seq = []
            for i in range(len(bit_seq)):
                if bit_seq[i] == 0:
                    new_bit_seq.append(prev)
                    prev *= -1
                    new_bit_seq.append(prev)
                else:
                    prev *= -1
                    new_bit_seq.append(prev)
                    new_bit_seq.append(prev)
            return new_bit_seq

        def ami(bit_seq):
            bit_seq = [int(i) for i in bit_seq]
            prev = 1
            new_bit_seq = []
            for i in range(len(bit_seq)):
                if bit_seq[i] == 0:
                    new_bit_seq.append(0)
                else:
                    prev *= -1
                    new_bit_seq.append(prev)
            return new_bit_seq

        def polar_rz(bit_sequence):
            signal = []
            for b in bit_sequence:
                if b == '0':
                    signal.extend([0, -1])
                else:
                    signal.extend([1, -1])
            return signal

        def plot_signal(signal, title, bit_sequence):
            n = len(signal)
            clock = [1 if i % 2 == 0 else 0 for i in range(n)]
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.set_title(title + '\n' + 'Bit Sequence: ' + str(bit_sequence))
            ax1.step(range(n), signal, where='post',
                     linewidth=2, color='black')
            ax1.set_ylim([-2, 2])
            ax1.set_xlim([0, n])
            ax1.set_xticks(range(n))
            ax1.set_yticks([-1, 0, 1])
            ax1.grid(True)
            ax2.step(range(n), clock, where='post',
                     linewidth=2, color='red', linestyle='--')
            ax2.set_ylim([-2, 2])
            ax2.set_xlim([0, n])
            ax2.set_xticks(range(n))
            ax2.set_yticks([-1, 0, 1])
            ax2.grid(True)

        bit_sequence = st.text_input("Enter the Bit Sequence", "01011010")

        line_coding_category = st.selectbox("Select the Line Coding Category", [
                                            "Unipolar", "Polar", "Bipolar"])

        if line_coding_category == "Unipolar":
            line_coding_scheme = st.selectbox(
                "Select the Line Coding Scheme", ["Unipolar NRZ-L"])
        elif line_coding_category == "Polar":
            line_coding_scheme = st.selectbox("Select the Line Coding Scheme", [
                "Polar NRZ-L", "Polar NRZ-I", "Polar RZ", "Manchester", "Differential Manchester"])
        elif line_coding_category == "Bipolar":
            line_coding_scheme = st.selectbox(
                "Select the Line Coding Scheme", ["AMI"])

        if line_coding_scheme == "Polar NRZ-L":
            signal = polar_nrz_l(bit_sequence)
        elif line_coding_scheme == "Unipolar NRZ-L":
            signal = unipolar_nrz_l(bit_sequence)
        elif line_coding_scheme == "Polar NRZ-I":
            signal = polar_nrz_i(bit_sequence)
        elif line_coding_scheme == "Manchester":
            signal = manchester(bit_sequence)
        elif line_coding_scheme == "Differential Manchester":
            signal = differential_manchester(bit_sequence)
        elif line_coding_scheme == "AMI":
            signal = ami(bit_sequence)
        elif line_coding_scheme == "Polar RZ":
            signal = polar_rz(bit_sequence)

        plot_signal(signal, line_coding_scheme, bit_sequence)

        st.pyplot()


with st.container():
    st.subheader("Quiz")
    with st.expander("PCM"):
        st.subheader("PCM")
        st.write(
            "1. What is PCM and what is its purpose in digital communication systems?")
        st.write(
            "2. Explain how a PCM encoder converts an analog signal into a digital signal.")
        st.write(
            "3. What is the Nyquist theorem and how is it related to the sampling rate used in PCM?")
        st.write(
            "4. What is quantization and how does it affect the accuracy of a PCM signal?")
        st.write(
            "5. How can the bit rate of a PCM signal be calculated given the sampling rate and quantization level?")
    with st.expander("Convolution"):

        st.subheader("Convolution")
        st.write(
            "1. What is convolution and what is its purpose in signal processing?")
        st.write("2. How is convolution used to implement linear filters?")
        st.write(
            "3. What is the impulse response of a system and how is it related to convolution?")
        st.write("4. What is the difference between a finite impulse response (FIR) filter and an infinite impulse response (IIR) filter?")
        st.write(
            "5. What is the frequency response of a filter and how is it related to its impulse response?")
    with st.expander("Modulation"):

        st.subheader("Modulation")
        st.write(
            "1. What is modulation and what is its purpose in digital communication systems?")
        st.write(
            "2. Explain the difference between baseband and passband modulation.")
        st.write("3. What is amplitude modulation (AM) and how does it work?")
        st.write("4. What is frequency modulation (FM) and how does it work?")
        st.write("5. What is phase modulation (PM) and how does it work?")
    with st.expander("Filter"):

        st.subheader("Filter")
        st.write(
            "1. What is a filter and what is its purpose in digital communication systems?")
        st.write(
            "2. Explain the difference between a low-pass filter, a high-pass filter, and a band-pass filter.")
        st.write(
            "3. What is the cutoff frequency of a filter and how does it affect the filter's frequency response?")
        st.write("4. What is the difference between a finite impulse response (FIR) filter and an infinite impulse response (IIR) filter?")
        st.write(
            "5. How can filters be designed using frequency-domain specifications?")
    with st.expander("Line Coding"):

        st.subheader("Line Coding")
        st.write(
            "1. What is line coding and what is its purpose in digital communication systems?")
        st.write(
            "2. Explain the difference between unipolar, polar, and bipolar line coding schemes.")
        st.write(
            "3. What is the difference between Non-Return-to-Zero (NRZ) and Return-to-Zero (RZ) line coding?")
        st.write("4. What is Manchester coding and how does it work?")
        st.write(
            "5. What is the difference between Differential Manchester coding and Manchester coding?")


with st.container():
    st.subheader("References")
    with st.expander("References"):
        st.subheader("References")
        st.write("1. Lathi, B. P. (1998). Modern Digital and Analog Communication Systems (3rd ed.). New York: Oxford University Press.")
        st.write(
            "2. Haykin, S. (2001). Communication Systems(4th ed.). Toronto: John Wiley & Sons, Inc.")
        st.write("3. Oppenheim, A. V., Willsky, A. S., with Young, I. T. (1996). Signals and Systems (2nd ed.). Prentice Hall.")
        st.write(
            "4. Orfanidis, S. J. (1996). Introduction to Signal Processing. Prentice Hall.")
        st.write(
            "5. Orfanidis, S. J. (1996). Introduction to Signal Processing. Prentice Hall.")
        st.write("6. MIT OpenCourseWare. (2012). Readings | 6.02 Introduction to EECS II: Digital Communication Systems, Fall 2012 | Electrical Engineering and Computer Science | Massachusetts Institute of Technology. Retrieved June 13, 2023, from https://ocw.mit.edu/courses/6-02-introduction-to-eecs-ii-digital-communication-systems-fall-2012/pages/readings/")
        st.write("7. 'Unipolar, Polar, and Bipolar Line Coding Schemes. 'GeeksforGeeks, 26 Mar. 2019, https://www.geeksforgeeks.org/difference-between-unipolar-polar-and-bipolar-line-coding-schemes/.")
