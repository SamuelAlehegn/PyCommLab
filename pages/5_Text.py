
import streamlit as st
import heapq
import collections
import math
import sys
from collections import defaultdict
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sk_dsp_comm import digitalcom as dc
from scipy.special import erfc


st.set_page_config(page_title="Text Source",  layout="wide",
                   initial_sidebar_state="auto")
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
with st.expander("Source Coding"):
    st.write("Source Coding")
    user_input = st.text_input("Enter the text")
    encoding_scheme = st.selectbox("Select the encoding scheme", [
        "Huffman Coding", "Arithmetic Coding"])
    if user_input:
        if encoding_scheme == "Huffman Coding":
            st.write("Huffman Coding")

            def get_frequency(text):
                frequency = collections.defaultdict(int)
                for char in text:
                    frequency[char] += 1
                return {char: freq/len(text) for char, freq in frequency.items()}

            def entropy(probabilities):
                return -sum(p * math.log2(p) for p in probabilities)

            # Define a function to build the Huffman tree as a dictionary
            def build_huffman_tree(frequency):
                heap = [[weight, [char, ""]]
                        for char, weight in frequency.items()]
                heapq.heapify(heap)
                while len(heap) > 1:
                    low = heapq.heappop(heap)
                    high = heapq.heappop(heap)
                    for pair in low[1:]:
                        pair[1] = '0' + pair[1]
                    for pair in high[1:]:
                        pair[1] = '1' + pair[1]
                    heapq.heappush(
                        heap, [low[0] + high[0]] + low[1:] + high[1:])
                return dict(sorted(heapq.heappop(heap)[1:], key=lambda x: (len(x[-1]), x)))

            # Define a function to encode the text using the Huffman tree
            def huffman_encoding(text, huffman_tree):
                encoding = ""
                for char in text:
                    encoding += huffman_tree[char]
                return encoding

            # Define a function to decode the Huffman encoded text using the Huffman tree
            def huffman_decoding(encoding, huffman_tree):
                decoding = ""
                code = ""
                for bit in encoding:
                    code += bit
                    for char, code_word in huffman_tree.items():
                        if code == code_word:
                            decoding += char
                            code = ""
                            break
                return decoding

            if user_input and len(user_input) >= 2:
                text = user_input

                # Convert analog signal to digital
                fs = 1000  # Sampling frequency
                t = np.arange(len(text)) / fs  # Time vector
                sig = np.array([ord(c) for c in text])  # Analog signal
                sig_digital = signal.resample(sig, len(text))  # Digital signal

                # Plot time and frequency domains of analog signal
                fig1, ax1 = plt.subplots(2, 1, figsize=(10, 10))
                ax1[0].plot(t, sig)
                ax1[0].set_title("Input Text Signal (Time Domain)")
                ax1[0].set_xlabel("Time (s)")
                ax1[0].set_ylabel("Amplitude")
                ax1[1].magnitude_spectrum(sig, Fs=fs)
                ax1[1].set_title("Input Text Signal (Frequency Domain)")
                ax1[1].set_xlabel("Frequency (Hz)")
                ax1[1].set_ylabel("Magnitude")

                # Calculate Huffman encoding
                frequency = get_frequency(text)
                huffman_tree = build_huffman_tree(frequency)
                encoding = huffman_encoding(text, huffman_tree)
                input_size = len(text.encode('utf-8'))
                compressed_size = math.ceil(len(encoding) / 8)
                if len(text) >= 2:
                    compression_ratio = input_size / compressed_size
                else:
                    compression_ratio = 1
                decoding = huffman_decoding(encoding, huffman_tree)

                # Compute entropy of the input text and the encoded output
                input_probabilities = list(frequency.values())
                input_entropy = entropy(input_probabilities)
                encoded_probabilities = list(get_frequency(encoding).values())
                encoded_entropy = entropy(encoded_probabilities)

                # Convert Huffman-encoded signal to digital and plot in time and frequency domains
                encoded_sig = np.array([int(bit) for bit in encoding])
                encoded_sig_digital = signal.resample(
                    encoded_sig, len(encoded_sig))
                t_encoded = np.arange(len(encoding)) / fs
                fig2, ax2 = plt.subplots(2, 1, figsize=(10, 10))
                ax2[0].plot(t_encoded, encoded_sig_digital)
                ax2[0].set_title("Encoded Signal (Time Domain)")
                ax2[0].set_xlabel("Time (s)")
                ax2[0].set_ylabel("Amplitude")
                ax2[1].magnitude_spectrum(encoded_sig_digital, Fs=fs)
                ax2[1].set_title("Encoded Signal (Frequency Domain)")
                ax2[1].set_xlabel("Frequency (Hz)")
                ax2[1].set_ylabel("Magnitude")

                st.pyplot(fig1)

                # Display the results
                st.write("Input text:", text)
                st.write("Huffman tree: ", huffman_tree)
                st.write("Encoded text: ", encoding)
                st.write("Decoded text: ", decoding)
                st.write("Input size: ", input_size, "bytes")
                st.write("Compressed size: ", compressed_size, "bytes")
                st.write("Compression ratio: ", compression_ratio)
                st.write("Input entropy: ", input_entropy, "bits")
                st.write("Encoded entropy", encoded_entropy, "bits")
                average_bits_per_symbol = len(encoding) / len(text)
                efficiency = input_entropy / average_bits_per_symbol
                st.write("Average number of bits", average_bits_per_symbol)
                st.write("Efficiency", efficiency)
                st.pyplot(fig2)

                huffman_encoding_output = encoding

            else:
                st.warning("Enter more than one alphabet")
        elif encoding_scheme == "Arithmetic Coding":
            st.write("Arithmetic Coding")

            def binary_arithmetic_encode(text):
                low = 0
                high = 1
                freq = {}
                for symbol in text:
                    freq[symbol] = freq.get(symbol, 0) + 1
                total_freq = sum(freq.values())
                range_start = 0
                range_end = 0
                symbol_range = {}
                for symbol, symbol_freq in freq.items():
                    range_end += symbol_freq / total_freq
                    symbol_range[symbol] = (range_start, range_end)
                    range_start = range_end
                for symbol in text:
                    symbol_range_start, symbol_range_end = symbol_range[symbol]
                    range_size = high - low
                    high = low + range_size * symbol_range_end
                    low = low + range_size * symbol_range_start
                message = format(int(low * (2 ** 32)), "032b")
                return message

            def binary_arithmetic_decode(message, text_length):
                low = 0
                high = 1
                freq = {}
                for symbol in set(message):
                    freq[symbol] = message.count(symbol)
                total_freq = sum(freq.values())
                range_start = 0
                range_end = 0
                symbol_range = {}
                for symbol, symbol_freq in freq.items():
                    range_end += symbol_freq / total_freq
                    symbol_range[(range_start, range_end)] = symbol
                    range_start = range_end
                decoded_text = ""
                for i in range(text_length):
                    range_size = high - low
                    value = (int((message - low) / range_size * total_freq))
                    symbol_range_start, symbol_range_end = list(symbol_range.keys())[
                        list(symbol_range.values()).index(value)]
                    decoded_symbol = symbol_range[(
                        symbol_range_start, symbol_range_end)]
                    decoded_text += decoded_symbol
                    high = low + range_size * symbol_range_end
                    low = low + range_size * symbol_range_start
                return decoded_text

            text = user_input
            encoded_message = binary_arithmetic_encode(text)
            st.write(f"Encoded message: {encoded_message}")
            # st.error("Please enter some text to encode.")
            # encoded_message = st.text_input(
            #     "Enter the encoded message to decode:", value="", max_chars=None, key=None, type='default')
            # text_length = st.number_input("Enter the length of the original text:",
            #                               value=None, min_value=0, max_value=None, step=1, format=None, key=None)
            # # if st.button("Decode"):
            #     if encoded_message and text_length:
            #         decoded_text = binary_arithmetic_decode(
            #             int(encoded_message, 2), text_length)
            #         st.write(f"Decoded text: {decoded_text}")
            #     else:
            #         st.error(
            #             "Please enter an encoded message and the length of the original text to decode.")


with st.expander("Cahnnel Coding"):
    st.write("Channel Coding")
    cahnnel_encoding_scheme = st.selectbox("Select the Cahnnel Coding scheme", [
        "Convolutional Coding", "Block Coding", "Turbo Coding"])
    if encoding_scheme == "Huffman Coding" and user_input:

        if cahnnel_encoding_scheme == "Convolutional Coding":
            st.write("Convolutional Coding")

            # Define the generator polynomials
            G1 = [1, 1, 0, 1]
            G2 = [1, 0, 1, 1]

            def convolutional_encode(input_bits):
                # Initialize the shift registers
                sr1 = [0] * (len(G1) - 1)
                sr2 = [0] * (len(G2) - 1)

                # Convert input_bits to a list of integers
                input_ints = [int(b) for b in input_bits]

                # Initialize the output bits list
                output_bits = []

                # Loop over the input bits
                for bit in input_ints:
                    # Compute the parity bits
                    p1 = sum([sr1[i] * G1[i+1] for i in range(len(sr1))]) % 2
                    p2 = sum([sr2[i] * G2[i+1] for i in range(len(sr2))]) % 2

                    # Shift the shift registers
                    sr1 = [bit] + sr1[:-1]
                    sr2 = [bit] + sr2[:-1]

                    # Append the output bits
                    output_bits += [p1, p2]

                return output_bits

            output_bits = convolutional_encode(huffman_encoding_output)

            # Display the output bits to the user
            st.write('Huffman Encoding Output:', huffman_encoding_output)
            st.write('Convolutinal Encoding Output:', ''.join(str(b)
                     for b in output_bits))
            convolutional_encode_output = ''.join(str(b) for b in output_bits)

        elif cahnnel_encoding_scheme == "Block Coding":
            st.write("Block Coding")

            def block_encode(binary_str, block_size):
                # Pad binary string with zeros to make it a multiple of block_size
                padded_binary_str = binary_str + '0' * \
                    (block_size - len(binary_str) % block_size)

                # Split binary string into blocks
                blocks = [padded_binary_str[i:i+block_size]
                          for i in range(0, len(padded_binary_str), block_size)]

                return blocks

            def block_decode(blocks):
                # Concatenate blocks to form binary string
                binary_str = ''.join(blocks)

                return binary_str.rstrip('0')

            binary_str = str(huffman_encoding_output)
            block_size = st.slider('Select block size',
                                   min_value=1, max_value=10, value=4)

            blocks = block_encode(binary_str, block_size)
            st.write('Encoded blocks:')
            for block in blocks:
                st.write(block)

            # blocks = st.text_input('Enter encoded blocks (comma-separated)')
            # blocks = blocks.split(',')
            # decoded_binary_str = block_decode(blocks)
            # st.write(f'Decoded binary string: {decoded_binary_str}')
        elif cahnnel_encoding_scheme == "Turbo Coding":
            st.write("Turbo Coding")

with st.expander("Modulation"):
    st.write("Modulation")
    modulation_scheme = st.selectbox(
        "Select the Modulation scheme", ["ASK", "PSK", "FSK", "QAM", "4QAM", "8QAM", "16QAM", "64QAM"])
    if modulation_scheme == "ASK":
        st.write("2-level ASK Modulation")
        # User input
        data = st.text_input('Data bits (e.g. 10101)')
        bit_rate = st.number_input(
            'Bit rate', min_value=1, max_value=10, value=2)
        amp = st.number_input('Amplitude', min_value=1, max_value=10, value=5)
        carrier_freq = st.number_input(
            'Carrier frequency', min_value=1, max_value=10, value=2)

        # Convert data to array of integers
        data = np.array([int(bit) for bit in data])

        # Time axis
        time = np.linspace(0, len(data)/bit_rate, num=len(data)*100)

        # Carrier signal
        carrier = amp * np.sin(2 * np.pi * carrier_freq * time)

        # ASK signal
        ask_signal = np.zeros_like(time)
        for i in range(len(data)):
            if data[i] == 1:
                ask_signal[i*100:(i+1)*100] = carrier[i*100:(i+1)*100]
            else:
                ask_signal[i*100:(i+1)*100] = 0

        # Unipolar NRZ signal
        nrz_signal = np.zeros_like(time)
        for i in range(len(data)):
            if data[i] == 1:
                nrz_signal[i*100:(i+1)*100] = amp
            else:
                nrz_signal[i*100:(i+1)*100] = 0

        # Plot signals
        fig, axs = plt.subplots(3, 1)

        axs[0].plot(time, nrz_signal)
        axs[0].set_title('Data bits')
        axs[0].set_xlabel('Time')
        axs[0].set_ylabel('Amplitude')

        axs[1].plot(time, carrier)
        axs[1].set_title('Carrier signal')
        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Amplitude')

        axs[2].plot(time, ask_signal)
        axs[2].set_title('ASK signal')
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Amplitude')

        # axs[3].stem(range(len(data)), data)
        # axs[3].set_title('Data bits')
        # axs[3].set_xlabel('Bit index')
        # axs[3].set_ylabel('Amplitude')

        st.pyplot(fig)

    elif modulation_scheme == "PSK":
        st.write("Phase Shift Keying")

        # st.title('Binary Phase-shift keying (BPSK) Modulation')

        # User input for bit sequence
        user_input = st.text_input('Enter a bit sequence (e.g. 1011):')

        # User input for carrier frequency
        fc = st.number_input(
            'Enter the carrier frequency (Hz):', min_value=0.0)

        if user_input and fc:
            # Convert user input to list of integers
            bit_sequence = [int(bit) for bit in user_input]

            # Parameters
            T = 1   # bit duration
            Fs = 100  # sampling frequency

            t = np.arange(0, len(bit_sequence)*T, 1/Fs)
            x = np.array([])
            carrier = np.array([])
            nrz = np.array([])

            # Generate BPSK signal, carrier frequency and NRZ signal
            for bit in bit_sequence:
                if bit == 1:
                    x = np.append(x, np.sin(2*np.pi*fc*t[:Fs]))
                    carrier = np.append(carrier, np.sin(2*np.pi*fc*t[:Fs]))
                    nrz = np.append(nrz, np.ones(Fs))
                else:
                    x = np.append(x, -np.sin(2*np.pi*fc*t[:Fs]))
                    carrier = np.append(carrier, np.sin(2*np.pi*fc*t[:Fs]))
                    nrz = np.append(nrz, -np.ones(Fs))

            # Plot BPSK signal, carrier frequency and NRZ signal
            fig, axs = plt.subplots(3, sharex=True)
            axs[0].plot(t, nrz)
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_title('Data bits')

            axs[1].plot(t, carrier)
            axs[1].set_ylabel('Amplitude')
            axs[1].set_title(f'Carrier frequency ({fc} Hz)')

            axs[2].plot(t, x)
            axs[2].set_ylabel('Amplitude')
            axs[2].set_title('BPSK Signal')

            st.pyplot(fig)
    elif modulation_scheme == "FSK":
        st.write("Frequency Shift Keying")

        # st.title('Frequency-shift keying (FSK)')

        # User input for bit sequence
        user_input = st.text_input('Enter a bit sequence (e.g. 1011):')

        # User input for carrier frequencies
        f1 = st.number_input(
            'Enter the carrier frequency for bit 1 (Hz):', min_value=0.0)
        f2 = st.number_input(
            'Enter the carrier frequency for bit 0 (Hz):', min_value=0.0)

        if user_input and f1 and f2:
            # Convert user input to list of integers
            bit_sequence = [int(bit) for bit in user_input]

            # Parameters
            T = 1   # bit duration
            Fs = 100  # sampling frequency

            t = np.arange(0, len(bit_sequence)*T, 1/Fs)
            x = np.array([])
            carrier1 = np.array([])
            carrier2 = np.array([])
            nrz = np.array([])

            # Generate FSK signal, carrier frequencies and NRZ signal
            for bit in bit_sequence:
                if bit == 1:
                    x = np.append(x, np.sin(2*np.pi*f1*t[:Fs]))
                    carrier1 = np.append(carrier1, np.sin(2*np.pi*f1*t[:Fs]))
                    carrier2 = np.append(carrier2, np.zeros(Fs))
                    nrz = np.append(nrz, np.ones(Fs))
                else:
                    x = np.append(x, np.sin(2*np.pi*f2*t[:Fs]))
                    carrier1 = np.append(carrier1, np.zeros(Fs))
                    carrier2 = np.append(carrier2, np.sin(2*np.pi*f2*t[:Fs]))
                    nrz = np.append(nrz, -np.ones(Fs))

            # Plot FSK signal, carrier frequencies and NRZ signal
            fig, axs = plt.subplots(4, sharex=True)
            axs[0].plot(t, nrz)
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Amplitude')
            axs[0].set_title('Data bits')

            axs[1].plot(t, carrier1)
            axs[1].set_ylabel('Amplitude')
            axs[1].set_title(f'Carrier frequency for bit 1 ({f1} Hz)')
            axs[2].plot(t, carrier2)
            axs[2].set_ylabel('Amplitude')
            axs[2].set_title(f'Carrier frequency for bit 0 ({f2} Hz)')

            axs[3].plot(t, x)
            axs[3].set_ylabel('Amplitude')
            axs[3].set_title('FSK Signal')
            st.pyplot(fig)

    elif modulation_scheme == "QAM":
        st.write("Quadrature Amplitude Modulation")
        # st.title('Quadrature Amplitude Modulation (QAM)')

        # User input for binary sequence
        user_input = st.text_input(
            'Enter a binary sequence (e.g. 000001010011100101101111):')

        # User input for carrier frequency
        fc = st.number_input(
            'Enter the carrier frequency (Hz):', min_value=0.0)

        # User input for number of symbols
        M = st.number_input('Enter the number of symbols:',
                            min_value=2, step=1)

        if user_input and fc and M:
            # Parameters
            k = int(np.log2(M))  # bits per symbol
            T = 1   # symbol duration
            Fs = 100  # sampling frequency

            # Convert binary sequence to symbol sequence
            symbol_sequence = [int(user_input[i:i+k], 2)
                               for i in range(0, len(user_input), k)]

            t = np.arange(0, len(symbol_sequence)*T, 1/Fs)
            x_I = np.array([])
            x_Q = np.array([])
            carrier_I = np.array([])
            carrier_Q = np.array([])
            nrz = np.array([])

            # Generate QAM signal and carrier frequencies
            n_side = int(np.sqrt(M))

            for symbol in symbol_sequence:
                I = (symbol % n_side) - (n_side-1)/2
                Q = (symbol // n_side) - (n_side-1)/2

                x_I = np.append(x_I, I * np.sin(2*np.pi*fc*t[:Fs]))
                x_Q = np.append(x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                carrier_Q = np.append(carrier_Q, np.sin(
                    2*np.pi*fc*t[:Fs] + np.pi/2))

                # Convert symbol to bit sequence
                bit_sequence = [int(bit) for bit in format(symbol, f'0{k}b')]
                for bit in bit_sequence:
                    if bit == 1:
                        nrz = np.append(nrz, np.ones(int(Fs/k)))
                    else:
                        nrz = np.append(nrz, -np.ones(int(Fs/k)))

            x = x_I + x_Q

            # Plot QAM signal and carrier frequencies
            fig1, axs1 = plt.subplots(4, sharex=True)

            axs1[0].plot(np.arange(0, len(nrz)/Fs, 1/Fs), nrz)
            axs1[0].set_xlabel('Time')
            axs1[0].set_ylabel('Amplitude')
            axs1[0].set_title('Data bits')

            axs1[1].plot(t, carrier_I)
            axs1[1].set_ylabel('Amplitude')
            axs1[1].set_title(f'In-phase carrier frequency ({fc} Hz)')

            axs1[2].plot(t, carrier_Q)
            axs1[2].set_ylabel('Amplitude')
            axs1[2].set_title(f'Quadrature carrier frequency ({fc} Hz)')

            axs1[3].plot(t, x)
            axs1[3].set_ylabel('Amplitude')
            axs1[3].set_title('QAM Signal')

            st.pyplot(fig1)

            # Plot constellation diagram in new figure
            fig2, ax2 = plt.subplots()
            constellation_x = [(i - (n_side-1)/2) for i in range(n_side)
                               for j in range(n_side)]
            constellation_y = [(j - (n_side-1)/2) for i in range(n_side)
                               for j in range(n_side)]
            ax2.scatter(constellation_x, constellation_y)
            ax2.set_xlabel('In-phase')
            ax2.set_ylabel('Quadrature')
            ax2.set_title('Constellation Diagram')
            st.pyplot(fig2)

    elif modulation_scheme == "4QAM":
        st.write("4 Quadrature Amplitude Modulation")

        # User input for binary sequence
        user_input = st.text_input('Enter a binary sequence (e.g. 00011011):')

        # User input for carrier frequency
        fc = st.number_input(
            'Enter the carrier frequency (Hz):', min_value=0.0)

        if user_input and fc:
            # Parameters
            M = 4  # number of symbols
            k = int(np.log2(M))  # bits per symbol
            T = 1   # symbol duration
            Fs = 100  # sampling frequency

            # Convert binary sequence to symbol sequence
            symbol_sequence = [int(user_input[i:i+k], 2)
                               for i in range(0, len(user_input), k)]

            t = np.arange(0, len(symbol_sequence)*T, 1/Fs)
            x_I = np.array([])
            x_Q = np.array([])
            carrier_I = np.array([])
            carrier_Q = np.array([])
            nrz = np.array([])

            # Generate QAM signal and carrier frequencies
            n_side = int(np.sqrt(M))

            for symbol in symbol_sequence:
                I = (symbol % n_side) - (n_side-1)/2
                Q = (symbol // n_side) - (n_side-1)/2

                x_I = np.append(x_I, I * np.sin(2*np.pi*fc*t[:Fs]))
                x_Q = np.append(x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                carrier_Q = np.append(carrier_Q, np.sin(
                    2*np.pi*fc*t[:Fs] + np.pi/2))

                # Convert symbol to bit sequence
                bit_sequence = [int(bit) for bit in format(symbol, f'0{k}b')]
                for bit in bit_sequence:
                    if bit == 1:
                        nrz = np.append(nrz, np.ones(int(Fs/k)))
                    else:
                        nrz = np.append(nrz, -np.ones(int(Fs/k)))

            x = x_I + x_Q

            # Plot QAM signal and carrier frequencies
            fig1, axs1 = plt.subplots(4, sharex=True)

            axs1[0].plot(np.arange(0, len(nrz)/Fs, 1/Fs), nrz)
            axs1[0].set_xlabel('Time')
            axs1[0].set_ylabel('Amplitude')
            axs1[0].set_title('Data bits')

            axs1[1].plot(t, carrier_I)
            axs1[1].set_ylabel('Amplitude')
            axs1[1].set_title(f'In-phase carrier frequency ({fc} Hz)')

            axs1[2].plot(t, carrier_Q)
            axs1[2].set_ylabel('Amplitude')
            axs1[2].set_title(f'Quadrature carrier frequency ({fc} Hz)')

            axs1[3].plot(t, x)
            axs1[3].set_ylabel('Amplitude')
            axs1[3].set_title('4-QAM Signal')
            st.pyplot(fig1)

            # Plot constellation diagram in new figure
            fig2, ax2 = plt.subplots()
            constellation_x = [(i - (n_side-1)/2) for i in range(n_side)
                               for j in range(n_side)]
            constellation_y = [(j - (n_side-1)/2) for i in range(n_side)
                               for j in range(n_side)]
            ax2.scatter(constellation_x, constellation_y)
            ax2.set_xlabel('In-phase')
            ax2.set_ylabel('Quadrature')
            ax2.set_title('Constellation Diagram')
            st.pyplot(fig2)

    elif modulation_scheme == "8QAM":
        st.write("8 Quadrature Amplitude Modulation")

        # User input for binary sequence
        user_input = st.text_input(
            'Enter a binary sequence (e.g. 000001010011100101101111):')

        # User input for carrier frequency
        fc = st.number_input(
            'Enter the carrier frequency (Hz):', min_value=0.0)

        if user_input and fc:
            # Parameters
            M = 8  # number of symbols
            k = int(np.log2(M))  # bits per symbol
            T = 1   # symbol duration
            Fs = 100  # sampling frequency

            # Convert binary sequence to symbol sequence
            symbol_sequence = [int(user_input[i:i+k], 2)
                               for i in range(0, len(user_input), k)]

            t = np.arange(0, len(symbol_sequence)*T, 1/Fs)
            x_I = np.array([])
            x_Q = np.array([])
            carrier_I = np.array([])
            carrier_Q = np.array([])
            nrz = np.array([])

            # Generate QAM signal and carrier frequencies
            for symbol in symbol_sequence:
                I = (symbol % 4) - 1.5
                Q = (symbol // 4) - 0.5
                x_I = np.append(x_I, I * np.sin(2*np.pi*fc*t[:Fs]))
                x_Q = np.append(x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                carrier_Q = np.append(carrier_Q, np.sin(
                    2*np.pi*fc*t[:Fs] + np.pi/2))

                # Convert symbol to bit sequence
                bit_sequence = [int(bit) for bit in format(symbol, f'0{k}b')]
                for bit in bit_sequence:
                    if bit == 1:
                        nrz = np.append(nrz, np.ones(int(Fs/k)))
                    else:
                        nrz = np.append(nrz, -np.ones(int(Fs/k)))

            x = x_I + x_Q

            # Plot QAM signal and carrier frequencies

            fig1, axs1 = plt.subplots(4, sharex=True)
            axs1[0].plot(np.arange(0, len(nrz)/Fs, 1/Fs), nrz)
            axs1[0].set_xlabel('Time')
            axs1[0].set_ylabel('Amplitude')
            axs1[0].set_title('Data bits')

            axs1[1].plot(t, carrier_I)
            axs1[1].set_ylabel('Amplitude')
            axs1[1].set_title(f'In-phase carrier frequency ({fc} Hz)')
            axs1[2].plot(t, carrier_Q)
            axs1[2].set_ylabel('Amplitude')
            axs1[2].set_title(f'Quadrature carrier frequency ({fc} Hz)')

            axs1[3].plot(t, x)
            axs1[3].set_ylabel('Amplitude')
            axs1[3].set_title('8-QAM Signal')

            st.pyplot(fig1)

            # Plot constellation diagram in new figure
            fig2, ax2 = plt.subplots()

            constellation = [(I-1.5, Q-0.5)
                             for I in range(4) for Q in range(2)]
            constellation_x, constellation_y = zip(*constellation)

            ax2.scatter(constellation_x, constellation_y)

            ax2.set_xlabel('In-phase')
            ax2.set_ylabel('Quadrature')

            ax2.set_title('Constellation Diagram')

            st.pyplot(fig2)

    elif modulation_scheme == "16QAM":
        st.write("16 Quadrature Amplitude Modulation")
        # User input for binary sequence
        user_input = st.text_input(
            'Enter a binary sequence (e.g. 000001010011100101101111):')

        # User input for carrier frequency
        fc = st.number_input(
            'Enter the carrier frequency (Hz):', min_value=0.0)

        if user_input and fc:
            # Parameters
            M = 16  # number of symbols
            k = int(np.log2(M))  # bits per symbol
            T = 1  # symbol duration
            Fs = 100  # sampling frequency

            # Convert binary sequence to symbol sequence
            symbol_sequence = [int(user_input[i:i+k], 2)
                               for i in range(0, len(user_input), k)]

            t = np.arange(0, len(symbol_sequence)*T, 1/Fs)
            x_I = np.array([])
            x_Q = np.array([])
            carrier_I = np.array([])
            carrier_Q = np.array([])
            nrz = np.array([])

            # Generate QAM signal and carrier frequencies
            for symbol in symbol_sequence:
                I = (symbol % 4) - 1.5
                Q = (symbol // 4) - 1.5
                x_I = np.append(x_I, I * np.sin(2*np.pi*fc*t[:Fs]))
                x_Q = np.append(x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                carrier_Q = np.append(carrier_Q, np.sin(
                    2*np.pi*fc*t[:Fs] + np.pi/2))

                # Convert symbol to bit sequence
                bit_sequence = [int(bit) for bit in format(symbol, f'0{k}b')]
                for bit in bit_sequence:
                    if bit == 1:
                        nrz = np.append(nrz, np.ones(int(Fs/k)))
                    else:
                        nrz = np.append(nrz, -np.ones(int(Fs/k)))

            x = x_I + x_Q

            # Plot QAM signal and carrier frequencies

            fig1, axs1 = plt.subplots(4, sharex=True)
            axs1[0].plot(np.arange(0, len(nrz)/Fs, 1/Fs), nrz)
            axs1[0].set_xlabel('Time')
            axs1[0].set_ylabel('Amplitude')
            axs1[0].set_title('Data bits')

            axs1[1].plot(t, carrier_I)
            axs1[1].set_ylabel('Amplitude')
            axs1[1].set_title(f'In-phase carrier frequency ({fc} Hz)')
            axs1[2].plot(t, carrier_Q)
            axs1[2].set_ylabel('Amplitude')
            axs1[2].set_title(f'Quadrature carrier frequency ({fc} Hz)')

            axs1[3].plot(t, x)
            axs1[3].set_ylabel('Amplitude')
            axs1[3].set_title('16-QAM Signal')

            st.pyplot(fig1)

            # Plot constellation diagram in new figure
            fig2, ax2 = plt.subplots()

            constellation = [(I-1.5, Q-1.5)
                             for I in range(4) for Q in range(4)]
            constellation_x, constellation_y = zip(*constellation)

            ax2.scatter(constellation_x, constellation_y)

            # Add lines to x and y axes
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.axvline(0, color='black', linewidth=0.5)

            ax2.set_xlabel('In-phase')
            ax2.set_ylabel('Quadrature')

            ax2.set_title('16-QAM Constellation Diagram')

            st.pyplot(fig2)

    elif modulation_scheme == "64QAM":
        st.write("64 Quadrature Amplitude Modulation")
        # User input for binary sequence
        user_input = st.text_input(
            'Enter a binary sequence (e.g. 000001010011100101101111):')

        # User input for carrier frequency
        fc = st.number_input(
            'Enter the carrier frequency (Hz):', min_value=0.0)

        if user_input and fc:
            M = 64  # number of symbols
            k = int(np.log2(M))  # bits per symbol
            T = 1   # symbol duration
            Fs = 100  # sampling frequency

            # Convert binary sequence to symbol sequence
            symbol_sequence = [int(user_input[i:i+k], 2)
                               for i in range(0, len(user_input), k)]

            t = np.arange(0, len(symbol_sequence)*T, 1/Fs)
            x_I = np.array([])
            x_Q = np.array([])
            carrier_I = np.array([])
            carrier_Q = np.array([])
            nrz = np.array([])

            # Generate QAM signal and carrier frequencies
            for symbol in symbol_sequence:
                I = (symbol % 8) - 3.5
                Q = (symbol // 8) - 3.5
                x_I = np.append(x_I, I * np.sin(2*np.pi*fc*t[:Fs]))
                x_Q = np.append(x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                carrier_Q = np.append(carrier_Q, np.sin(
                    2*np.pi*fc*t[:Fs] + np.pi/2))

                # Convert symbol to bit sequence
                bit_sequence = [int(bit) for bit in format(symbol, f'0{k}b')]
                for bit in bit_sequence:
                    if bit == 1:
                        nrz = np.append(nrz, np.ones(int(Fs/k)))
                    else:
                        nrz = np.append(nrz, -np.ones(int(Fs/k)))

            x = x_I + x_Q

            # Plot QAM signal and carrier frequencies

            fig1, axs1 = plt.subplots(4, sharex=True)
            axs1[0].plot(np.arange(0, len(nrz)/Fs, 1/Fs), nrz)
            axs1[0].set_xlabel('Time')
            axs1[0].set_ylabel('Amplitude')
            axs1[0].set_title('Data bits')

            axs1[1].plot(t, carrier_I)
            axs1[1].set_ylabel('Amplitude')
            axs1[1].set_title(f'In-phase carrier frequency ({fc} Hz)')
            axs1[2].plot(t, carrier_Q)
            axs1[2].set_ylabel('Amplitude')
            axs1[2].set_title(f'Quadrature carrier frequency ({fc} Hz)')

            axs1[3].plot(t, x)
            axs1[3].set_ylabel('Amplitude')
            axs1[3].set_title('64-QAM Signal')

            st.pyplot(fig1)

            # Plot constellation diagram in new figure
            fig2, ax2 = plt.subplots()

            constellation = [(I-3.5, Q-3.5)
                             for I in range(8) for Q in range(8)]
            constellation_x, constellation_y = zip(*constellation)

            ax2.scatter(constellation_x, constellation_y)

            # Add lines to x and y axes
            ax2.axhline(0, color='black', linewidth=0.5)
            ax2.axvline(0, color='black', linewidth=0.5)

            ax2.set_xlabel('In-phase')
            ax2.set_ylabel('Quadrature')

            ax2.set_title('Constellation Diagram')

            st.pyplot(fig2)


with st.expander("Channel"):
    st.title("Cahnnel")
    channel_model_type = st.selectbox(
        'Select the Channel Type', ['AWGN', 'Rayleigh', 'Rician'])

    if channel_model_type == "AWGN":
        st.write("Additive White Gaussian Noise")

        def awgn_ask_ber(EbN0_dB, M):
            # BER calculation for ASK modulation in AWGN channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            return 0.5 * np.exp(-EbN0/(2*M))

        def awgn_psk_ber(EbN0_dB, M):
            # BER calculation for PSK modulation in AWGN channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            k = np.log2(M)  # Number of bits per symbol
            return 1/k * erfc(np.sqrt(EbN0/k) * np.sin(np.pi/M))

        def awgn_fsk_ber(EbN0_dB, M):
            # BER calculation for FSK modulation in AWGN channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            k = np.log2(M)  # Number of bits per symbol
            return 0.5 * np.exp(-EbN0/(2*k))

        # Define the Streamlit app
        st.subheader("ASK, PSK and FSK")
        # Define the input parameters
        modulation_scheme = st.selectbox(
            'Select the Modulation Scheme', ['ASK', 'PSK', 'FSK'])
        # channel_type = st.selectbox('Select the Channel Type', ['AWGN', 'Rayleigh'])
        EbN0_min = st.slider('Eb/N0 (dB) minimum', -10, 20, -10)
        EbN0_max = st.slider('Eb/N0 (dB) maximum', -10, 20, 10)
        M = st.slider('Number of Symbols (M)', 2, 16, 4)
        input_bit = st.text_input('Enter the input bit', '0')

        # Calculate the BER for the selected modulation scheme and channel type
        if modulation_scheme == 'ASK':
            ber_func = awgn_ask_ber
        elif modulation_scheme == 'PSK':
            ber_func = awgn_psk_ber
        elif modulation_scheme == 'FSK':
            ber_func = awgn_fsk_ber

        EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
        ber = ber_func(EbN0_dB, M)

        # Plot the BER vs. Eb/N0 curve
        fig1, ax = plt.subplots()
        ax.semilogy(EbN0_dB, ber)
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title(
            f'{modulation_scheme} Modulation Scheme in {channel_model_type} Channel for {M} Symbols')
        st.pyplot(fig1)

        if st.button("Plot the BER vs. Eb/N0 curves for all three modulation schemes"):
            EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
            ask_ber_func = awgn_ask_ber
            psk_ber_func = awgn_psk_ber
            fsk_ber_func = awgn_fsk_ber
            ask_ber = ask_ber_func(EbN0_dB, M)
            psk_ber = psk_ber_func(EbN0_dB, M)
            fsk_ber = fsk_ber_func(EbN0_dB, M)
            # Plot the BER vs. Eb/N0 curves for all three modulation schemes
            fig, ax = plt.subplots()
            ax.semilogy(EbN0_dB, ask_ber, label='ASK')
            ax.semilogy(EbN0_dB, psk_ber, label='PSK')
            ax.semilogy(EbN0_dB, fsk_ber, label='FSK')
            ax.set_xlabel('Eb/N0 (dB)')
            ax.set_ylabel('Bit Error Rate (BER)')
            ax.set_title(
                f'ASK, PSK, and FSK Modulation Schemes in {channel_model_type} Channel for {M} Symbols')
            ax.legend()
            st.pyplot(fig)

        def awgn_qam_ber(EbN0_dB, M):
            # BER calculation for QAM modulation in AWGN channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            k = np.log2(M)  # Number of bits per symbol
            return 2*(1-1/np.sqrt(M))*erfc(np.sqrt(3*EbN0*k/(2*(M-1))))

        # Define the Streamlit app
        st.subheader("QAM")
        # Define the input parameters
        modulation_scheme = st.selectbox(
            'Select the Modulation Scheme', ['4-QAM', '8-QAM', '16-QAM', '64-QAM', '128-QAM', '256-QAM'])
        min_ebn0 = st.slider('Eb/N0 (dB) minimum', -10,
                             20, -10, key='EbN0_min')
        EbN0_max = st.slider('Eb/N0 (dB) maximum', -10, 20, 10, key='EbN0_max')
        M = int(modulation_scheme.split('-')[0])
        input_bit = st.text_input('Enter the input bit', '0', key='inp')

        # Calculate the BER for the selected modulation scheme and channel type
        ber_func = awgn_qam_ber

        EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
        ber = ber_func(EbN0_dB, M)

        # Plot the BER vs. Eb/N0 curve
        fig1, ax = plt.subplots()
        ax.semilogy(EbN0_dB, ber)
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title(f'{modulation_scheme} Modulation Scheme in AWGN Channel')
        st.pyplot(fig1)

        if st.button("Plot the BER vs. Eb/N0 curves for all QAM modulation schemes"):
            EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
            qam4_ber_func = awgn_qam_ber
            qam8_ber_func = awgn_qam_ber
            qam16_ber_func = awgn_qam_ber
            qam64_ber_func = awgn_qam_ber
            qam128_ber_func = awgn_qam_ber
            qam256_ber_func = awgn_qam_ber
            qam4_ber = qam4_ber_func(EbN0_dB, 4)
            qam8_ber = qam8_ber_func(EbN0_dB, 8)
            qam16_ber = qam16_ber_func(EbN0_dB, 16)
            qam64_ber = qam64_ber_func(EbN0_dB, 64)
            qam128_ber = qam128_ber_func(EbN0_dB, 128)
            qam256_ber = qam256_ber_func(EbN0_dB, 256)
            # Plot the BER vs. Eb/N0 curves for all QAM modulation schemes
            fig, ax = plt.subplots()
            ax.semilogy(EbN0_dB, qam4_ber, label='4-QAM')
            ax.semilogy(EbN0_dB, qam8_ber, label='8-QAM')
            ax.semilogy(EbN0_dB, qam16_ber, label='16-QAM')
            ax.semilogy(EbN0_dB, qam64_ber, label='64-QAM')
            ax.semilogy(EbN0_dB, qam128_ber, label='128-QAM')
            ax.semilogy(EbN0_dB, qam256_ber, label='256-QAM')
            ax.set_xlabel('Eb/N0 (dB)')
            ax.set_ylabel('Bit Error Rate (BER)')
            ax.set_title(f'QAM Modulation Schemes in AWGN Channel')
            ax.legend()
            st.pyplot(fig)

    elif channel_model_type == 'Rayleigh':
        st.write("RAYLEIGH FADING CHANNEL MODEL")

        st.subheader("ASK, PSK and FSK")

        def rayleigh_ask_ber(EbN0_dB):
            # BER calculation for ASK modulation in Rayleigh fading channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            return 0.5 * (1 - np.sqrt(EbN0 / (EbN0 + 1)))

        def rayleigh_fsk_ber(EbN0_dB):
            # BER calculation for FSK modulation in Rayleigh fading channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            return 0.5 * np.exp(-EbN0)

        def rayleigh_psk_ber(EbN0_dB):
            # BER calculation for PSK modulation in Rayleigh fading channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            return 1 - (1 - erfc(np.sqrt(EbN0)))*np.exp(-EbN0)

        # Define the Streamlit app

        # Define the input parameters
        modulation_scheme = st.selectbox(
            'Select the Modulation Scheme', ['ASK', 'FSK', 'PSK'])
        EbN0_min = st.slider('Eb/N0 (dB) minimum', -10, 20, -10, key="RAFPmin")
        EbN0_max = st.slider('Eb/N0 (dB) maximum', -10, 20, 10, key="RAFPmax")
        input_bit = st.text_input('Enter the input bit', '0', key="RAFPminInp")

        # Calculate the BER for the selected modulation scheme and channel type
        if modulation_scheme == 'ASK':
            ber_func = rayleigh_ask_ber
        elif modulation_scheme == 'FSK':
            ber_func = rayleigh_fsk_ber
        elif modulation_scheme == 'PSK':
            ber_func = rayleigh_psk_ber

        EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
        ber = ber_func(EbN0_dB)

        # Plot the BER vs. Eb/N0 curve
        fig1, ax = plt.subplots()
        ax.semilogy(EbN0_dB, ber)
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title(
            f'{modulation_scheme} Modulation Scheme in Rayleigh Fading Channel')
        st.pyplot(fig1)

        if st.button("Plot the BER vs. Eb/N0 curves for all ASK, FSK and PSK modulation schemes"):
            EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
            ask_ber_func = rayleigh_ask_ber
            fsk_ber_func = rayleigh_fsk_ber
            psk_ber_func = rayleigh_psk_ber
            ask_ber = ask_ber_func(EbN0_dB)
            fsk_ber = fsk_ber_func(EbN0_dB)
            psk_ber = psk_ber_func(EbN0_dB)

            # Plot the BER vs. Eb/N0 curves for all ASK, FSK and PSK modulation schemes
            fig2, ax2 = plt.subplots()
            ax2.semilogy(EbN0_dB, ask_ber, label='ASK')
            ax2.semilogy(EbN0_dB, fsk_ber, label='FSK')
            ax2.semilogy(EbN0_dB, psk_ber, label='PSK')

            ax2.set_xlabel('Eb/N0 (dB)')
            ax2.set_ylabel('Bit Error Rate (BER)')
            ax2.set_title(
                f'ASK, FSK and PSK Modulation Schemes in Rayleigh Fading Channel')
            ax2.legend()

            st.pyplot(fig2)

        st.subheader("QAM")

        def rayleigh_qam_ber(EbN0_dB, M):
           # BER calculation for QAM modulation in Rayleigh fading channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            k = np.log2(M)  # Number of bits per symbol
            return 1 - (1 - 2*(1-1/np.sqrt(M))*erfc(np.sqrt(3*EbN0*k/(2*(M-1)))))*np.exp(-EbN0*k)

        # Define the Streamlit app

        # Define the input parameters
        modulation_scheme = st.selectbox(
            'Select the Modulation Scheme', ['4-QAM', '8-QAM', '16-QAM', '64-QAM', '128-QAM', '256-QAM'])
        EbN0_min = st.slider('Eb/N0 (dB) minimum', -10,
                             20, -10, key='RayleighEb/No_min')
        EbN0_max = st.slider('Eb/N0 (dB) maximum', -10, 20,
                             10, key='RayleighEb/No_max')
        M = int(modulation_scheme.split('-')[0])
        input_bit = st.text_input(
            'Enter the input bit', '0', key='RayleighInp')

        # Calculate the BER for the selected modulation scheme and channel type
        ber_func = rayleigh_qam_ber

        EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
        ber = ber_func(EbN0_dB, M)

        # Plot the BER vs. Eb/N0 curve
        fig1, ax = plt.subplots()
        ax.semilogy(EbN0_dB, ber)
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title(
            f'{modulation_scheme} Modulation Scheme in Rayleigh Fading Channel')
        st.pyplot(fig1)

        if st.button("Plot the BER vs. Eb/N0 curves for all QAM modulation schemes"):
            EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
            qam4_ber_func = rayleigh_qam_ber
            qam8_ber_func = rayleigh_qam_ber
            qam16_ber_func = rayleigh_qam_ber
            qam64_ber_func = rayleigh_qam_ber
            qam128_ber_func = rayleigh_qam_ber
            qam256_ber_func = rayleigh_qam_ber
            qam4_ber = qam4_ber_func(EbN0_dB, 4)
            qam8_ber = qam8_ber_func(EbN0_dB, 8)
            qam16_ber = qam16_ber_func(EbN0_dB, 16)
            qam64_ber = qam64_ber_func(EbN0_dB, 64)
            qam128_ber = qam128_ber_func(EbN0_dB, 128)
            qam256_ber = qam256_ber_func(EbN0_dB, 256)
            # Plot the BER vs. Eb/N0 curves for all QAM modulation schemes
            fig2, ax2 = plt.subplots()
            ax2.semilogy(EbN0_dB, qam4_ber, label='4-QAM')
            ax2.semilogy(EbN0_dB, qam8_ber, label='8-QAM')
            ax2.semilogy(EbN0_dB, qam16_ber, label='16-QAM')
            ax2.semilogy(EbN0_dB, qam64_ber, label='64-QAM')
            ax2.semilogy(EbN0_dB, qam128_ber, label='128-QAM')
            ax2.semilogy(EbN0_dB, qam256_ber, label='256-QAM')

            ax2.set_xlabel('Eb/N0 (dB)')
            ax2.set_ylabel('Bit Error Rate (BER)')
            ax2.set_title(f'QAM Modulation Schemes in Rayleigh Fading Channel')
            ax2.legend()

            st.pyplot(fig2)

    elif channel_model_type == 'Rician':
        st.write("RICIAN CHANNEL MODEL")
        st.subheader("ASK, PSK and FSK")

        def rician_ask_ber(EbN0_dB, M, K):
            # BER calculation for ASK modulation in Rician channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            return 0.5 * np.exp(-EbN0/(2*M)) * (1 - np.sqrt(K/(K+1)))

        def rician_psk_ber(EbN0_dB, M, K):
            # BER calculation for PSK modulation in Rician channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            k = np.log2(M)  # Number of bits per symbol
            return 1/k * erfc(np.sqrt(EbN0/k) * np.sin(np.pi/M)) * (1 - np.sqrt(K/(K+1)))

        def rician_fsk_ber(EbN0_dB, M, K):
            # BER calculation for FSK modulation in Rician channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            k = np.log2(M)  # Number of bits per symbol
            return 0.5 * np.exp(-EbN0/(2*k)) * (1 - np.sqrt(K/(K+1)))

        # Define the Streamlit app

        modulation_scheme = st.selectbox(
            'Select the Modulation Scheme', ['ASK', 'PSK', 'FSK'])
        EbN0_min = st.slider('Eb/N0 (dB) minimum', -10, 20, -10)
        EbN0_max = st.slider('Eb/N0 (dB) maximum', -10, 20, 10)
        M = st.slider('Number of Symbols (M)', 2, 16, 4)
        input_bit = st.text_input('Enter the input bit', '0')
        K = st.slider('Ricean K-factor', 1, 20)

        # Calculate the BER for the selected modulation scheme and channel type
        if modulation_scheme == 'ASK':
            ber_func = rician_ask_ber
        elif modulation_scheme == 'PSK':
            ber_func = rician_psk_ber
        elif modulation_scheme == 'FSK':
            ber_func = rician_fsk_ber

        EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
        ber = ber_func(EbN0_dB, M, K)

        # Plot the BER vs. Eb/N0 curve
        fig1, ax = plt.subplots()
        ax.semilogy(EbN0_dB, ber)
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title(
            f'{modulation_scheme} Modulation Scheme in {channel_model_type} Channel for {M} Symbols')
        st.pyplot(fig1)

        if st.button("Plot the BER vs. Eb/N0 curves for all three modulation schemes"):
            EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
            ask_ber_func = rician_ask_ber
            psk_ber_func = rician_psk_ber
            fsk_ber_func = rician_fsk_ber
            ask_ber = ask_ber_func(EbN0_dB, M, K)
            psk_ber = psk_ber_func(EbN0_dB, M, K)
            fsk_ber = fsk_ber_func(EbN0_dB, M, K)

            # Plot the BER vs. Eb/N0 curves for all three modulation schemes
            fig2, ax2 = plt.subplots()
            ax2.semilogy(EbN0_dB, ask_ber, label='ASK')
            ax2.semilogy(EbN0_dB, psk_ber, label='PSK')
            ax2.semilogy(EbN0_dB, fsk_ber, label='FSK')

            ax2.set_xlabel('Eb/N0 (dB)')
            ax2.set_ylabel('Bit Error Rate (BER)')
            ax2.set_title(
                f'ASK, PSK, and FSK Modulation Schemes in {channel_model_type} Channel for {M} Symbols')
            ax2.legend()
            st.pyplot(fig2)

        st.subheader("QAM")

        def rician_qam_ber(EbN0_dB, M, K):
            # BER calculation for QAM modulation in Rician fading channel
            EbN0 = 10**(EbN0_dB/10)  # Convert dB to linear scale
            k = np.log2(M)  # Number of bits per symbol
            return (1 - (1 - 2*(1-1/np.sqrt(M))*erfc(np.sqrt(3*EbN0*k/(2*(M-1)))))*np.exp(-K))/(1+K)

        # Define the Streamlit app
        modulation_scheme = st.selectbox(
            'Select the Modulation Scheme', ['4-QAM', '8-QAM', '16-QAM', '64-QAM', '128-QAM', '256-QAM'])
        EbN0_min = st.slider('Eb/N0 (dB) minimum', -10, 20, -10, key='forR')
        EbN0_max = st.slider('Eb/N0 (dB) maximum', -10, 20, 10, key='forr')
        K = st.slider('Rician K-factor', 0.0, 10.0, 1.0)
        M = int(modulation_scheme.split('-')[0])
        input_bit = st.text_input('Enter the input bit', '0', key='r')
        # Calculate the BER for the selected modulation scheme and channel type
        ber_func = rician_qam_ber
        EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
        ber = ber_func(EbN0_dB, M, K)

        # Plot the BER vs. Eb/N0 curve
        fig1, ax = plt.subplots()
        ax.semilogy(EbN0_dB, ber)
        ax.set_xlabel('Eb/N0 (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.set_title(
            f'{modulation_scheme} Modulation Scheme in Rician Fading Channel')
        st.pyplot(fig1)

        if st.button("Plot the BER vs. Eb/N0 curves for all QAM modulation schemes"):
            EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
            qam4_ber_func = rician_qam_ber
            qam8_ber_func = rician_qam_ber
            qam16_ber_func = rician_qam_ber
            qam64_ber_func = rician_qam_ber
            qam128_ber_func = rician_qam_ber
            qam256_ber_func = rician_qam_ber
            qam4_ber = qam4_ber_func(EbN0_dB, 4, K)
            qam8_ber = qam8_ber_func(EbN0_dB, 8, K)
            qam16_ber = qam16_ber_func(EbN0_dB, 16, K)
            qam64_ber = qam64_ber_func(EbN0_dB, 64, K)
            qam128_ber = qam128_ber_func(EbN0_dB, 128, K)
            qam256_ber = qam256_ber_func(EbN0_dB, 256, K)
            # Plot the BER vs. Eb/N0 curves for all QAM modulation schemes
            fig2, ax2 = plt.subplots()
            ax2.semilogy(EbN0_dB, qam4_ber, label='4-QAM')
            ax2.semilogy(EbN0_dB, qam8_ber, label='8-QAM')
            ax2.semilogy(EbN0_dB, qam16_ber, label='16-QAM')
            ax2.semilogy(EbN0_dB, qam64_ber, label='64-QAM')
            ax2.semilogy(EbN0_dB, qam128_ber, label='128-QAM')
            ax2.semilogy(EbN0_dB, qam256_ber, label='256-QAM')
            ax2.set_xlabel('Eb/N0 (dB)')
            ax2.set_ylabel('Bit Error Rate (BER)')
            ax2.set_title(f'QAM Modulation Schemes in Rician Fading Channel')
            ax2.legend()
            st.pyplot(fig2)
