# Core packages
import streamlit as st
import sys

import heapq
import numpy as np
import matplotlib.pyplot as plt
from sk_dsp_comm.fec_conv import FECConv


st.set_page_config(page_title="Text Source",  layout="wide",
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
    with st.expander("Source Coding"):
        st.write("Source Coding")
        user_input = st.text_input("Enter the text")
        encoding_scheme = st.selectbox("Select the encoding scheme", [
            "Huffman Coding", "Arithmetic Coding", "Shannon-Fano", "Run-Length Encoding"])

        if encoding_scheme == "Huffman Coding":
            st.write("Huffman Coding")
            # st.write("Decoded data:", decoded_data)

            class Nodes:
                def __init__(self, probability, symbol, left=None, right=None):
                    # probability of the symbol
                    self.probability = probability
                    # the symbol
                    self.symbol = symbol
                    # the left node
                    self.left = left
                    # the right node
                    self.right = right
                    # the tree direction (0 or 1)
                    self.code = ''

            # """ A supporting function in order to calculate the probabilities of symbols in specified data """
            def CalculateProbability(the_data):
                the_symbols = dict()
                for item in the_data:
                    if the_symbols.get(item) == None:
                        the_symbols[item] = 1
                    else:
                        the_symbols[item] += 1
                return the_symbols

            # """ A supporting function in order to print the codes of symbols by travelling a Huffman Tree """
            the_codes = dict()

            def CalculateCodes(node, value=''):
                # a huffman code for current node
                newValue = value + str(node.code)

                if (node.left):
                    CalculateCodes(node.left, newValue)
                if (node.right):
                    CalculateCodes(node.right, newValue)

                if (not node.left and not node.right):
                    the_codes[node.symbol] = newValue

                return the_codes

            # """ A supporting function in order to get the encoded result """
            def OutputEncoded(the_data, coding):
                encodingOutput = []
                for element in the_data:
                    # print(coding[element], end = '')
                    encodingOutput.append(coding[element])

                the_string = ''.join([str(item) for item in encodingOutput])
                return the_string

            # """ A supporting function in order to calculate the space difference between compressed and non compressed data"""
            def TotalGain(the_data, coding):
                # total bit space to store the data before compression
                beforeCompression = len(the_data) * 8
                afterCompression = 0
                the_symbols = coding.keys()
                for symbol in the_symbols:
                    the_count = the_data.count(symbol)
                    # calculating how many bit is required for that symbol in total
                    afterCompression += the_count * len(coding[symbol])

                return beforeCompression, afterCompression

            def HuffmanEncoding(the_data):
                symbolWithProbs = CalculateProbability(the_data)
                the_symbols = symbolWithProbs.keys()
                the_probabilities = symbolWithProbs.values()
                # print("symbols: ", the_symbols)
                # print("probabilities: ", the_probabilities)

                the_nodes = []

                # converting symbols and probabilities into huffman tree nodes
                for symbol in the_symbols:
                    the_nodes.append(
                        Nodes(symbolWithProbs.get(symbol), symbol))

                while len(the_nodes) > 1:
                    # sorting all the nodes in ascending order based on their probability
                    the_nodes = sorted(the_nodes, key=lambda x: x.probability)
                    # for node in nodes:
                    #      print(node.symbol, node.prob)

                    # picking two smallest nodes
                    right = the_nodes[0]
                    left = the_nodes[1]

                    left.code = 0
                    right.code = 1

                    # combining the 2 smallest nodes to create new node
                    newNode = Nodes(left.probability + right.probability,
                                    left.symbol + right.symbol, left, right)

                    the_nodes.remove(left)
                    the_nodes.remove(right)
                    the_nodes.append(newNode)

                huffmanEncoding = CalculateCodes(the_nodes[0])
                # print("symbols with codes", huffmanEncoding)
                TotalGain(the_data, huffmanEncoding)
                encodedOutput = OutputEncoded(the_data, huffmanEncoding)
                return encodedOutput, the_nodes[0], the_symbols, the_probabilities, symbolWithProbs, huffmanEncoding

            # def HuffmanDecoding(encodedData, huffmanTree):
            #     treeHead = huffmanTree
            #     decodedOutput = []
            #     for x in encodedData:
            #         if x == '1':
            #             huffmanTree = huffmanTree.right
            #         elif x == '0':
            #             huffmanTree = huffmanTree.left
            #         try:
            #             if huffmanTree.left.symbol == None and huffmanTree.right.symbol == None:
            #                 pass
            #         except AttributeError:
            #             decodedOutput.append(huffmanTree.symbol)
            #             huffmanTree = treeHead

            #     string = ''.join([str(item) for item in decodedOutput])
            #     return string

            # the_data = st.text_input('Enter text')
            if user_input is None:
                st.write("Please enter the text")
                user_input = "none"

            encoding, the_tree, the_symbols, the_probabilities, symbolWithProbs, huffmanEncoding = HuffmanEncoding(
                user_input)
            encoded_output = encoding
            # huffmanDecoding = HuffmanDecoding(encoded_output, the_tree)
            beforeCompression, afterCompression = TotalGain(
                user_input, huffmanEncoding)

            st.write("Input Text: ", user_input)
            st.write("Symbols: ", the_symbols)
            st.write("probabilities: ", the_probabilities)
            st.write("symbol With Probabilities ", symbolWithProbs)
            st.write("Symbols with code", huffmanEncoding)
            st.write("Encoded data:", encoded_output)
            st.write("Space usage before compression (in bits):",
                     beforeCompression)
            st.write("Space usage after compression (in bits):",
                     afterCompression)

        elif encoding_scheme == "Arithmetic Coding":
            st.write("Arithmetic Coding")

            def arithmetic_encode(text):
                # Get size of input string
                input_size = sys.getsizeof(text)

                freq = {}
                for char in text:
                    if char in freq:
                        freq[char] += 1
                    else:
                        freq[char] = 1

                # Calculate probabilities
                total = len(text)
                prob = {}
                start = 0
                for char, f in freq.items():
                    prob[char] = (start, start + f/total)
                    start += f/total

                # Encode the text
                low, high = 0, 1
                for char in text:
                    r_low, r_high = prob[char]
                    range_size = high - low
                    high = low + range_size * r_high
                    low = low + range_size * r_low

                # Get size of encoded text
                encoded_text = (low + high) / 2
                encoded_size = sys.getsizeof(encoded_text)

                return (low + high) / 2, prob, input_size, encoded_size

            # def arithmetic_decode(encoded_text, text_length, prob):
            #     # Initialize variables
            #     decoded_text = ""
            #     low, high = 0, 1

            #     # Decode the text
            #     for i in range(text_length):
            #         for char, (r_low, r_high) in prob.items():
            #             range_size = high - low
            #             if r_low <= (encoded_text - low) / range_size < r_high:
            #                 decoded_text += char
            #                 high = low + range_size * r_high
            #                 low = low + range_size * r_low
            #                 break

            #     return decoded_text
            # text = user_input
            # encoded_text, prob, input_size, encoded_size = arithmetic_encode(text)
            # decoded_text = arithmetic_decode(encoded_text, len(text), prob)
            # print("Original text:", text)
            # print("Encoded text:", encoded_text)
            # print("Decoded text:", decoded_text)
            # print("Input size:", input_size)
            # print("Encoded size:", encoded_size)

            encoded_text, prob, input_size, encoded_size = arithmetic_encode(
                user_input)
            st.write("Input Text: ", user_input)
            st.write("Symbol Probabilities: ", prob)
            st.write("Encoded data:", encoded_text)
            st.write("Space usage before compression (in bits):",
                     input_size)
            st.write("Space usage after compression (in bits):",
                     encoded_size)

        elif encoding_scheme == "Shannon-Fano":
            st.write("Shannon-Fano")
        elif encoding_scheme == "Run-Length Encoding":
            st.write("Run-Length Encoding")

    with st.expander("Cahnnel Coding"):
        st.write("Channel Coding")
        cahnnel_encoding_scheme = st.selectbox("Select the Cahnnel Coding scheme", [
                                               "Convolutional Coding", "Turbo Coding", "Block Coding"])
        if cahnnel_encoding_scheme == "Convolutional Coding":
            st.write("Convolutional Coding")

            # Define the convolutional code parameters
            k = 2   # Number of input bits
            n = 3   # Number of output bits
            g1 = np.array([1, 0, 1])   # Generator polynomial for output bit 1
            g2 = np.array([1, 1, 1])   # Generator polynomial for output bit 2

            # upper row operates on the outputs for the G1 polynomial and the lower row operates on the outputs of the G2 polynomial.
            cc = FECConv(('101', '111'))

            # Accept input from source coding output
            msg = str(encoded_output)
            msg = np.array([int(i) for i in msg])
            state = '00'
            # Convolutionally encode the message sequence
            encoded_msg, state = cc.conv_encoder(msg, state)

            # Calculate the Hamming distance between the original and encoded message sequences
            # dist = hamming_dist(msg, encoded_msg)

            # Print the results
            convolutional_coded_msg = encoded_msg
            st.write("Original message:", str(msg))
            st.write("Encoded message:", str(encoded_msg))
            # print("Hamming distance:", dist)

            # Plot the original and encoded message sequences
            plt.subplot(2, 1, 1)
            plt.stem(msg)
            plt.title("Original Message")
            plt.subplot(2, 1, 2)
            plt.stem(encoded_msg)
            plt.title("Encoded Message")
            st.pyplot()

        #     # Define the convolutional code parameters
        #     K = 3  # Number of input bits to each encoder
        #     N = 2  # Number of output bits from each encoder
        #     rate = K/N  # Code rate

        #     # Accept input data from the user
        #     source_coding = encoded_output
        #     data_str = str(source_coding)
        #     data = np.array([int(d) for d in data_str])

        #     # Encode the data using the convolutional code
        #     encoded_data = conv_encode.conv_encode(data, K, N)
        #     st.write("convolutional code: ", encoded_data)

        #     # # Modulate the encoded data using BPSK modulation
        #     # modulated_data = modnorm.bpsk_mod(encoded_data)

        #     # # Add noise to the modulated data
        #     # noise_power = 0.1  # Noise power
        #     # noisy_data = modulated_data + \
        #     #     np.sqrt(noise_power)*np.random.randn(len(modulated_data))

        #     # # Decode the noisy data using the Viterbi algorithm
        #     # decoded_data = viterbi.viterbi_decode(noisy_data, K, N)

        #     # # Calculate the bit error rate (BER)
        #     # ber = np.sum(np.abs(decoded_data - data))/len(data)
        #     # print('Bit Error Rate: {:.2e}'.format(ber))

        # elif cahnnel_encoding_scheme == "Turbo Coding":
        #     st.write("Turbo Coding")
        # elif cahnnel_encoding_scheme == "Block Coding":
        #     st.write("Block Coding")

    with st.expander("Modulation"):
        st.write("Modulation")
    with st.expander("Channel"):
        st.write("Channel")
    with st.expander("Demodulation"):
        st.write("Demodulation")
    with st.expander("Channel Decoding"):
        st.write("Channel Decoding")
    with st.expander("Source Decoding"):
        st.write("Source Decoding")


with st.container():
    st.subheader("Quize")
    with st.expander("Quize"):
        st.write("Quize")
with st.container():
    st.subheader("References")
    with st.expander("References"):
        st.write("References")
