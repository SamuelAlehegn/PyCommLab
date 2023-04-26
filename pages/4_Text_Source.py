import heapq
import streamlit as st

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

            """ A supporting function in order to calculate the probabilities of symbols in specified data """
            def CalculateProbability(the_data):
                the_symbols = dict()
                for item in the_data:
                    if the_symbols.get(item) == None:
                        the_symbols[item] = 1
                    else:
                        the_symbols[item] += 1
                return the_symbols

            """ A supporting function in order to print the codes of symbols by travelling a Huffman Tree """
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

            """ A supporting function in order to get the encoded result """
            def OutputEncoded(the_data, coding):
                encodingOutput = []
                for element in the_data:
                    # print(coding[element], end = '')
                    encodingOutput.append(coding[element])

                the_string = ''.join([str(item) for item in encodingOutput])
                return the_string

            """ A supporting function in order to calculate the space difference between compressed and non compressed data"""
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
                print("symbols with codes", huffmanEncoding)
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
            encoding, the_tree, the_symbols, the_probabilities, symbolWithProbs, huffmanEncoding = HuffmanEncoding(
                user_input)
            encoded_output = encoding
            # huffmanDecoding = HuffmanDecoding(encoded_output, the_tree)
            beforeCompression, afterCompression = TotalGain(
                user_input, huffmanEncoding)

            st.write("Input Text: ", user_input)
            st.write("Symbols: ", the_symbols)
            st.write("probabilities: ", the_probabilities)
            st.write("symbolWithProbs ", symbolWithProbs)
            st.write("Symbols with code", huffmanEncoding)
            st.write("Encoded data:", encoded_output)
            st.write("Space usage before compression (in bits):",
                     beforeCompression)
            st.write("Space usage after compression (in bits):",
                     afterCompression)

        elif encoding_scheme == "Arithmetic Coding":
            st.write("Arithmetic Coding")
        elif encoding_scheme == "Shannon-Fano":
            st.write("Shannon-Fano")
        elif encoding_scheme == "Run-Length Encoding":
            st.write("Run-Length Encoding")

    with st.expander("Cahnnel Coding"):
        st.write("Channel Coding")
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
