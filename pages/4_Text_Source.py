
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
        st.write("Text information sources are a fundamental component of digital communication. They are the sources of information that are converted into digital signals and transmitted over communication channels.")
        st.write("we will delve into the specifics of text information sources and how they are encoded and decoded in digital communication. This will include an overview of the different encoding schemes used for text, such as ASCII and Unicode, as well as the methods used for error detection and correction.")
        st.write("By the end of this lab , you will have a solid understanding of how text input sources are processed and transmitted through a digital communication channel using source coding, channel coding, and modulation techniques. This knowledge will be valuable in many areas, including digital communication, data entry, and other related fields.")
with st.container():
    st.subheader("Theory")
    with st.expander("Theory"):
        st.write("Source coding")
        st.write("Source coding, also known as data compression or information compression, refers to the process of encoding information in a more efficient manner to reduce the amount of data required to represent it. The goal of source coding is to minimize the number of bits required to represent the information without losing any of the essential details")
        st.write("There are two main types of source coding: lossless and lossy. Lossless compression is a type of data compression in which the original data can be completely reconstructed from the compressed data without any loss of information. Lossy compression, on the other hand, involves discarding some of the information in the original data in order to achieve greater compression.")
        st.write("Source coding is used in a wide variety of applications, such as digital audio and video compression, image compression, and file compression. It is also used in communication systems to reduce the amount of data that needs to be transmitted over a given channel, thereby improving the efficiency of the system.")
        st.write("Types of source coding")
        st.write("Huffman coding:")
        st.write("Huffman coding is a type of data compression algorithm that is used to compress data by assigning shorter codes to more frequently occurring symbols or characters in the data.")
        st.write("The basic idea behind Huffman coding is to use variable-length codes to represent the symbols in the data, with the most frequently occurring symbols represented by the shortest codes. This can result in significant compression of the data, especially if there are only a few symbols that occur very frequently.")
        st.write("The Huffman coding algorithm works by first calculating the frequency of each symbol in the data. It then builds a binary tree called the Huffman tree, where each leaf node represents a symbol and the path from the root to the leaf node represents the code for that symbol. The Huffman tree is built by repeatedly combining the two least frequent symbols, assigning them a common parent node, and assigning the sum of their frequencies as the frequency of the parent node. This process is continued until all the symbols are represented in the tree.")
        st.write("Arithmetic coding:")
        st.write("Arithmetic coding is a data compression technique that encodes a message by representing it as a fraction in a certain range and then converting the fraction to a binary representation.")
        st.write("In arithmetic coding, the message is divided into symbols, and each symbol is assigned a probability based on its frequency of occurrence in the message. The probabilities are used to create a probability distribution, which is then used to encode the message")
        st.write("The encoding process involves repeatedly subdividing the range of possible values into subranges, with each subrange corresponding to a particular symbol. The size of each subrange is proportional to the probability of the corresponding symbol. The current range is then updated to be the subrange that corresponds to the next symbol in the message.")
        st.write("Once all the symbols have been encoded, the resulting range represents the encoded message. This range can then be converted to a binary representation by mapping the range to a sequence of bits.")
        st.write("Arithmetic coding is a powerful compression technique that can achieve high compression ratios, especially for small alphabets or for messages with a skewed probability distribution. However, it is also computationally intensive and can be slower than other compression techniques.")
        st.write("Arithmetic coding is widely used in data compression applications, such as in image and video compression, and in file compression utilities like 7zip and WinRAR.")
        st.write("Source decoding")
        st.write("Source decoding, on the other hand, is the reverse process of source coding. It involves decompressing or decoding the compressed data to recover the original signal or message. This is achieved by using a decoder that is designed to reverse the operations performed by the source encoder. The decoder typically uses a decoding algorithm that is specific to the compression method used.")
        st.write("In general, there are two main types of source coding: lossless and lossy. Lossless source coding techniques ensure that the original information is perfectly preserved after compression and decompression. Examples of lossless source coding techniques include Huffman coding, arithmetic coding, and Lempel-Ziv coding.")
        st.write("Lossy source coding techniques, on the other hand, allow for some loss of information during compression and decompression. The amount of loss depends on the specific compression method and the quality settings used. Common examples of lossy source coding techniques include JPEG for images and MP3 for audio")
        st.write("In summary, source decoding is the process of converting compressed or encoded data back to its original form. Source decoding is an important part of data compression and is used to reduce the amount of data that needs to be transmitted or stored while maintaining the essential information.")
        st.write("Source decoding example")
        st.write("Huffman decoding")
        st.write("Huffman decoding is the process of converting a sequence of encoded binary data, generated by the Huffman coding algorithm, back into its original form. Huffman coding is a lossless data compression algorithm that assigns variable-length codes to symbols based on their frequency of occurrence, with the more frequent symbols receiving shorter codes.")
        st.write("To decode a Huffman code, the first step is to build the Huffman tree that was used to encode the data. This can be done by starting with a list of symbols and their associated frequencies, and then repeatedly combining the two least frequent symbols into a new node until there is only one node left, which becomes the root of the tree.")
        st.write("Once the Huffman tree has been built, the encoded binary data can be read bit by bit, starting from the root of the tree. At each step, if the current bit is a 0, the decoder moves to the left child node of the current node; if it is a 1, the decoder moves to the right child node. This process is repeated until a leaf node is reached, at which point the symbol associated with that leaf node is output as the next decoded symbol")
        st.write("The decoder then resets to the root of the Huffman tree and repeats the process for the next encoded binary data until all of the data has been decoded")
        st.write("Huffman decoding involves building the Huffman tree that was used to encode the data and then using that tree to decode the binary data bit by bit, outputting the corresponding symbols until the entire sequence has been decoded.")
        st.write("Channel coding")
        st.write("Channel coding, also known as error-correcting coding, is a technique used to add redundancy to data before it is transmitted over a communication channel. The goal of channel coding is to detect and correct errors that may occur during transmission, thereby improving the reliability of the communication system.")
        st.write("In channel coding, additional redundant bits are added to the original data before it is transmitted over the channel. These redundant bits are used to detect and correct errors that may occur during transmission. The receiver uses the redundant bits to check for errors and correct them if possible.")
        st.write("Channel coding is used in a wide variety of communication systems, such as wireless communication, satellite communication, and digital television. It is an important technique for ensuring reliable and error-free communication over noisy and unreliable channels.")
        st.write("Types of channel coding")
        st.write("Convolutional coding:")
        st.write("Convolutional coding is a type of error-correcting code used in digital communications to improve the reliability of data transmission over noisy channels. It works by adding redundancy to the data stream, which allows errors to be detected and corrected at the receiver.")
        st.write("The basic idea behind convolutional coding is to encode each data bit using a set of previous bits. This is done by using a shift register and a set of modulo-2 adders (also known as XOR gates). The shift register holds the previous k bits, where k is the constraint length of the code. The modulo-2 adders perform a bitwise XOR of the shift register contents with a set of coefficients, producing a new encoded bit")
        st.write("The encoded bits are then transmitted over the channel. At the receiver, a decoder uses a Viterbi algorithm to estimate the original data bits based on the received bits and the known coding scheme. The Viterbi algorithm searches through all possible paths in the convolutional code trellis, selecting the path that is most likely to have been taken by the encoder.")
        st.write("Convolutional coding offers several advantages over other error-correcting codes, such as block codes. One advantage is that it can provide a higher coding gain for a given code rate, meaning that it can achieve better error correction performance for the same amount of redundancy. Convolutional codes are also more flexible than block codes, allowing for different code rates and constraint lengths to be used for different applications.")
        st.write("However, convolutional coding also has some disadvantages. For example, it can be more complex and computationally intensive than block coding, especially for long constraint lengths. It can also have higher latency since the Viterbi algorithm requires several symbol intervals to decode the received bits.")
        st.write("suppose we want to transmit the message '110101' over a noisy channel using a convolutional code with a constraint length of 3 and a code rate of 1/2. This means that we will encode each data bit using the previous two bits and that we will add one redundant bit for every two data bits.")
        st.write("To encode the message, we start with a shift register of length 3, initialized to all zeros. We then use a set of modulo-2 adders to compute the encoded bits.")
        st.write("For example, the first data bit is 1. We compute the corresponding encoded bits by XORing the first data bit with the second and third bits in the shift register, which are both 0. This gives us the first two encoded bits: 11. We then shift the shift register to the right by one bit and repeat the process for the second data bit.")
        st.write("The second data bit is 1. We compute the encoded bits by XORing the second data bit with the first and third bits in the shift register, which are 1 and 0, respectively. This gives us the next two encoded bits: 01. We then shift the shift register to the right by one bit, and repeat the process for the third data bit.")
        st.write(
            "Continuing in this way, we compute the remaining encoded bits for the message '110101': 110011.")
        st.write("We can see that the convolutional code has added redundancy to the message, which can be used to detect and correct errors at the receiver. For example, if one of the encoded bits is flipped during transmission, the Viterbi decoder at the receiver can use redundancy to estimate the original data bits and correct the error.")
        st.write("Turbo coding:")
        st.write("Turbo coding is a type of forward error correction (FEC) coding scheme that uses two or more convolutional codes in parallel to improve the error correction performance. It was first introduced in the 1990s and has since become widely used in many digital communication systems, including wireless and satellite communication systems.")
        st.write("The basic idea behind turbo coding is to use two or more convolutional codes with different constraint lengths and generator polynomials to encode the data. The encoded bits are then interleaved and transmitted over the channel. At the receiver, a soft-input soft-output (SISO) decoder is used to decode the received bits.")
        st.write("The SISO decoder iteratively processes the received bits, exchanging soft information between the two decoders. The soft information is used to update the posterior probabilities of the encoded bits, which are then used to compute a new estimate of the original data bits.")
        st.write("The iterative process continues until a stopping criterion is met, such as a maximum number of iterations or a minimum decoding threshold. The final estimate of the original data bits is then used for further processing or transmission.")
        st.write("Turbo coding offers several advantages over other FEC coding schemes, such as convolutional coding and Reed-Solomon coding. One advantage is that it can achieve very high coding gains, especially at low signal-to-noise ratios (SNRs). It is also more robust to burst errors and other types of channel impairments.")
        st.write("However, turbo coding is also more complex and computationally intensive than other coding schemes, which can make it more difficult to implement in some systems. It can also have higher latency since the iterative decoding process requires several iterations to converge.")
        st.write("block coding :")
        st.write("block coding refers to a method of error-correcting code that divides data into fixed-length blocks and adds redundant bits to each block. The redundant bits, also known as parity bits, are used to detect and correct errors that may occur during transmission over a noisy channel")
        st.write("In block coding for channel coding, the data is divided into blocks of a fixed size, and each block is encoded separately using a specific coding scheme. The encoded blocks are then transmitted over the channel, with the receiver decoding each block independently.")
        st.write("Block coding is popular in channel coding because it is simple and efficient to implement, and it can provide good error correction performance for a wide range of communication systems. Common examples of block codes used in channel coding include the Hamming code, Reed-Solomon code, and BCH code.")
        st.write("Overall, block coding is a widely used technique in channel coding that helps to improve the reliability and robustness of digital communication systems.")
        st.write("Channel decoding")
        st.write("Channel decoding is the process of recovering the original data from a received signal that has been corrupted by noise, interference, or other channel impairments during transmission. Channel coding is used to add redundancy to the transmitted data in order to allow for error detection and correction.")
        st.write("In digital communication systems, channel coding is used to add extra bits to the original data stream to create a code word that can be transmitted over the channel. The receiver then uses the channel decoding process to recover the original data by using the redundant bits to correct any errors that may have occurred during transmission.")
        st.write("There are two main types of channel coding: block codes and convolutional codes. Block codes divide the data into blocks and add redundancy to each block, while convolutional codes add redundancy to the data stream in a continuous manner. Both types of codes can be used for error detection and correction.")
        st.write("The channel decoding process involves comparing the received code word to all possible code words and selecting the code word that is closest to the received signal. The distance between the received signal and the closest code word is measured using a metric, such as the Hamming distance or the Euclidean distance.")
        st.write("Once the closest codeword has been identified, the channel decoder uses the redundancy in the code word to correct any errors that may have occurred during transmission. The specific algorithm used for error correction depends on the type of code used and the amount of redundancy added.")
        st.write("In summary, channel decoding is the process of recovering the original data from a received signal that has been corrupted by noise, interference, or other channel impairments during transmission. Channel coding is used to add redundancy to the transmitted data to allow for error detection and correction, and the channel decoding process involves comparing the received signal to all possible code words and using the redundancy to correct any errors.")
        st.write("Channel decoding type")
        st.write("Viterbi decoding")
        st.write("Viterbi decoding is a maximum likelihood decoding algorithm used in communication systems to decode convolutional codes. Convolutional codes are error-correcting codes that are used to protect digital data from transmission errors in noisy communication channels.")
        st.write("The Viterbi decoding algorithm uses a dynamic programming approach to find the most likely sequence of encoded data that was transmitted over the channel. The algorithm works by constructing a trellis diagram that represents all possible paths through the convolutional code and then finding the path with the highest likelihood.")
        st.write("At each step in the trellis, the algorithm calculates the metric, which is a measure of the similarity between the received data and the expected data. The metric is calculated for each possible transition from the previous state to the current state, and the algorithm chooses the transition with the highest metric. This process is repeated for each step in the trellis until the end of the code is reached.")
        st.write("Once the most likely path through the trellis has been determined, the decoded data is extracted by reversing the encoding process. The Viterbi decoding algorithm is able to correct errors that occur during transmission by using the redundancy of the convolutional code.")
        st.write("Viterbi decoding is widely used in communication systems that require high reliability and low error rates, such as satellite communications, wireless networks, and digital television. The algorithm provides a robust and efficient way to decode convolutional codes and recover digital data that has been transmitted over noisy channels.")
        st.write("Modulation")
        st.write("Modulation is a process of varying a carrier signal by adding information to it. The carrier wave's amplitude, frequency, or phase is changed to represent the information being transmitted. The modulated signal is transmitted through a transmission medium that may be wired or wireless.")
        st.write("Modulation techniques are classified into two categories: analog modulation and digital modulation. Analog modulation is used to transmit analog signals, such as voice and music, while digital modulation is used to transmit digital signals, such as data and video.")
        st.write("Digital Modulation Techniques")
        st.write("Digital modulation techniques are used to transmit digital signals. The three types of digital modulation techniques are amplitude shift keying (ASK), frequency shift keying (FSK), and phase shift keying (PSK).")
        st.write("Amplitude Shift Keying (ASK)")
        st.write("Amplitude Shift Keying (ASK) is a digital modulation technique that represents digital signals by varying the amplitude of a carrier wave. In ASK, the amplitude of the carrier wave is switched between two different levels to represent binary 0 and 1. The signal is demodulated by comparing the amplitude of the modulated signal with a threshold value.")
        st.write("Types of ASK")
        st.write("There are two types of ASK:")
        st.write("On-Off Keying (OOK): On-Off Keying is a type of ASK in which the amplitude of the carrier wave is switched between two levels, one of which is zero. In OOK, the carrier wave is either present or absent to represent binary 1 or 0. OOK is commonly used in optical communication systems, such as fiber-optic communication.")
        st.write("Amplitude Shift Keying with Carrier (ASK-SC): Amplitude Shift Keying with Carrier is a type of ASK in which the carrier wave is present in both the '1' and '0' states. The amplitude of the carrier wave is switched between two levels to represent binary 0 and 1. ASK-SC is commonly used in radio communication systems, such as wireless communication.")
        st.image("file\img\photo_2023-06-11_09-36-12.jpg")
        st.write("Phase Shift Keying (PSK)")
        st.write("Phase Shift Keying (PSK) is a digital modulation technique that represents digital signals by varying the phase of a carrier wave. In PSK, the phase of the carrier wave is switched between several predetermined values to represent binary 0 and 1. The signal is demodulated by comparing the phase of the modulated signal with a reference phase.")
        st.write("Types of PSK")
        st.write("There are several types of PSK:")
        st.write("Binary Phase Shift Keying (BPSK): Binary Phase Shift Keying is a type of PSK in which the phase of the carrier wave is switched between two values, 0 and 180 degrees. BPSK is the simplest form of PSK and is commonly used in low-data-rate applications.")
        st.write("Quadrature Phase Shift Keying (QPSK): Quadrature Phase Shift Keying is a type of PSK in which the phase of the carrier wave is switched between four values, separated by 90 degrees, to represent two bits per symbol. QPSK is commonly used in satellite and wireless communication systems.")
        st.write("8-PSK: 8-PSK is a type of PSK in which the phase of the carrier wave is switched between eight values, separated by 45 degrees, to represent three bits per symbol. 8-PSK is commonly used in digital broadcasting and wireless communication systems.")
        st.write("16-PSK: 16-PSK is a type of PSK in which the phase of the carrier wave is switched between 16 values, separated by 22.5 degrees, to represent four bits per symbol. 16-PSK is commonly used in high-speed data communication systems, such as satellite communication and wireless LANs.")
        st.image("file\img\photo_2023-06-11_09-38-03.jpg")
        st.write("Quadrature Amplitude Modulation (QAM)")
        st.write("Quadrature Amplitude Modulation (QAM) is a digital modulation technique that combines both amplitude and phase modulation in order to transmit digital data over a communication channel. QAM is used in both wired and wireless communication systems, including cable modems, digital television broadcasting, and Wi-Fi.")
        st.write("QAM works by modulating two carrier waves that are 90 degrees out of phase with each other. The amplitude and phase of each carrier wave are varied to represent digital bits. The combination of the amplitude and phase of the two carrier waves produces a unique point on a constellation diagram, which is used to represent a specific digital symbol.")
        st.write("The number of points on the constellation diagram determines the number of bits that can be transmitted per symbol.")
        st.write("Types of QAM")
        st.write("4QAM")
        st.write("In 4QAM, two orthogonal carrier signals are used to transmit two bits of digital data simultaneously. The carrier signals are in-phase (I) and quadrature (Q) and are modulated by the amplitude of the digital signal. The four possible amplitude levels of the modulated carrier wave correspond to the four possible binary combinations of the two bits")
        st.write("The four states of 4QAM are represented in a constellation diagram, where the I and Q axes represent the two carrier signals and the four points on the diagram represent the four possible amplitude levels. The diagram is usually arranged in a square, with the four points at the corners.")
        st.write("The advantages of 4QAM include its ability to transmit two bits of information per symbol and its ability to efficiently use the available bandwidth. However, it is susceptible to noise and interference, which can cause errors in the received signal.")
        st.write("4QAM is commonly used in low-data rate applications, such as voice and data communications over cellular networks, satellite communications, and digital television broadcasting. It can also be used in high-speed data transmission applications, but higher-order QAM schemes are usually preferred due to their higher spectral efficiency and greater resistance to noise.")
        st.write("8 QAM")
        st.write("8QAM, or 8-ary Quadrature Amplitude Modulation, is a digital modulation scheme that uses eight different amplitude levels to transmit three bits of digital data per symbol. It is an extension of 4QAM, which uses four amplitude levels to transmit two bits per symbol.")
        st.write("In 8QAM, the amplitude levels are represented by eight points on a constellation diagram, which is a graphical representation of the modulation scheme. The constellation diagram for 8QAM is typically arranged in a square with the eight points located at the corners and the midpoint of each side.")
        st.write("Like 4QAM, 8QAM uses two orthogonal carrier signals, one in-phase (I) and one quadrature (Q), to transmit the modulated signal. The amplitude of the signal is determined by the combination of the three bits being transmitted, with each combination corresponding to a specific point on the constellation diagram.")
        st.write("One advantage of 8QAM over 4QAM is that it can transmit more data per symbol, which can increase the data rate of a communication system. However, 8QAM is also more susceptible to noise and interference, which can cause errors in the received signal.")
        st.write("8QAM is commonly used in digital communication systems such as cable modems, digital subscriber line (DSL) modems, and wireless local area networks (WLANs). It is also used in digital terrestrial television broadcasting and in some satellite communication systems.")
        st.write("16 QAM")
        st.write("16QAM, or 16-ary Quadrature Amplitude Modulation, is a digital modulation scheme that uses 16 different amplitude levels to transmit four bits of digital data per symbol. It is an extension of both 4QAM and 8QAM, which use four and eight amplitude levels, respectively.")
        st.write("In 16QAM, the amplitude levels are represented by 16 points on a constellation diagram, which is a graphical representation of the modulation scheme. The constellation diagram for 16QAM is typically arranged in a square with the 16 points located at the corners and the midpoint of each side.")
        st.write("Like 4QAM and 8QAM, 16QAM uses two orthogonal carrier signals, one in-phase (I) and one quadrature (Q), to transmit the modulated signal. The amplitude of the signal is determined by the combination of the four bits being transmitted, with each combination corresponding to a specific point on the constellation diagram.")
        st.write("One advantage of 16QAM over 8QAM is that it can transmit even more data per symbol, which can increase the data rate of a communication system. However, 16QAM is also more susceptible to noise and interference, which can cause errors in the received signal.")
        st.write("16QAM is commonly used in digital communication systems such as cable modems, digital subscriber line (DSL) modems, and wireless local area networks (WLANs). It is also used in digital terrestrial television broadcasting and in some satellite communication systems.")
        st.write("64-QAM")
        st.write("64QAM, or 64-ary Quadrature Amplitude Modulation, is a digital modulation technique that uses 64 different amplitude levels to transmit six bits of digital data per symbol. It is an extension of 16QAM, which uses 16 amplitude levels to transmit four bits per symbol.")
        st.write("In 64QAM, the amplitude levels are represented by 64 points on a constellation diagram, which is a graphical representation of the modulation scheme. The constellation diagram for 64QAM is typically arranged in a square with the 64 points located at the corners and the midpoint of each side.")
        st.write("Like 16QAM, 64QAM uses two orthogonal carrier signals, one in-phase (I) and one quadrature (Q), to transmit the modulated signal. The amplitude of the signal is determined by the combination of the six bits being transmitted, with each combination corresponding to a specific point on the constellation diagram.")
        st.write("One advantage of 64QAM over 16QAM is that it can transmit even more data per symbol, which can increase the data rate of a communication system. However, 64QAM is also more susceptible to noise and interference, which can cause errors in the received signal.")
        st.write("64QAM is commonly used in digital communication systems such as digital cable TV, satellite TV, and wireless communication systems like Wi-Fi and 4G LTE. It can achieve higher data rates than 16QAM but requires a higher signal-to-noise ratio for reliable transmission.")
        st.write("Channel type")
        st.write("Additive White Gaussian Noise (AWGN) channel: ")
        st.write("The Additive White Gaussian Noise (AWGN) channel is a commonly used communication channel model in which the transmitted signal is corrupted by a random noise that is added to it. The noise is assumed to be Gaussian with a mean of zero and a constant power spectral density.")
        st.write("The AWGN channel is used to model various types of communication channels, including wired and wireless channels, and is used to evaluate the performance of communication systems with respect to their ability to overcome noise and interference.")
        st.write("In the AWGN channel model, the received signal is given by:")
        st.write("y(t) = x(t) + n(t)")
        st.write("where y(t) is the received signal, x(t) is the transmitted signal, and n(t) is the Gaussian noise added to the signal. The noise is assumed to be independent and identically distributed (i.i.d.) across time and frequency.")
        st.write("The power of the noise is usually specified by the noise power spectral density (PSD), which is denoted by N_0/2. The noise power is proportional to the bandwidth of the channel, and the signal-to-noise ratio (SNR) is defined as the ratio of the signal power to the noise power.")
        st.write("The AWGN channel is a useful tool in designing and evaluating communication systems, as it enables the performance of different modulation and coding techniques to be compared under realistic conditions. It is also used in channel coding, where error correction codes are designed to correct errors induced by the noise in the AWGN channel.")
        st.write(" Rayleigh Fading Channel Model: ")
        st.write("The Rayleigh Fading Channel Model is a widely used channel model in wireless communication systems, which describes the effect of multipath propagation on the received signal. The Rayleigh Fading Channel Model assumes that the received signal is the sum of multiple copies of the transmitted signal, each of which has been attenuated and delayed due to reflections and scattering from the surrounding environment.")
        st.write(
            "The Rayleigh Fading Channel Model is characterized by a random amplitude and phase shift that varies over time and frequency. The amplitude of the received signal follows a Rayleigh distribution, which is a special case of the more general Ricean distribution. The phase of the received signal is uniformly distributed over the range [0, 2π].")
        st.write("The Rayleigh Fading Channel Model is particularly useful for modeling wireless channels in urban and indoor environments, where the signal is subject to significant attenuation and delay due to reflections from buildings and other objects. The model is also used to evaluate the performance of various modulation and coding techniques in the presence of fading.")
        st.write("In practical wireless systems, the effect of Rayleigh fading can be mitigated through the use of diversity techniques, such as space diversity, time diversity, and frequency diversity. These techniques involve the use of multiple antennas, multiple time slots, or multiple frequency channels to improve the reliability of the received signal.")
        st.image("file\img\photo_2023-06-11_09-38-23.jpg")
        st.write("Rician Fading Channel Model:")
        st.write("The Rician Fading Channel Model is a type of wireless channel model that describes the effect of multipath propagation on a transmitted signal. The Rician Fading Channel Model is similar to the Rayleigh Fading Channel Model but with the addition of a dominant line-of-sight (LOS) signal component in addition to the scattered signals.")
        st.write("In the Rician Fading Channel Model, the received signal is the sum of a dominant LOS component and multiple scattered multipath components. The amplitude and phase of each component are assumed to be random, but the amplitude of the LOS component is typically larger than that of the scattered components. The amplitude of the received signal follows a Rician distribution, which is a special case of the more general Ricean distribution.")
        st.write("The Rician Fading Channel Model is commonly used to model wireless channels in environments where there is a clear line-of-sight path between the transmitter and the receiver, such as open fields or rural areas. The model is also used to evaluate the performance of various modulation and coding techniques in the presence of fading.")
        st.write("In practical wireless systems, the effect of Rician fading can be mitigated through the use of diversity techniques, such as space diversity, time diversity, and frequency diversity. These techniques involve the use of multiple antennas, multiple time slots, or multiple frequency channels to improve the reliability of the received signal.")
        st.write("")

with st.container():
    st.subheader("Procedure")
    with st.expander("Procedure"):
        st.write("Procedure ")
        st.write("For a text input source")
        st.write("1.Select the source coding tab under the simulation title ")
        st.write("2.Enter the text you want to transmit and press “Enter”")
        st.write("3.Select the source encoding scheme you want")
        st.write(
            "4.Observe the input text signal and the encoded signal in time and frequency domain ")
        st.write("5.Select the channel encoding scheme you want ")
        st.write("6.Select modulation technique ")
        st.write(
            "7.Enter the channel encoding output bits to the “enter the bits” section")
        st.write("8.Under the channel title select the type of channel ")


with st.container():
    st.subheader("Simulation")
with st.expander("Source Coding"):
    st.subheader("Source Coding")
    user_input = st.text_input("Enter the text")
    text_input = user_input
    encoding_scheme = st.selectbox("Select the encoding scheme", [
        "Huffman Coding", "Arithmetic Coding"])
    if user_input:
        fs = 1000  # Sampling frequency
        t = np.arange(len(user_input)) / fs  # Time vector
        sig = np.array([ord(c) for c in user_input])  # Analog signal
        sig_digital = signal.resample(sig, len(user_input))  # Digital signal

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
        st.pyplot(fig1)

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


with st.expander("Cahnnel Coding"):
    st.write("Channel Coding")
    cahnnel_encoding_scheme = st.selectbox("Select the Cahnnel Coding scheme", [
        "Convolutional Coding", "Block Coding"])
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
        # elif cahnnel_encoding_scheme == "Turbo Coding":
        #     st.write("Turbo Coding")

# if cahnnel_encoding_scheme == "Convolutional Coding" and encoding_scheme == "Huffman Coding":
if encoding_scheme == "Huffman Coding" and user_input:
    demodulation = ''
    f_psk = ''
    f_fskmin = ''
    f_fskmax = ''

    with st.expander("Modulation"):
        st.write("Modulation")
        user_input = str(convolutional_encode_output)

        modulation_scheme = st.selectbox(
            "Select the Modulation scheme", ["ASK", "PSK", "FSK", "QAM", "4QAM", "8QAM", "16QAM", "64QAM"])
        if modulation_scheme == "ASK":
            demodulation = "ASK"
            st.write("2-level ASK Modulation")
            # User input
            data = user_input
            bit_rate = st.number_input(
                'Bit rate', min_value=1, max_value=10, value=2)
            amp = st.number_input(
                'Amplitude', min_value=1, max_value=10, value=5)
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
            demodulation = "PSK"

            # st.title('Binary Phase-shift keying (BPSK) Modulation')

            # User input for bit sequence
            user_input = str(convolutional_encode_output)

            # User input for carrier frequency
            fc = st.number_input(
                'Enter the carrier frequency (Hz):', min_value=0.0)
            f_psk = fc

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
            demodulation = "FSK"

            # st.title('Frequency-shift keying (FSK)')

            # User input for bit sequence
            user_input = str(convolutional_encode_output)

            # User input for carrier frequencies
            f1 = st.number_input(
                'Enter the carrier frequency for bit 1 (Hz):', min_value=0.0)
            f2 = st.number_input(
                'Enter the carrier frequency for bit 0 (Hz):', min_value=0.0)
            f_fskmax = f2
            f_fskmin = f1

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
                        carrier1 = np.append(
                            carrier1, np.sin(2*np.pi*f1*t[:Fs]))
                        carrier2 = np.append(carrier2, np.zeros(Fs))
                        nrz = np.append(nrz, np.ones(Fs))
                    else:
                        x = np.append(x, np.sin(2*np.pi*f2*t[:Fs]))
                        carrier1 = np.append(carrier1, np.zeros(Fs))
                        carrier2 = np.append(
                            carrier2, np.sin(2*np.pi*f2*t[:Fs]))
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
            user_input = str(convolutional_encode_output)

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
                    x_Q = np.append(
                        x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                    carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                    carrier_Q = np.append(carrier_Q, np.sin(
                        2*np.pi*fc*t[:Fs] + np.pi/2))

                    # Convert symbol to bit sequence
                    bit_sequence = [int(bit)
                                    for bit in format(symbol, f'0{k}b')]
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
            user_input = str(convolutional_encode_output)

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
                    x_Q = np.append(
                        x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                    carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                    carrier_Q = np.append(carrier_Q, np.sin(
                        2*np.pi*fc*t[:Fs] + np.pi/2))

                    # Convert symbol to bit sequence
                    bit_sequence = [int(bit)
                                    for bit in format(symbol, f'0{k}b')]
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
            user_input = str(convolutional_encode_output)

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
                    x_Q = np.append(
                        x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                    carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                    carrier_Q = np.append(carrier_Q, np.sin(
                        2*np.pi*fc*t[:Fs] + np.pi/2))

                    # Convert symbol to bit sequence
                    bit_sequence = [int(bit)
                                    for bit in format(symbol, f'0{k}b')]
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
            user_input = str(convolutional_encode_output)

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
                    x_Q = np.append(
                        x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                    carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                    carrier_Q = np.append(carrier_Q, np.sin(
                        2*np.pi*fc*t[:Fs] + np.pi/2))

                    # Convert symbol to bit sequence
                    bit_sequence = [int(bit)
                                    for bit in format(symbol, f'0{k}b')]
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
            user_input = str(convolutional_encode_output)

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
                    x_Q = np.append(
                        x_Q, Q * np.sin(2*np.pi*fc*t[:Fs] + np.pi/2))
                    carrier_I = np.append(carrier_I, np.sin(2*np.pi*fc*t[:Fs]))
                    carrier_Q = np.append(carrier_Q, np.sin(
                        2*np.pi*fc*t[:Fs] + np.pi/2))

                    # Convert symbol to bit sequence
                    bit_sequence = [int(bit)
                                    for bit in format(symbol, f'0{k}b')]
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
            input_bit = str(1010)

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
            EbN0_max = st.slider('Eb/N0 (dB) maximum', -
                                 10, 20, 10, key='EbN0_max')
            M = int(modulation_scheme.split('-')[0])
            input_bit = str(1010)

            # Calculate the BER for the selected modulation scheme and channel type
            ber_func = awgn_qam_ber

            EbN0_dB = np.arange(EbN0_min, EbN0_max+1, 1)
            ber = ber_func(EbN0_dB, M)

            # Plot the BER vs. Eb/N0 curve
            fig1, ax = plt.subplots()
            ax.semilogy(EbN0_dB, ber)
            ax.set_xlabel('Eb/N0 (dB)')
            ax.set_ylabel('Bit Error Rate (BER)')
            ax.set_title(
                f'{modulation_scheme} Modulation Scheme in AWGN Channel')
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
            EbN0_min = st.slider('Eb/N0 (dB) minimum', -
                                 10, 20, -10, key="RAFPmin")
            EbN0_max = st.slider('Eb/N0 (dB) maximum', -
                                 10, 20, 10, key="RAFPmax")
            input_bit = str(1010)

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
            input_bit = str(1010)

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
                ax2.set_title(
                    f'QAM Modulation Schemes in Rayleigh Fading Channel')
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
            input_bit = str(1010)
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
            EbN0_min = st.slider('Eb/N0 (dB) minimum', -
                                 10, 20, -10, key='forR')
            EbN0_max = st.slider('Eb/N0 (dB) maximum', -10, 20, 10, key='forr')
            K = st.slider('Rician K-factor', 0.0, 10.0, 1.0)
            M = int(modulation_scheme.split('-')[0])
            input_bit = str(1010)
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
                ax2.set_title(
                    f'QAM Modulation Schemes in Rician Fading Channel')
                ax2.legend()
                st.pyplot(fig2)

    with st.expander("Demodulation"):
        st.subheader("Demodulation")
        modulation_scheme = st.selectbox(
            "Select the Demodulation scheme", ["ASK", "PSK", "FSK"])
        # if modulation_scheme == "ASK":
        if modulation_scheme == "ASK" and demodulation == "ASK":
            st.subheader("ASK Demodulation")
            demodulated_signal = ask_signal * carrier

            # Low pass filter
            b, a = signal.butter(5, carrier_freq * 2 / (bit_rate * 100), 'low')
            filtered_signal = signal.filtfilt(b, a, demodulated_signal)

            # Thresholding
            threshold = amp / 2
            recovered_data = np.zeros_like(data)
            for i in range(len(data)):
                if np.mean(filtered_signal[i*100:(i+1)*100]) > threshold:
                    recovered_data[i] = 1
                else:
                    recovered_data[i] = 0

            # Plot signals
            fig4, axs = plt.subplots(2, 1)

            axs[0].plot(time, demodulated_signal)
            axs[0].set_title('Demodulated signal')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Amplitude')

            axs[1].stem(range(len(recovered_data)), recovered_data)
            axs[1].set_title('Recovered data')
            axs[1].set_xlabel('Bit index')
            axs[1].set_ylabel('Amplitude')
            recovered_data_str = ''.join(map(str, recovered_data))
            st.write(f'Demodulated data : {recovered_data_str}')

            st.pyplot(fig4)

        elif modulation_scheme == "FSK" and demodulation == "FSK":
            st.subheader("FSK Demodulation")
            user_input = str(convolutional_encode_output)

            # User input for carrier frequencies

            f2 = f_fskmax
            f1 = f_fskmin

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
                        carrier1 = np.append(
                            carrier1, np.sin(2*np.pi*f1*t[:Fs]))
                        carrier2 = np.append(carrier2, np.zeros(Fs))
                        nrz = np.append(nrz, np.ones(Fs))
                    else:
                        x = np.append(x, np.sin(2*np.pi*f2*t[:Fs]))
                        carrier1 = np.append(carrier1, np.zeros(Fs))
                        carrier2 = np.append(
                            carrier2, np.sin(2*np.pi*f2*t[:Fs]))
                        nrz = np.append(nrz, -np.ones(Fs))

                # Multiply FSK signal with carrier frequencies
                demodulated_signal1 = x * carrier1
                demodulated_signal2 = x * carrier2

                # Low pass filter
                b, a = signal.butter(5, 0.1, 'low')
                filtered_signal1 = signal.filtfilt(b, a, demodulated_signal1)
                filtered_signal2 = signal.filtfilt(b, a, demodulated_signal2)

                # Decision making
                recovered_data = np.zeros_like(bit_sequence)
                for i in range(len(bit_sequence)):
                    if np.mean(filtered_signal1[i*Fs:(i+1)*Fs]) > np.mean(filtered_signal2[i*Fs:(i+1)*Fs]):
                        recovered_data[i] = 1
                    else:
                        recovered_data[i] = 0

                # Plot signals
                fig4, axs = plt.subplots(3, 1)

                axs[0].plot(t, demodulated_signal1)
                axs[0].set_title('Demodulated signal for bit 1')
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Amplitude')

                axs[1].plot(t, demodulated_signal2)
                axs[1].set_title('Demodulated signal for bit 0')
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Amplitude')

                axs[2].stem(range(len(recovered_data)), recovered_data)
                axs[2].set_title('Recovered data')
                axs[2].set_xlabel('Bit index')
                axs[2].set_ylabel('Amplitude')
                recovered_data_str = ''.join(map(str, recovered_data))
                st.write(f'Demodulated data : {recovered_data_str}')

                st.pyplot(fig4)

        elif modulation_scheme == "PSK" and demodulation == "PSK":
            st.subheader("PSK Modulation")
            user_input = str(convolutional_encode_output)

            # User input for carrier frequency
            f = f_psk

            if user_input and f:
                # Convert user input to list of integers
                bit_sequence = [int(bit) for bit in user_input]

                # Parameters
                T = 1   # bit duration
                Fs = 100  # sampling frequency

                t = np.arange(0, len(bit_sequence)*T, 1/Fs)
                x = np.array([])
                carrier = np.array([])
                nrz = np.array([])

                # Generate PSK signal, carrier frequency and NRZ signal
                for bit in bit_sequence:
                    if bit == 1:
                        x = np.append(x, np.sin(2*np.pi*f*t[:Fs]))
                        carrier = np.append(carrier, np.sin(2*np.pi*f*t[:Fs]))
                        nrz = np.append(nrz, np.ones(Fs))
                    else:
                        x = np.append(x, np.sin(2*np.pi*f*t[:Fs] + np.pi))
                        carrier = np.append(carrier, np.sin(2*np.pi*f*t[:Fs]))
                        nrz = np.append(nrz, -np.ones(Fs))

                # Plot PSK signal, carrier frequency and NRZ signal

                # Demodulation
                demodulated_signal = x * carrier

                # Low pass filter
                b, a = signal.butter(5, 0.1, 'low')
                filtered_signal = signal.filtfilt(b, a, demodulated_signal)

                # Decision making
                recovered_data = np.zeros_like(bit_sequence)
                for i in range(len(bit_sequence)):
                    if np.mean(filtered_signal[i*Fs:(i+1)*Fs]) > 0:
                        recovered_data[i] = 1
                    else:
                        recovered_data[i] = 0

                # Plot signals
                fig4, axs = plt.subplots(3, 1)

                axs[0].plot(t, demodulated_signal)
                axs[0].set_title('Demodulated signal')
                axs[0].set_xlabel('Time')
                axs[0].set_ylabel('Amplitude')

                axs[1].plot(t, filtered_signal)
                axs[1].set_title('Filtered signal')
                axs[1].set_xlabel('Time')
                axs[1].set_ylabel('Amplitude')

                axs[2].stem(range(len(recovered_data)), recovered_data)
                axs[2].set_title('Recovered data')
                axs[2].set_xlabel('Bit index')
                axs[2].set_ylabel('Amplitude')
                recovered_data_str = ''.join(map(str, recovered_data))
                st.write(f'Demodulated data : {recovered_data_str}')
                st.pyplot(fig4)

        # elif modulation_scheme == "QAM":
        #     st.subheader("QAM Demodulation")
        #     # st.subheader("QAM Modulation")
        #     user_input = str(convolutional_encode_output)

        #     # User input for carrier frequency
        #     f = st.number_input(
        #         'Enter the carrier frequency (Hz):', min_value=0.0)

        #     if user_input and f:
        #         # Convert user input to list of integers
        #         bit_sequence = [int(bit) for bit in user_input]

        #         # Parameters
        #         T = 1   # bit duration
        #         Fs = 100  # sampling frequency

        #         t = np.arange(0, len(bit_sequence)*T, 1/Fs)
        #         x = np.array([])
        #         carrier1 = np.array([])
        #         carrier2 = np.array([])
        #         nrz = np.array([])

        #         # Generate QAM signal, carrier frequencies and NRZ signal
        #         for i in range(0, len(bit_sequence), 2):
        #             if bit_sequence[i] == 1:
        #                 x = np.append(x, np.sin(2*np.pi*f*t[:Fs]))
        #                 carrier1 = np.append(
        #                     carrier1, np.sin(2*np.pi*f*t[:Fs]))
        #                 nrz = np.append(nrz, np.ones(Fs))
        #             else:
        #                 x = np.append(x, -np.sin(2*np.pi*f*t[:Fs]))
        #                 carrier1 = np.append(
        #                     carrier1, np.sin(2*np.pi*f*t[:Fs]))
        #                 nrz = np.append(nrz, -np.ones(Fs))

        #             if bit_sequence[i+1] == 1:
        #                 x += np.sin(2*np.pi*f*t[:Fs] + np.pi/2)
        #                 carrier2 = np.append(carrier2, np.sin(
        #                     2*np.pi*f*t[:Fs] + np.pi/2))
        #             else:
        #                 x -= np.sin(2*np.pi*f*t[:Fs] + np.pi/2)
        #                 carrier2 = np.append(carrier2, np.sin(
        #                     2*np.pi*f*t[:Fs] + np.pi/2))

        #         # Plot QAM signal and carrier frequencies
        #         fig, axs = plt.subplots(3, sharex=True)
        #         axs[0].plot(t, nrz)
        #         axs[0].set_xlabel('Time')
        #         axs[0].set_ylabel('Amplitude')
        #         axs[0].set_title('Data bits')

        #         axs[1].plot(t, carrier1)
        #         axs[1].set_ylabel('Amplitude')
        #         axs[1].set_title(f'In-phase carrier frequency ({f} Hz)')

        #         axs[2].plot(t, x)
        #         axs[2].set_ylabel('Amplitude')
        #         axs[2].set_title('QAM Signal')
        #         st.pyplot(fig)

        #         # # Demodulation
                # demodulated_signal1 = x * carrier1
                # demodulated_signal2 = x * carrier2

                # # Low pass filter
                # b, a = signal.butter(5, 0.1, 'low')
                # filtered_signal1 = signal.filtfilt(b, a, demodulated_signal1)
                # filtered_signal2 = signal.filtfilt(b, a, demodulated_signal2)

                # # Decision making
                # recovered_data = np.zeros_like(bit_sequence)
                # for i in range(0, len(bit_sequence), 2):
                #     if np.mean(filtered_signal1[i*Fs:(i+1)*Fs]) > 0:
                #         recovered_data[i] = 1
                #     else:
                #         recovered_data[i] = 0

                #     if np.mean(filtered_signal2[i*Fs:(i+1)*Fs]) > 0:
                #         recovered_data[i+1] = 1
                #     else:
                #         recovered_data[i+1] = 0

                # # Plot signals
                # fig4, axs = plt.subplots(4, 1)

                # axs[0].plot(t, demodulated_signal1)
                # axs[0].set_title('Demodulated signal for in-phase carrier')
                # axs[0].set_xlabel('Time')
                # axs[0].set_ylabel('Amplitude')

                # axs[1].plot(t, filtered_signal1)
                # axs[1].set_title('Filtered signal for in-phase carrier')
                # axs[1].set_xlabel('Time')
                # axs[1].set_ylabel('Amplitude')

                # axs[2].plot(t, demodulated_signal2)
                # axs[2].set_title('Demodulated signal for quadrature carrier')
                # axs[2].set_xlabel('Time')
                # axs[2].set_ylabel('Amplitude')

                # axs[3].stem(range(len(recovered_data)), recovered_data)
                # axs[3].set_title('Recovered data')
                # axs[3].set_xlabel('Bit index')
                # axs[3].set_ylabel('Amplitude')

                # st.pyplot(fig4)

    # elif modulation_scheme == "8QAM":
    #     st.subheader("8 QAM Demodulation")

    # elif modulation_scheme == "16QAM":
    #     st.subheader("16 QAM Demodulation")

    # elif modulation_scheme == "64QAM":
    #     st.subheader("64 QAM Demodulation")
    with st.expander("Channel Decoding"):
        st.subheader("Channel Decoding")
        # recovered_data_str = ''.join(map(str, recovered_data))
        G1 = [1, 1, 0, 1]
        G2 = [1, 0, 1, 1]

        def viterbi_decode(encoded_bits, G1, G2):
            # Define the trellis
            trellis = {(0, 0): ([], 0)}

            # Loop over the encoded bits
            for i in range(0, len(encoded_bits), 2):
                # Get the current parity bits
                p1, p2 = encoded_bits[i], encoded_bits[i+1]

                # Initialize the next trellis
                next_trellis = {}

                # Loop over the current trellis
                for state, (path, cost) in trellis.items():
                    # Compute the next states and their costs
                    next_state_0 = (0,) + state[:-1]
                    next_state_1 = (1,) + state[:-1]
                    next_cost_0 = cost + \
                        hamming_distance(
                            (p1, p2), convolutional_encode_state(next_state_0, G1, G2))
                    next_cost_1 = cost + \
                        hamming_distance(
                            (p1, p2), convolutional_encode_state(next_state_1, G1, G2))

                    # Update the next trellis
                    if next_state_0 not in next_trellis or next_trellis[next_state_0][1] > next_cost_0:
                        next_trellis[next_state_0] = (path + [0], next_cost_0)
                    if next_state_1 not in next_trellis or next_trellis[next_state_1][1] > next_cost_1:
                        next_trellis[next_state_1] = (path + [1], next_cost_1)

                # Update the trellis
                trellis = next_trellis

            # Find the minimum cost path
            min_cost = float('inf')
            min_path = None
            for state, (path, cost) in trellis.items():
                if cost < min_cost:
                    min_cost = cost
                    min_path = path

            return min_path

        def convolutional_encode_state(state, G1, G2):
            # Compute the parity bits
            p1 = sum([state[i] * G1[i+1] for i in range(len(state))]) % 2
            p2 = sum([state[i] * G2[i+1] for i in range(len(state))]) % 2

            return p1, p2

        def hamming_distance(a, b):
            return sum(x != y for x, y in zip(a, b))

        decoded_bits = viterbi_decode(output_bits, G1, G2)

        # Display the decoded bits to the user
        # st.write(output_bits)
        st.write('Demodulation Output:', convolutional_encode_output)
        st.write('Viterbi Decoding Output:', huffman_encoding_output)

    with st.expander("Source Decoding"):
        st.subheader("Source Decoding")

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

        # if user_input and len(user_input) >= 2:
        text = text_input

        # Convert analog signal to digital

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
            '5. "Wireless Communications: Principles and Practice" by Theodore S. Rappaport')
