import numpy as np
import commpy.channelcoding.convcode as cc
import commpy.channelcoding.interleavers as RandInterlv
import commpy.channelcoding.turbo as tb
import commpy.utilities as util

import streamlit as st


def turbo_encode_decode():
    # Define the parameters for the turbo encoder and decoder

    # Information bits
    n_info_bits = 100

    # Encoder parameters
    generator_matrix = np.array([[0o7, 0o5]])
    constraint_length = 2

    # Interleaver parameters
    interleaver_length = n_info_bits
    interleaver = RandInterlv.generate_interleaver(
        interleaver_length, 'identity')

    # Modulation parameters
    modulation_order = 4  # QPSK
    snr_db = 10

    # Create the turbo encoder and decoder objects
    turbo_encoder = tb.TurboEncoder(generator_matrix, constraint_length)
    turbo_decoder = tb.TurboDecoder(
        generator_matrix, constraint_length, interleaver, maxiter=30, extrametric=True)

    # Generate random information bits
    info_bits = np.random.randint(0, 2, n_info_bits)

    # Encode the information bits using the turbo encoder
    encoded_bits = turbo_encoder.encode(info_bits)

    # Add noise to the encoded bits
    noise_var = 10 ** (-snr_db / 10) / (2 * np.log2(modulation_order))
    received_bits = util.awgn(encoded_bits, snr_db, noise_var)

    # Decode the received bits using the turbo decoder
    decoded_bits, extrinsic_info = turbo_decoder.decode(
        received_bits, snr_db, noise_var)

    # Calculate the bit error rate (BER)
    ber = np.sum(info_bits != decoded_bits) / n_info_bits

    # Display the results
    st.write("Turbo Encoding and Decoding using CommPy")
    st.write("Information bits: ", info_bits)
    st.write("Encoded bits: ", encoded_bits)
    st.write("Received bits: ", received_bits)
    st.write("Decoded bits: ", decoded_bits)
    st.write("Bit Error Rate: ", ber)


# Run the turbo encoding and decoding function in Streamlit
if __name__ == "__main__":
    turbo_encode_decode()
