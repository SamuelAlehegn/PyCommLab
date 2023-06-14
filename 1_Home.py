import streamlit as st

st.set_page_config(page_title="Python Based Digital Communication System Trainer",
                   layout="wide", initial_sidebar_state="auto")


def main():
    st.title("Python Based Digital Communication System Trainer")
    st.subheader(
        "Welcome to our Python-based digital communication system trainer homepage!")
    st.write("Our trainer is designed to help you learn the fundamentals of digital communication systems using the Python programming language. With our trainer, you can gain hands-on experience in building and simulating end to end digital communication systems.")
    st.write("Digital communication systems are everywhere around us, from the internet to wireless communication systems, and they play a crucial role in our daily lives. Understanding how these systems work is essential for anyone interested in a career in telecommunications, computer networking, or signal processing.")
    st.write("Our trainer is perfect for both beginners and experienced users who want to deepen their understanding of digital communication systems. You will learn how to use Python libraries such as NumPy, SciPy, Commpy,  and Matplotlib to simulate and visualize digital communication systems.")
    st.write("Whether you're a student, researcher, or industry professional, our Python-based digital communication system trainer is the perfect tool to help you learn and master the fundamentals of digital communication systems. Start your learning journey today and discover the power of Python for digital communication systems!")

    st.info("Complete Digital Communication Systems Block Diagram")
    st.image("file\img\Screenshot from 2023-01-03 23-03-13.png")


if __name__ == "__main__":
    main()
