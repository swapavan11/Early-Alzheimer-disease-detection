import streamlit as st
import requests
from streamlit_lottie import st_lottie
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# Load the model
loaded_model = load_model('alzheimer_model.h5', compile=False)
loaded_model.compile(loss=SparseCategoricalCrossentropy(),
                     optimizer='adam',
                     metrics=['accuracy'])

# Function to load Lottie animation from URL
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

    
# Load assets
lottie_coding = load_lottieurl("https://lottie.host/6b8742e7-e4f5-4403-8bc0-1d5c2fae6e55/ah80z8X1Hu.json")
img_contact_form = Image.open("images/yt_contact_form.png")
img_lottie_animation = Image.open("images/yt_lottie_animation.png")

# Custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# Header
st.subheader("Hi, Good Morning! :wave:")
st.title("Welcome To Alzheimer Disease Detection Web App")
st.write("Alzheimer's disease is a progressive brain disorder that causes memory loss, cognitive decline, and changes in behavior.")
st.write("[Learn More >](https://en.wikipedia.org/wiki/Alzheimer%27s_disease)")
 
# Description of classes
st.write("---")
st.header("Description:")
st.write("##")
st.write("""
        We have divided Alzheimer's disease into four stages:
        - Non-Demented: Individuals with no signs of dementia.
        - Mild Demented: Individuals with mild cognitive impairment.
        - Moderate Demented: Individuals with moderate cognitive impairment.
        - Very Mild Demented: Individuals with very mild cognitive impairment."
        We aim to provide early detection of the disease using this web application, facilitating early diagnosis.
        """)

# Lottie animation
st_lottie(lottie_coding, height=300, key="coding")

# Input and Output section
st.header('Insert image for classification:')
upload = st.file_uploader('', type=['png', 'jpg'])

if upload is not None:
    im = Image.open(upload)
    im = im.convert('RGB')
    
    # Resize the image to match the expected input shape
    resized_im = im.resize((180, 180))
    
    img = np.asarray(resized_im)
    img_array = img.reshape(1, 180, 180, 3)

    st.header('Input Image')
    st.image(resized_im, use_column_width=False, width=180, caption='Uploaded Image')

    pred = loaded_model.predict([img_array, img_array, img_array])
    labels = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}

    st.header('Output')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Predicted class:')
        st.write(labels[pred.argmax()])
    with col2:
        st.subheader('With:')
        st.write(f'{int(pred.max() * 100)}% assurity')

# Contact section
st.write("---")
st.header("For any queries or suggestions!")
st.write("##")

contact_form = """
<form action="https://formsubmit.co/sainimohit572000@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
    <input type="text" name="name" placeholder="Your name" required>
    <input type="email" name="email" placeholder="Your email" required>
    <textarea name="message" placeholder="Your message here" required></textarea>
    <button type="submit">Send</button>
</form>
"""

left_column, right_column = st.columns(2)
with left_column:
    st.markdown(contact_form, unsafe_allow_html=True)
with right_column:
    st.empty()
