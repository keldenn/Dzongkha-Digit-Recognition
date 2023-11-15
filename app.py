import streamlit as st
import tensorflow as tf
import os
from tensorflow import keras
import numpy as np
# from scipy.ndimage.interpolation import zoom
from scipy.ndimage import zoom
from streamlit_drawable_canvas import st_canvas

# Load trained model
model = keras.models.load_model(os.path.abspath('dzongkha_digits_classifier.h5'))

def process_image(image_data, size=224):
    """Convert drawn image to grayscale, resize to 224x224, and add color channel."""
    # Convert image to grayscale
    grayscale_image = np.sum(image_data, axis=2)
    # Resize image to 224x224
    resized_image = zoom(grayscale_image, size / grayscale_image.shape[0])
    # Add color channel to match the model's input shape
    processed_image = resized_image.reshape((224, 224, 1))
    # Stack the same image along the color channels to make it (224, 224, 3)
    processed_image = np.dstack([processed_image] * 3)
    # Normalize pixel values
    normalized_image = processed_image.astype(np.float32) / 255
    # Return processed image
    return normalized_image.reshape(1, 224, 224, 3)



st.markdown(
  "<div style='padding-top: 0px; margin-top: 0px;'><span style='font-size: 90px; color: #FFCD00; font-weight: bold;'>རྫོང་</span> <span style='font-size: 90px; color: #FF6720; font-weight: bold; margin-right: 4px;'>ཁ་</span> <span style='font-size: 48px; color: black; font-weight: bold;'>Digit Recognition🤖</span></div>",
    unsafe_allow_html=True
)

# st.markdown("# :orange[****རྫོང་ཁ****་] Digit :blue[Recognition] 🤖")


col1, col2 = st.columns([1, 1])
with col1:
    st.header('Draw a digit')
    # Display canvas for drawing
    st.markdown("<style>div.Widget.row-widget.stButton > div{flex-direction: column; justify-content: center; align-items: center;}</style>", unsafe_allow_html=True)
    canvas_result = st_canvas(stroke_width=10, height=60*5, width=60*5)
with col2:
    # Process drawn image and make prediction using model
    if np.any(canvas_result.image_data):
        #st.write(canvas_result.image_data)
        # Convert drawn image to grayscale and resize to 28x28
        processed_image = process_image(canvas_result.image_data)
        # Make prediction using model
        prediction = model.predict(processed_image).argmax()
        # Display prediction
        st.header('Prediction')
        st.markdown('This number appears to be a \n # :red[' + str(prediction) + ']')
    else:
        # Display message if canvas is empty
        st.header('Prediction:')
        st.write('Please draw a digit to get a prediction.')





