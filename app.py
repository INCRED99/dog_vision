import gdown
import os

if not os.path.exists("dog_breed.h5"):
    gdown.download(
        "https://drive.google.com/uc?id=1Jc5fFV7AhL4noXXFbLfmBO_mqJ6dIKrV",
        "dog_breed.h5", quiet=False
    )

import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Dog Breed Classifier", page_icon="🐶")

# -------------------------------
# Load model (cached)
# -------------------------------
@st.cache_resource
def load_dog_breed_model():
    
    class HubWrapper(tf.keras.layers.Layer):
        def __init__(self, handle=None, trainable=False, **kwargs):
            if handle is None:
                handle = kwargs.pop('model_url', None)
            super().__init__(**kwargs)
            self.handle = handle
            self.layer = hub.KerasLayer(self.handle, trainable=trainable)

        def call(self, inputs):
            return self.layer(inputs)

        def get_config(self):
            config = super().get_config()
            config['handle'] = self.handle
            return config

        @classmethod
        def from_config(cls, config):
            handle = config.pop('model_url', None) or config.pop('handle', None)
            return cls(handle=handle, **config)

    with tf.keras.utils.custom_object_scope({
        'CustomLayers>HubWrapper': HubWrapper
    }):
        model = tf.keras.models.load_model("dog_breed.h5", compile=False)
    return model

model = load_dog_breed_model()

# -------------------------------
# Load labels
# -------------------------------
with open("labels.txt") as f:
    class_names = [line.strip() for line in f.readlines()]

# -------------------------------
# UI
# -------------------------------
st.title("🐶 Dog Breed Classifier")
st.caption("Upload a dog photo and the model will predict its breed.")
st.divider()

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    col1, col2 = st.columns(2, gap="large")

    with col1:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    preds = model.predict(img_array)
    class_index = np.argmax(preds[0])
    confidence = preds[0][class_index]

    with col2:
        st.subheader("Results")
        st.metric(label="Predicted Breed", value=class_names[class_index])
        st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
        st.caption(f"Class index: {class_index}")