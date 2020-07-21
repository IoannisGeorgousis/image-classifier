from PIL import Image
import numpy as np
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import json

# Create the process_image function
def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image.numpy()


# Create the predict function
def predict(image_path, model, top_k=5):
    
    image = Image.open(image_path)              # Load image
    image = np.asarray(image)                   # Convert to ndarray
    image = process_image(image)                # Process image
    image = np.expand_dims(image, axis=0)       # Add extra dimension
    
    y = model.predict(image)                    # Make prediction
    
    
    # Calculate top K classes and probilities
    y = y[0]                                    
    indices = y.argsort()[::-1][:top_k]   
    probs = y[indices]
    classes = indices + 1
    
    return probs, classes
