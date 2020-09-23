#!/usr/bin/env python
# coding: utf-8

# In[28]:


from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
import tensorflow as tf
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# In[29]:


# Define a flask app
app = Flask(__name__)


# In[30]:


MODEL_PATH = 'D:/deploy/road_crack_new.h5'


# In[31]:


model = tf.keras.models.load_model(MODEL_PATH)
model._make_predict_function()


# In[32]:


def model_predict(img_path, model):
    img = cv2.imread(os.path.join(img_path),0)
    img_size = 224
    th = 1
    max_value = 255
    blocksize  = 79
    constant = 2
    img_f = cv2.bilateralFilter(img,9,75,75)
    ret, o2 = cv2.threshold(img_f, th, max_value, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    x = cv2.resize(o2,(img_size,img_size))
    x = x/255
    x = np.array(x).reshape(-1,img_size,img_size,1)
    preds = model.predict(x)
    return preds


# In[33]:


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


# In[37]:


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = 'D:/deploy'
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        preds = model_predict(file_path, model)
        negative = round(preds[0][0],5)
        positive = round(preds[0][1],5)
        if negative >= 0.5:
            result = 'Negative - No crack found'
        else:
            result = 'Positive - Crack found'
        return result
    return None


# In[38]:


if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




