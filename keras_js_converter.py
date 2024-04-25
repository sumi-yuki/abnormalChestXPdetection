#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 17:53:51 2023
@author: Yuki Sumi

This program converts python Keras model: "xray_model_finetuning.h5" to TensorFlow.js model "tmdu_pneumonia_model_js"
https://www.tensorflow.org/js/tutorials/conversion/import_keras?hl=en

"""

# Load the python tensorflow model
import tensorflow as tf
model = tf.keras.models.load_model("xray_model_finetuning.h5")

# Check its architecture
model.summary()

# save the trained model for javascript
import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, "tmdu_pneumonia_model_js")

# replace all "kernel_initializer" with "depthwise_initializer", "kernel_regularizer" with "depthwise_intializer" and "kernel_contraint" with "depthwise_constraint" in model.json file.
# https://github.com/tensorflow/tfjs/issues/1739

#open text file in read mode
text_file = open("./tmdu_pneumonia_model_js/model.json", "r")
#read whole file to data
data = text_file.read()
#close file
text_file.close()
 
print("Before replacement:\n",data)
data = data.replace("kernel_initializer", "depthwise_initializer")
data = data.replace("kernel_regularizer", "depthwise_intializer")
data = data.replace("kernel_contraint", "depthwise_constraint")
print("\nAfter replacement:\n",data)

#open text file in write mode
text_file = open("./tmdu_pneumonia_model_js/model.json", "w")
#write data to the filea
text_file.write(data)
#close file
text_file.close()
