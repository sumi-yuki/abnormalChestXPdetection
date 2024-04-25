# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:37:25 2023

@author: Yuki Sumi

This program evaluate the keras models which classify chest X-ray images as normal or pneumonia
There are two models:
 original_model = "xray_model_original.h5"
 fine_tuned_model = "xray_model_finetuning.h5"

keras models
 -> Input [360,360] RGB[3] integer[0-255]
 -> Output [1]: probability for pneumonia 
     
"xray_model_original.h5" is the original model made by "pneumonina.py" 
 adapted from
 https://keras.io/examples/vision/xray_classification_with_tpus/
 modified by Yuki Sumi
 
"xray_model_finetuning.h5" is the fine-tuned model from "xray_model_original.h5" by "finetuning.py"

To evaluate model, use chest X-ray images:
testXP_dir("testXP")/
...normalXP_subdir("normalXP")/
......normal_1.jpg
......normal_2.jpg
...pneumoniaXP_subdir("pneumoniaXP")/
......pneumonia_1.jpg
......pneumonia_2.jpg

"""

# installing modules

# tensorflow https://www.tensorflow.org/?hl=en
# To install,
# pip install tensorflow==2.10.0

# Load modules
import os
import tensorflow as tf

# model
original_model = "xray_model_original.h5"
fine_tuned_model = "xray_model_finetuning.h5"

# directories 
testXP_dir = "testXP"
normalXP_subdir = "normalXP"
pneumoniaXP_subdir = "pneumoniaXP"
BATCH_SIZE = 25
IMAGE_SIZE = [360, 360]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# The following code is not really necessary, but I added it to prevent OMP Abort error in Anaconda environment.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

"""
## Predict using the model
"""

# Create a test dataset.
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(".", testXP_dir),
    label_mode='int',
    color_mode='rgb',
    # color_mode='grayscale',
    interpolation='lanczos3',
    shuffle=True, #for Debugging
    # shuffle=False,
    seed=None,
    validation_split=None,
    subset=None,
    image_size=(IMAGE_SIZE[0], IMAGE_SIZE[1]),
    batch_size=BATCH_SIZE,
)
#print("datasets", ds)


"""
## Evaluate model using model.evaluate
"""

model = tf.keras.models.load_model(original_model)
print("Evaluating model:",original_model)
results = model.evaluate(ds, return_dict=True)
print("test model:",results)

model = tf.keras.models.load_model(fine_tuned_model)
print("Evaluating model:",fine_tuned_model)
results = model.evaluate(ds, return_dict=True)
print("test model:",results)
