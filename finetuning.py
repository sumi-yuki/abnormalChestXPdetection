# -*- coding: utf-8 -*-
"""
This is the finetuning program.

This program produces keras model: "xray_model_finetuning.h5" from "xray_model_original.h5"

To finetune the model, use chest X-ray:
finetuningXP_dir("finetuningXP")/
...normalXP_subdir("normalXP")/
......normal_1.jpg
......normal_2.jpg
...pneumoniaXP_subdir("pneumoniaXP")/
......pneumonia_1.jpg
......pneumonia_2.jpg

The original model was adapted from
 https://keras.io/examples/vision/xray_classification_with_tpus/
 modified by Yuki Sumi
 modification 1: Implementation for TPU was removed. 
 modification 2: Image resolution was increased from 180x180x3 to 360x360x3
   IMAGE_SIZE = [360, 360] <- [180, 180]
   I added the following convolutional layer to the model.
     conv_block(512, x)
     x = layers.Dropout(0.4)(x)
   And, I Increased the dropout rate of the last layer from 0.2 to 0.3.
     x = layers.Dropout(0.3)(x)
     
In this program, following parameters were changed
 patience = 100 <- 10
 initial_learning_rate = 0.001 <- 0.015
 decay_steps = 100 <- 100000
 decay_rate=0.96
    
The specification of the original model named as "xray_model_original.h5" created by pneumonia.py.
 Input [360,360] RGB[3] integer[0-255]
 Output [1]: probability for pneumonia 
    
"""

# Load modules
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# model
original_model = "xray_model_original.h5"
fine_tuned_model = "xray_model_finetuning.h5"

# directories 
finetuningXP_dir = "finetuningXP"
destination_normalXP_dir = "normalXP"
destination_pneumoniaXP_dir = "pneumoniaXP"
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 25
IMAGE_SIZE = [360, 360]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

# Check if GPU is working
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Create a training dataset.
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(".", finetuningXP_dir),
    label_mode = 'int',
    color_mode = 'rgb',
    # color_mode='grayscale',
    validation_split = 0.1,
    interpolation='lanczos3',
    shuffle=True, 
    subset = "both",
    seed = 123456,
    image_size = (IMAGE_SIZE[0], IMAGE_SIZE[1]),
    batch_size = BATCH_SIZE,
)

train_ds = ds[0]
val_ds = ds[1]                 
#print("datasets", ds)
#print("train datasets", train_ds)
#print("validation datasets", val_ds)

class_names = train_ds.class_names
print(class_names)  # ['normalXP', 'pneumoniaXP']
#print(class_names[0])  # normalXP

# For demonstration, iterate over the batches yielded by the dataset.
#for data, labels in train_ds:
#    print(data.shape)  # (25, 180, 180, 1)
#    print(data.dtype)  # <dtype: 'float32'>
#    print(data)  # float32
#    print(labels.shape)  # (25,)
#    print(labels.dtype)  # <dtype: 'int32'>
#    print(labels)  # tf.Tensor([0 0 1 1 0 1 0 0 0 0 0 0 1 1 1 0 0 0 0 1 0 1 0 0 0], shape=(25,), dtype=int32)

# Display 9 images with labels
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

def prepare_for_training(ds, cache=True):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

COUNT_NORMAL = len(
    [
        filename
        for filename in os.listdir(os.path.join(".", finetuningXP_dir, destination_normalXP_dir))
        if filename.endswith(".jpg") and not filename.startswith(".")
    ]
)
print("Normal images count in training set: " + str(COUNT_NORMAL))

COUNT_PNEUMONIA = len(
    [
        filename
        for filename in os.listdir(os.path.join(".", finetuningXP_dir, destination_pneumoniaXP_dir))
        if filename.endswith(".jpg") and not filename.startswith(".")
    ]
)
print("Pneumonia images count in training set: " + str(COUNT_PNEUMONIA))

initial_bias = np.log([COUNT_PNEUMONIA / COUNT_NORMAL])
print("Initial bias: {:.5f}".format(initial_bias[0]))

TRAIN_IMG_COUNT = COUNT_NORMAL + COUNT_PNEUMONIA
weight_for_0 = (1 / COUNT_NORMAL) * (TRAIN_IMG_COUNT) / 2.0
weight_for_1 = (1 / COUNT_PNEUMONIA) * (TRAIN_IMG_COUNT) / 2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print("Weight for class 0: {:.2f}".format(weight_for_0))
print("Weight for class 1: {:.2f}".format(weight_for_1))


checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "xray_model_finetuning.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=100, restore_best_weights=True
)

initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=1000, decay_rate=0.96, staircase=True
)

# Loads the original model
model = tf.keras.models.load_model("xray_model_original.h5")
# Check its architecture
model.summary()

METRICS = [
    tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss="binary_crossentropy",
    metrics=METRICS,
)

history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    class_weight=class_weight,
    callbacks=[checkpoint_cb, early_stopping_cb],
)

"""
## Visualizing model performance
"""

fig, ax = plt.subplots(1, 4, figsize=(20, 3))
ax = ax.ravel()

for i, met in enumerate(["precision", "recall", "binary_accuracy", "loss"]):
    ax[i].plot(history.history[met])
    ax[i].plot(history.history["val_" + met])
    ax[i].set_title("Model {}".format(met))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(met)
    ax[i].legend(["train", "val"])
plt.savefig("Finetuninglearning", dpi=600)
plt.show()

