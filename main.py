import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt
import uuid

# import tensorflow dependencies
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# import local dependencies
from network import siamese_model
from preprocess import train_data

# Avoid OOM Errors by settings the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Loss and optimizer
binary_cross_loss = tf.losses.BinaryCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

# Checkpoints
checkpoint_dir = './training_checkpoints'
# Create folder if it doesn't exist
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# TODO - Load the checkpoints if exists
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=opt, siamese_model=siamese_model)


# Train step
@tf.function  # convert to a tensorflow function graphable
def train_step(batch):
    # Automatic differentiation
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        X = batch[:2]
        # Get the labels
        y = batch[2]

        # Forward pass
        yhat = siamese_model(X, training=True)
        # Calculate the loss
        loss = binary_cross_loss(y, yhat)

    # Calculate the gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate updated weights and apply to the model
    opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return the loss
    return loss


# Training loop
def train(data, EPOCHS):
    # Loop through the epochs
    for epoch in range(1, EPOCHS + 1):
        print("Epoch {}/{}".format(epoch, EPOCHS))
        progbar = tf.keras.utils.Progbar(len(data))
        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train_step
            train_step(batch)
            progbar.update(idx + 1)

        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


# Train the model
EPOCHS = 50
train(train_data, EPOCHS)

