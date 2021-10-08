# This trainer will take training data from current folder /data/train and train a model
# The trained model will be saved as "export_model"

import re
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Set the batch size for both dataset and training.
batch_size = 32

# We will be splitting the training dataset into train and validation set.
# We need the same random seed to be used for load_train_dataset and load_val_dataset.
train_valid_random_seed = 42

# Vectorization layer
vectorization_layer: TextVectorization


# This function loads the training dataset.
def load_train_dataset():
    return tf.keras.preprocessing.text_dataset_from_directory(
        '../data/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=train_valid_random_seed)


# This function loads the validation dataset from the data/train directory, but it returns the 20% of the data.
# Note we need to use the same random seed to make sure the data that was returned for training is not in this.
def load_val_dataset():
    return tf.keras.preprocessing.text_dataset_from_directory(
        '../data/train',
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=train_valid_random_seed)


# This function loads the test dataset from the data/test directory
def load_test_dataset():
    return tf.keras.preprocessing.text_dataset_from_directory(
        '../data/test',
        batch_size=batch_size)


# Define the vectorization function
def text_to_vector(text, label):
    text = tf.expand_dims(text, -1)
    return vectorization_layer(text), label


# Train method. Loads the datasets and trains the model and
def train():
    max_features = 10000
    sequence_length = 250

    # Load all datasets
    raw_train_ds = load_train_dataset()
    raw_val_ds = load_val_dataset()
    raw_test_ds = load_test_dataset()

    # Define the vectorization layer
    global vectorization_layer
    vectorization_layer = TextVectorization(
        standardize='lower_and_strip_punctuation',
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Adapt the vectorization layer
    vectorization_layer.adapt(raw_train_ds.map(lambda x, y: x))

    # Vectorize the dataset
    train_ds = raw_train_ds.map(text_to_vector)
    val_ds = raw_val_ds.map(text_to_vector)
    test_ds = raw_test_ds.map(text_to_vector)

    auto_tune = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=auto_tune)
    val_ds = val_ds.cache().prefetch(buffer_size=auto_tune)
    test_ds = test_ds.cache().prefetch(buffer_size=auto_tune)

    embedding_dim = 16

    # Define the model
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(0.2),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.2),
        layers.Dense(1)])

    # Compile the model
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

    # Train the model
    epochs = 10
    csv_logger = CSVLogger('training.log')
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[csv_logger])

    # Test the model
    loss, accuracy = model.evaluate(test_ds)
    print("Model loss against test dataset = ", loss)
    print("Model accuracy against test dataset = ", accuracy)

    # Save the model 
    export_model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype="string"),
        vectorization_layer,
        model,
        layers.Activation('sigmoid')
    ])

    export_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )
    export_model.save('saved_model')


if __name__ == '__main__':
    train()