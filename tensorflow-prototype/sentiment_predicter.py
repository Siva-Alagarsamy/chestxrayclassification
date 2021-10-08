# This trainer will take training data from current folder /data/train and train a model
# The trained model will be saved as "export_model"

import re
import string
import tensorflow as tf
import sys

def predict():
    loaded_model = tf.keras.models.load_model('saved_model')
    
    while True:
        line = sys.stdin.readline()
        result = loaded_model.predict([line])
        print("result = ", result[0][0])


if __name__ == '__main__':
    predict()