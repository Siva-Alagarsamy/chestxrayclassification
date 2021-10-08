# This trainer will take training data from current folder /data/train and train a model
# The trained model will be saved as "export_model"

import tensorflow as tf
from flask import Flask, request, jsonify
import json


app = Flask(__name__)
loaded_model = ""


@app.route('/check_sentiment', methods=['POST'])
def check_sentiment():
    post_data = json.loads(request.data)
    result = loaded_model.predict([post_data['text']])
    return jsonify({'result': str(result[0][0])})


if __name__ == '__main__':
    loaded_model = tf.keras.models.load_model('saved_model')
    app.run()

