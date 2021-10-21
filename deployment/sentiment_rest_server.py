# This server will server the web page and also server the REST API to do the sentiment analysis

import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
import json


# Create the flask app
app = Flask(__name__)
# Loaded model global variable
loaded_model = ""


# REST API to check sentiment. The post context has the text to analyze.
# The response has the result number. 0.5 < negative, 0.5 > positive sentiment.
@app.route('/check_sentiment', methods=['POST'])
def check_sentiment():
    post_data = json.loads(request.data)
    result = loaded_model.predict([post_data['text']])
    return jsonify({'result': str(result[0][0])})


# Send any file from current directory
@app.route('/<path:path>')
def send_web_file(path):
    return send_from_directory('.', path)


# Send the index.html file if no path
@app.route('/')
def send_index_file():
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':
    # Load the saved model
    loaded_model = tf.keras.models.load_model('lstm_model')
    app.run(host='0.0.0.0')

