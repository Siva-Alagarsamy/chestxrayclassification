# Introduction
This project has the [code for training the sentiment analysis model](sentiment_trainer.py) using Tensorflow deep learning neural network. The data for this training has to be converted to the Keras format using the [convert script](convert_to_keras_dataset.py). The data is available under the data directory under the mllearn folder.

# Trainer
The [trainer code](sentiment_trainer.py) was based on the Python notebook prototype. A vector layer is used to convert the words into indices. The vector layer uses a vocabalary of 10000 words. The Embedding layer converts the word indices to a vector.  The convolution layer does a global 1D averaging on the vector. A Dense hidden layer is used and finally a Dense output layer is used to output a single output for the sentiment. We also use Dropout layer to insert random dropouts during training to avoid overfitting. 
The trainer then saves the model along with the vector layer and a sigmoid layer to generate a value between 0 and 1. 

# REST API
The [REST API server](sentiment_rest_server.py) uses flask to serve the saved model for prediction. The web server loads the saved model and adds a route /check_sentiment which calls the function "check_sentiment" which uses the model to do the prediction and returns the result as a JSON message with result value. 

**URL:** /check_sentiment

**Method:** POST

**Request Type:**  application/json

**Post Content:** JSON message with "text" field. For example:

    {
      "text" : "Text to be analyzed"
    }
	
**Response Type:** application/json

**Response Content:** A JSON message with the "result" field. For example:

    {
       "result" : 0.75
    }
	
 Any value less then 0.5 shall be considered a negative sentiment and any value greater than 0.5 shall be considered a positive sentiment. 

# Web Page
 A simple web page was developed to use the REST API and provide the end user a easy way to test the API. The [index.html](index.html) is the main page. The [index.js](index.js) has the javascript code to make the REST API and update the results. The [index.css](index.css) has styles to format the webpage.


# Docker Image
 The [Dockerfile](Dockerfile) has the docker image definition. A docker image can be easily built with the Dockerfile and the image can be deployed on the cloud. 