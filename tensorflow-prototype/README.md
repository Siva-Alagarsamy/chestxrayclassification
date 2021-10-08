# Tensorflow neural network model
Tensorflow using Keras API was used to build a deep neural network model. A [prototype was developed using Jupyter notebook](Sentiment%20Analysis%20using%20TensorFlow.ipynb). The script was then converted to command line.  

The model was trained using a command line [trainer script](sentiment_trainer.py). 

A script was written to [convert the dataset file into the Keras DataSet](convert_to_keras_dataset.py) directory structure. This script was executed to convert the full file to the Keras dataset folder and then the trainer script was executed to generate the model. The model is saved under a folder "saved_model"

A command line script to do prediction was also written. [The prediction script](sentiment_predicter.py) loads the saved model and allows user to enter a text on the command line. The script then uses the model to predict the sentiment. Any sentiment value less than 0.5 is considered negative sentiment and any sentiment value greater than 0.5 is considered positive.

The deep neural network model has an accuracy of 0.90

