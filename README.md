# Introduction
This is the repo for the ML project that I am working on as part of the [Machine Learning course at Springboard](https://www.springboard.com/courses/ai-machine-learning-career-track).

[Different dataset for this project was explored](DatasetExplored.md) and the Amazon reviews dataset was selected for this project. This project will use the Amazon review dataset to build a model that will predict the sentitment based on a given review

# Google Language Service benchmark
A sample dataset was provided to the Google Cloud language service to analyze the sentiment. This is under the [Google language benchmark](google-benchmark)

# Explorative Data Analysis
[Explorative Data Analysis](Data%20Wrangling%20and%20Exploration) was performed on the dataset. The folder "Data Wrangling and Exploration" contains the Jupyter notebook files.

# Benchmark different model types
[Three different model types](Benchmark%20Various%20Models) Logistic Regression, Random Forest and XGBoost was tried on the dataset. The folder "Benchmark Various Models" contains the Jupyter notebook.

# Prototypes
The folder [Prototypes](Prototypes) contains initial prototype using the Logistic regression.

# Tensorflow neural network model
Tensorflow using Keras API was used to build a deep neural network model. The model was trained using a command line [trainer script](tensorflow-prototype/sentiment_trainer.py). 

A script was written to [convert the dataset file into the Keras DataSet](tensorflow-prototype/convert_to_keras_dataset.py) directory structure. This script was executed to convert the full file to the Keras dataset folder and then the trainer script was executed to generate the model. The model is saved under a folder "saved_model"

A command line script to do prediction was also written. [The prediction script](tensorflow-prototype/sentiment_predicter.py) loads the saved model and allows user to enter a text on the command line. The script then uses the model to predict the sentiment. Any sentiment value less than 0.5 is considered negative sentiment and any sentiment value greater than 0.5 is considered positive.

The deep neural network model has an accuracy of 0.90

