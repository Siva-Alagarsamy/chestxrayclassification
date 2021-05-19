# Google Sentiment Analysis
In this mini project, the Google Cloud Language API is used to analyze the sentiment of the reviews dataset. Since the performance of the API is not fast enough (1 analysis per second) and there is a quota on the maximum number of analysis. A sample dataset of 500 reviews was selected for this project. 

# Code
The code uses the bz2 module to open the file that is in the compressed in bz2 format. The lines are split to extact the label for the review. A label value of 1 indicates a negative sentiment and a value of 2 indicates a positive sentiment. The label and the review text is stored in an array.

You need a Google Cloud serivce account to use the API. The JSON file with the service account key is set as an environment variable for the Google Clound API to use. The review text is passed to the Google Clound API for analysis. The response has a score and a magnitude for the sentiment. A score of 0.2 or more is considered a positive sentiment and a value less that is considered a negative sentiment. The magnitude indicates the strength of the sentiment.  The score and the magnitude is saved in an array.

The arrays are converted to Pandas dataframe. With the Pandas dataframe we benchmark the Google cloud API performance. 

# Result
The Google Cloud API predicted the sentiment correctly for 448 reviews out of 500 reviews. 
