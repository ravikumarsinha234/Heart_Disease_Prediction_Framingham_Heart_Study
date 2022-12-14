# Heart_Disease_Prediction_Framingham_Heart_Study
π₯π« This repository is about using the machine learning algorithm for the prediction of heart disease on a person's heart in the next 10 years.

## Introduction about Logistic Regression
  
Many data science problems require us to give a βYesβ or βNoβ answer. One of the common methods
we use is logistic regression. Here we are going to split a data set into a training set
and testing set. We will will fit the model to the training set and test it with the test set. We will toy
with the threshold to get different levels of recall and precision.  

## Data Exploration
We are given framingham.csv which is a real data set from a famous study on coronary heart
disease: https://en.wikipedia.org/wiki/Framingham_Heart_Study  
We will create a program called chd explore.py that:  
β’ Reads framingham.csv into a pandas data frame.  
β’ Uses pandas profiling.ProfileReport to make a report.  
β’ Save the report to data report.html  
Open data report.html in a browser and look at the data. You are trying to predict TenYearCHD,
whether the person will have coronary heart disease event in the next 10 years.  

## Cleaning and Splitting of Data

Create another program called chd split.py that  
β’ Reads farmingham.csv into a data frame.  
β’ (You know from the report that some of the rows are missing values. ) Drops rows with missing values.  
β’ Uses sklearn.model selection.train test split to divide the dataframe into two dataframes: the test data frame will be a randomly chosen 20% of the rows, the train data frame will have the rest.  
β’ Save the data frames into test.csv and train.csv.  

## Fit the model to the training data

We will create another program called chd train.py that  
β’ Reads train.csv into a data frame.  
β’ Divides it into two numpy arrays: Y is TenYearCHD, X is the other columns.  
β’ Uses sklearn.preprocessing.StandardScaler to standardize the columns of X.  
β’ Uses sklearn.linear model.LogisticRegression to fit the data.  
β’ Prints out the accuracy of the model on the training data.  
β’ Saves the scaler and the logistic regression model to a single pickle file called classifier.pkl.  

## Test the model on the testing data

Create another program called chd test.py that  
β’ Configure a logger.  
β’ Reads test.csv into a data frame.  
β’ Divide it into numpy arrays X and Y , as above.  
β’ Loads the scaler and logistic regression model from classifier.pkl.  
β’ Apply the scaler to X so that it is scaled exactly as the training data was.  
β’ Print the accuracy of the model on the testing data.  
β’ Use sklearn.metrics.confusion matrix to print a confusion matrix.  
β’ Try 40 thresholds between 0 and 1. For each one, use the logger to print:  
&nbsp; β The threshold  
&nbsp; β The accuracy  
&nbsp; β The recall score  
&nbsp; β The precision score  
&nbsp; β The F1 score  
Like this: INFO@14:49:15: Threshold=0.220 Accuracy=0.776 Recall=0.50 Precision=0.32  
F1 = 0.393  
β’ Make another confusion matrix using the threshold that gave you the best F1 score.  
β’ Create a graph of the recall and precision vs. threshold as threshold.png  
