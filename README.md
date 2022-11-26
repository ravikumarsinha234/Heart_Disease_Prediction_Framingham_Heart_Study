# Heart_Disease_Prediction_Framingham_Heart_Study
🏥🫀 This repository is about using the machine learning algorithm for the prediction of heart disease on a person's heart in the next 10 years.

## Introduction about Logistic Regression
  
Many data science problems require us to give a ”Yes” or ”No” answer. One of the common methods
we use is logistic regression. Here we are going to split a data set into a training set
and testing set. We will will fit the model to the training set and test it with the test set. We will toy
with the threshold to get different levels of recall and precision.  

## Data Exploration
We are given framingham.csv which is a real data set from a famous study on coronary heart
disease: https://en.wikipedia.org/wiki/Framingham_Heart_Study  
We will create a program called chd explore.py that:  
• Reads framingham.csv into a pandas data frame.  
• Uses pandas profiling.ProfileReport to make a report.  
• Save the report to data report.html  
Open data report.html in a browser and look at the data. You are trying to predict TenYearCHD,
whether the person will have coronary heart disease event in the next 10 years.  

## Cleaning and Splitting of Data

Create another program called chd split.py that  
• Reads farmingham.csv into a data frame.  
• (You know from the report that some of the rows are missing values. ) Drops rows with missing values.  
• Uses sklearn.model selection.train test split to divide the dataframe into two dataframes: the test data frame will be a randomly chosen 20% of the rows, the train data frame will have the rest.  
• Save the data frames into test.csv and train.csv.  

## Fit the model to the training data

We will create another program called chd train.py that  
• Reads train.csv into a data frame.  
• Divides it into two numpy arrays: Y is TenYearCHD, X is the other columns.  
• Uses sklearn.preprocessing.StandardScaler to standardize the columns of X.  
• Uses sklearn.linear model.LogisticRegression to fit the data.  
• Prints out the accuracy of the model on the training data.  
• Saves the scaler and the logistic regression model to a single pickle file called classifier.pkl.  

## Test the model on the testing data

Create another program called chd test.py that  
• Configure a logger.  
• Reads test.csv into a data frame.  
• Divide it into numpy arrays X and Y , as above.  
• Loads the scaler and logistic regression model from classifier.pkl.  
• Apply the scaler to X so that it is scaled exactly as the training data was.  
• Print the accuracy of the model on the testing data.  
• Use sklearn.metrics.confusion matrix to print a confusion matrix.  
• Try 40 thresholds between 0 and 1. For each one, use the logger to print:  
&nbsp; – The threshold  
&nbsp; – The accuracy  
&nbsp; – The recall score  
&nbsp; – The precision score  
&nbsp; – The F1 score  
Like this: INFO@14:49:15: Threshold=0.220 Accuracy=0.776 Recall=0.50 Precision=0.32  
F1 = 0.393  
• Make another confusion matrix using the threshold that gave you the best F1 score.  
• Create a graph of the recall and precision vs. threshold as threshold.png  
