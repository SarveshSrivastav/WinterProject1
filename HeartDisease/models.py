from sklearn.model_selection import train_test_split
from flask import Flask, render_template
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def correctPredictions(model, modelPredictions, test):
  count = 0     
  correct = 0
  for val in modelPredictions:
    if val == test[count]:
      correct += 1
    count += 1
  
  percent = "%s" % (round(100*(correct/len(test)),2)) + "% accuracy of " + model
  print(percent)
  print("Number Tested: " + str(count))
  print("Number Correct: " + str(correct))



#Decision Tree Method
def decisionTree(dataset):
  #Importing data and creating datasets
  
  data = pd.read_csv(dataset)
  X = data.iloc[:, :-1].values
  y = data.iloc[:, -1].values

  #Splitting Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  #Creating Model
  decisionTreeModel = DecisionTreeClassifier()

  #Training Model
  decisionTreeModel.fit(X_train, y_train)

  #Testing Model/Predicting Model
  decisionTreePredictions = decisionTreeModel.predict( X_test )

  correctPredictions("Decision Tree", decisionTreePredictions, y_test)
  return "Done!"



#Logistic Regression Method 
def logisticRegression(dataset):
  #Importing data and creating datasets
  
  data = pd.read_csv(dataset)
  X = data.iloc[:, :-1].values
  y = data.iloc[:, -1].values

  #Splitting Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # Feature Scaling
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #Creating Model
  logisticModel = LogisticRegression()

  #Training Model
  logisticModel.fit(X_train, y_train)

  #Testing Model/Predicting Model
  logisticModelPredictions = logisticModel.predict( X_test )
  countT = 0     
  correctT = 0
  for val in logisticModelPredictions:
    if val == y_test[countT]:
      correctT += 1
    countT += 1
  
  correctPredictions("Logistic Regression", logisticModelPredictions, y_test)
  return "Done!"

def supportVectorMachine(dataset):
  #Importing data and creating datasets

  data = pd.read_csv(dataset)
  X = data.iloc[:, :-1].values
  y = data.iloc[:, -1].values

  #Splitting Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # Feature Scaling
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #Creating Model
  supportVectorModel = SVC(kernel = 'poly', random_state = 0)

  #Training Model
  supportVectorModel.fit(X_train, y_train)

  #Testing Model/Predicting Model
  supportVectorMachinePredictions= supportVectorModel.predict( X_test )
  
  correctPredictions("Support Vector Machine", supportVectorMachinePredictions, y_test)
  return "Done!"

def randomForest(dataset):

 #Importing data and creating datasets
  
  data = pd.read_csv(dataset)
  X = data.iloc[:, :-1].values
  y = data.iloc[:, -1].values

  #Splitting Data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

  # Feature Scaling
  sc = StandardScaler()
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  #Creating Model
  randomForestModel = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

  #Training Model
  randomForestModel.fit(X_train, y_train)

  #Testing Model/Predicting Model
  randomForestPredictions= randomForestModel.predict( X_test )
  
  correctPredictions("Random Forest", randomForestPredictions, y_test)
  return "Done!"
