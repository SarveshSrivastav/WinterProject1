from HeartDisease import app
from HeartDisease.models import decisionTree, logisticRegression, supportVectorMachine, randomForest

if __name__ == '__main__':
  print(decisionTree('heart.csv') + "\n")
  print(logisticRegression('heart.csv') + "\n")
  print(supportVectorMachine('heart.csv') + "\n")
  print(randomForest('heart.csv') + "\n")
  
  app.run(host='0.0.0.0', port=8080)
  
