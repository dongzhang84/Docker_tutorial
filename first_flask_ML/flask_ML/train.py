import pandas as pd
import joblib

from sklearn import svm 
from sklearn import metrics 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load data
df = pd.read_csv('iris.csv')


# dataset spliting    
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df.Species

# split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 24)

print("size of X_train: " ,X_train.shape)
print("size of y_train: " ,y_train.shape)
print("size of X_test: " ,X_test.shape)
print("size of y_test: " ,y_test.shape)

model = svm.SVC() # select the svm algorithm

# we train the algorithm with training data and training output
model.fit(X_train, y_train)

# we pass the testing data to the stored algorithm to predict the outcome
y_pred = model.predict(X_test)
print('The accuracy of the SVM is: ', metrics.accuracy_score(y_pred, y_test))

print(classification_report(y_test, y_pred))

joblib.dump(model, "model.pkl")