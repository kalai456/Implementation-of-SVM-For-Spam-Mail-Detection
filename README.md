# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
 
## Algorithm
1. Load and Preprocess Data: Load the spam.csv dataset, check for missing values, and extract the text (v1) and labels (v2) for features and target.
2. Split Data: Split the data into training and testing sets (80/20 split).
3. Text Vectorization: Use CountVectorizer to convert the text data (x_train, x_test) into a matrix of token counts.
4. Train Model: Train a Support Vector Classifier (SVC) on the transformed training data (x_train, y_train).
5. Make Predictions: Use the trained model to predict labels for the test data (x_test).
6. Evaluate Model: Calculate and print the accuracy of the model by comparing predicted labels (y_pred) to the actual labels (y_test).






```
Program to implement the SVM For Spam Mail Detection..
Developed by: KALAISELVAN J
RegisterNumber:  212223080022
```
```
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
## Head Function
![image](https://github.com/user-attachments/assets/449d8e92-f550-4016-b3d9-437fe0f8a74a)
## Data Information
![image](https://github.com/user-attachments/assets/489dc53d-ae91-4040-9263-7918077625be)
## Y-Predict
![image](https://github.com/user-attachments/assets/b363dc1a-aa13-4018-872f-4dde9ff46898)
## Accuracy
![image](https://github.com/user-attachments/assets/339d38a5-c5d5-4aaa-af3a-cc11e2c5997e)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
