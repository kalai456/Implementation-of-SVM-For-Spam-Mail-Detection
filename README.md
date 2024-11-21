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

x=data["v2"].values
y=data["v1"].values

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
![image](https://github.com/user-attachments/assets/a860bfb1-59aa-4b97-abe1-f5de190ca6c4)

## Data Information
![image](https://github.com/user-attachments/assets/5d8b0715-0334-4443-ac7e-f55010db845c)
![image](https://github.com/user-attachments/assets/425e2444-3950-49e3-b8a8-ba6165ca3ebf)


## Y-Predict
![image](https://github.com/user-attachments/assets/d978daa9-53bc-4bdf-b6c8-cbfde01e3fc3)

## Accuracy
![image](https://github.com/user-attachments/assets/18e87f4a-9a9b-477c-8016-94f4fee730bb)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
