# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Convert the Alphabetical data to numeric using CountVectorizer.

Step 7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 8. Find the accuracy of the model.

Step 9. End the Program.
## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: P PARTHIBAN
RegisterNumber:  212223230145
```
```python
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
