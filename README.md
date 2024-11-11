# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
Step 1. Start the Program.

Step 2. Import the necessary packages.

Step 3. Read the given csv file and display the few contents of the data.

Step 4. Assign the features for x and y respectively.

Step 5. Split the x and y sets into train and test sets.

Step 6. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.

Step 7. Find the accuracy of the model.

Step 8. End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Naveenaa A K
RegisterNumber:  212222230094
```
```
import chardet

file ='spam.csv'
with open (file, 'rb' )as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding = 'Windows-1252')
data.head()

data.info()

data.isnull().sum()

x = data["v2"].values
y = data["v1"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train,y_train)
y_pred = svc.predict(x_test)
print("Predicted value : ",y_pred)

from sklearn import metrics 
accuracy = metrics.accuracy_score(y_test,y_pred)
print("Accuracy: ",accuracy)

```
## Output:
### detecting the character encoding:
![image](https://github.com/user-attachments/assets/27035b25-58ce-4f1a-84d6-00149d2bc787)
### head():
![image](https://github.com/user-attachments/assets/6ea9469d-b23e-45be-bcd1-f73f6f6335b9)
### .info()
![image](https://github.com/user-attachments/assets/bddcf3a2-e9c7-4a2b-800c-754adad49420)
### checking for null values :
![image](https://github.com/user-attachments/assets/c5be687a-7a29-4934-8ea0-16f0cdb38c50)
### Predicted values :
![image](https://github.com/user-attachments/assets/31fcc5c8-4cf5-4cdd-8bd8-47fce7cf0eee)
### Accuracy :
![image](https://github.com/user-attachments/assets/1393653c-fb3c-4f4c-8967-dfceead18665)




## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
