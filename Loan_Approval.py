import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
data = pd.read_csv("LoanApprovalPrediction.csv") 
data.head(5)
obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))
# Import label encoder 
from sklearn import preprocessing 
# label_encoder object knows how  
# to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
 data[col] = label_encoder.fit_transform(data[col])
 # To find the number of columns with  
# datatype==object 
obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f',linewidths=2,
annot=True)
plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f', linewidths=2,annot=True)
for col in data.columns: 
 data[col] = data[col].fillna(data[col].mean())  
data.isna().sum()
from sklearn.model_selection import train_test_split 
X = data.drop(['Loan_Status'],axis=1) 
Y = data['Loan_Status'] 
X.shape,Y.shape X_train, X_test, Y_train, Y_test =
 train_test_split(X, Y, test_size=0.4,random_state=1) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
