#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.express as px

#To ignore warning messages
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv(r"C:\Users\amani\Downloads\salarydata.csv") # reading the dataset
data.head()

data.shape

data.isin(['?']).sum() # dataset contains meaningless values of '?' in certain columns

data = data.replace(to_replace='?', value=np.nan) # replacing fields having '?' with null values

# Making sure that the data does not contain unecessary spaces.
data=data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# converting the target column into numerical classes (values of either 0 or 1).
data['salary'] = data['salary'].replace({'<=50K':0,'>50K':1})

uniq= pd.DataFrame(data.nunique(),columns=['Unique Values'])
uniq

data.isna().sum()

# filling missing values
msv_col = ['workclass','occupation','native-country']
for col in msv_col:
    data[col] = data[col].fillna(data[col].mode()[0])
    
    data.isna().sum()
    
    # education & education number column are just the same, so dropping education number column.
data.drop(labels='education-num', axis=1, inplace=True)

# Transforming Maritial Status column with value as either married or not married.
data = data.replace({'Married-civ-spouse':'married','Married-AF-spouse':'married','Married-spouse-absent':'married',
                    'Never-married':'not married','Divorced':'not married','Separated':'not married','Widowed':'not married'})
                    
from sklearn.preprocessing import LabelEncoder
column = ['workclass', 'education', 'marital-status', 'occupation','relationship', 'race', 'sex', 'native-country']
le = LabelEncoder()

for col in column:
    data[col] = le.fit_transform(data[col])
data.head()

data.drop(['capital-gain','capital-loss'],axis=1,inplace=True)
data.head()

X=data.drop('salary',axis=1)
y=data['salary']

# Feature scaling on training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X[['age',  'hours-per-week']])
input_scaled = scaler.transform(X[['age',  'hours-per-week']])
scaled_data = pd.DataFrame(input_scaled,columns=['age',  'hours-per-week'])

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

from sklearn.linear_model import LogisticRegression
#Defining Logistic Regression Model & fitting train data
lr=LogisticRegression()
logit_model=lr.fit(X_train,y_train)
#Predicting the result of test data using obtained model
y_pred_logit=logit_model.predict(X_test)

from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
precision_score(y_test,y_pred_logit)

accuracy_score(y_test,y_pred_logit)

X=data.drop('salary',axis=1)
y=data['salary']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf = dt_clf.fit(X_train,y_train)
y_pred_dt = dt_clf.predict(X_test)

accuracy_score(y_test,y_pred_dt)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
model_rf=rf.fit(X_train,y_train)
y_pred_rf=model_rf.predict(X_test)

accuracy_score(y_test,y_pred_rf)

# fit model no training data
from numpy import loadtxt
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
svc=SVC(kernel='rbf')
svc.fit(X_train,y_train)
Y_pred=svc.predict(X_test)
print('Accuracy on training data is:',svc.score(X_train,y_train))
print('Accuracy is:',accuracy_score(y_test,y_pred))
print('Precision is:',precision_score(y_test,y_pred,average='weighted'))
print('Recall is:',recall_score(y_test,y_pred,average='weighted'))
print('f1 score is:',f1_score(y_test,y_pred,average='weighted'))

from sklearn.model_selection import GridSearchCV
# Creating the hyperparameter grid
param_grid = {'C': [1,10,100,1000]}
# Instantiating logistic regression classifier
logreg = LogisticRegression()
# Instantiating the GridSearchCV object
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)
logreg_cv.fit(X_train, y_train)
logreg_cv.predict(X_test)
# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))

pd.Series(model_rf.feature_importances_,index=X.columns).sort_values(ascending=False)*100

param_grid = { 
    'n_estimators': [100,200,300],
    'max_depth' : [20,25,30],
    'criterion' :['gini', 'entropy']
}

from sklearn.model_selection import GridSearchCV
CV_rfc = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv= 5)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_

rft = RandomForestClassifier(n_estimators=300,max_depth=25,random_state=42,criterion='entropy')
rft.fit(X_train,y_train)

y_pred=rft.predict(X_test)
accuracy_score(y_test,y_pred)

X=data.iloc[:,[0,1,2,3,4,5,6,7,8,9]].values
X=np.array(X)

y=data['salary'].values
y=np.array(y)

import pickle
filename = 'salary_pred_model.pkl'
pickle.dump(rft,open(filename,'wb'))
loaded_model = pickle.load(open(filename,'rb'))