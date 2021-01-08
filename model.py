import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('Tata_Steel.csv')
df=df.dropna()

df.isnull().sum()

X=df.iloc[:,:-1]
y=df['DS_S']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

print("Coefficient of determination R^2 <-- on train set: {}".format(regressor.score(X_test, y_test)))

import pickle
file=open('regression2_model.pkl','wb')
pickle.dump(regressor,file)