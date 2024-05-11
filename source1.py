import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
df=sns.load_dataset('mpg')
df
df.isnull().sum()
df.dropna(inplace=True)
df
df.isnull().sum()
X = df[['displacement','horsepower','weight','acceleration']]
Y=df.mpg
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.15,random_state=42)
y_train
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train,y_train)

model.score(X_test,y_test)

from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(criterion="poisson",random_state=0)
model2.fit(X_train,y_train)

model2.score(X_test,y_test)

from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor(criterion="squared_error",random_state=0) #"squared_error","friedman_mse","absolute_error","poisson"
model2.fit(X_train,y_train)

import pickle
filename = 'mpg_regression.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open('mpg_regression.sav', 'rb'))