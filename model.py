
#@author: kankk

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv("C:\\Users\kankk\OneDrive\Desktop\deployment\deployement using heroku of Machine learning model\Salary Data.csv")
from sklearn.preprocessing import LabelEncoder
lc = LabelEncoder()

dataset['Gender'] = lc.fit_transform(dataset['Gender'])
X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[22, 1, 6]]))