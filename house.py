import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv(
    r'house_data.csv')

columns = ['bedrooms', 'bathrooms','sqft_living', 'floors', 'yr_built', 'price']
df = df[columns]

X = df.iloc[:, 0:5]
y = df.iloc[:, 5:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, y_train)

pickle.dump(lr, open('modelf.pkl', 'wb'))
