# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Load the dataset
dataset = pd.read_csv('hiring.csv')

# Fill missing values
dataset['experience'] = dataset['experience'].fillna(0)
dataset['test_score'] = dataset['test_score'].fillna(dataset['test_score'].mean())

# Convert experience to integers
def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0:0}
    return word_dict[word]

dataset['experience'] = dataset['experience'].apply(lambda x: convert_to_int(x))

# Define features and labels
X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]

# Train the Linear Regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Save the model
pickle.dump(regressor, open('model.pkl', 'wb'))

# Test the model
model = pickle.load(open('model.pkl', 'rb'))
input_data = pd.DataFrame([[2, 9, 6]], columns=['experience', 'test_score', 'interview_score'])
print(model.predict(input_data))
