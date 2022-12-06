from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

"""
# Andrew Cronin's Final Project!

This should predict 


"""


with st.echo(code_location='below'):
    
df = pd.read_csv('highest.csv', header =0)

y_data = df['WealthDegree']
y_data = y_data.astype('int')

x_data = df.drop('WealthDegree', axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.3)

clf = DecisionTreeClassifier(random_state=123)
params =  {
    'min_weight_fraction_leaf': [0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth': [1, 3, 5, 7, 9, 11],
    'max_leaf_nodes':[None,20,40,60,80] 
}
grid = GridSearchCV(estimator=clf,
                    param_grid=params,
                    cv=10,
                    n_jobs=1,

)
grid.fit(x_train, y_train)
print(grid.best_score_)
print(grid.best_params_)
