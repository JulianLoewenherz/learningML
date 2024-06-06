import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv("Fish.csv")
print('Shape of dataset= ', df.shape) # To get no of rows and columns
df.head(5) # head(n) returns first n records only. Can also use sample(n) for random n records.

df.rename(columns={'Length1':'VerticalLen','Length2':'DiagonalLen','Length3':'CrossLen'},inplace = True) # 'inplace= true' to make change in current dataframe
print(df.sample(5)) # Display random 5 records

print(df.info())