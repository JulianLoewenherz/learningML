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

#data visualization
df_sp = df.Species.value_counts()
df_sp = pd.DataFrame(df_sp)
df_sp.T 

#plotting data
sns.barplot(x= df_sp.index, y = df_sp.Species) # df_sp.index will returns row labels of dataframe
plt.xlabel('Species')
plt.ylabel('Count of Species')
plt.rcParams["figure.figsize"] = (10,6)
plt.title('Fish Count Based On Species')
plt.show()

#removing outliers
print(df[df.Weight <= 0])

#we see that roach's weight is 0 so remove
df1 = df.drop([40])
print('New dimension of dataset is= ', df1.shape)



#making correlation matrix
df1.corr()
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
sns.heatmap(df1.corr(), annot =True)
plt.title('Correlation Matrix')
plt.show()

#dropping VerticalLen', 'DiagonalLen' and 'Crosslen' column
df2 = df1.drop(['VerticalLen', 'DiagonalLen', 'CrossLen'], axis =1) # Can also use axis = 'columns'
print('New dimension of dataset is= ', df2.shape)
df2.head()


#outlier detection and removal 
#making a boxplot
sns.boxplot(x=df2['Weight'])
plt.title('Outlier Detection based on Weight')


#making a boxplot 
sns.boxplot(x=df2['Weight'])
plt.title('Outlier Detection based on Weight')

#function for outlier detection 
def outlier_detection(dataframe):
  Q1 = dataframe.quantile(0.25)
  Q3 = dataframe.quantile(0.75)
  IQR = Q3 - Q1
  upper_end = Q3 + 1.5 * IQR
  lower_end = Q1 - 1.5 * IQR 
  outlier = dataframe[(dataframe > upper_end) | (dataframe < lower_end)]
  return outlier

print(outlier_detection(df2['Weight']))

# 3 outliers at 142 143 144
df3 = df2.drop([142,143,144])
df3.shape

#checking height now 
sns.boxplot(x=df2['Height'])
plt.title('Outlier Detection based on Height')
# no outliers for height


# building machine learning model 