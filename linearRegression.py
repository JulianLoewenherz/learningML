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
X = df3[['Height','Width']] # Select columns using column name
print(X.head())

y = df3[['Weight']]
print(y.head())

#splitting the dataset to use one set for testing and the other for training the model
#test_size=0.2 means that 20% of data will be used for testing and 80% for training the model
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state = 1) 
print('X_train dimension= ', X_train.shape)
print('X_test dimension= ', X_test.shape)
print('y_train dimension= ', y_train.shape)
print('y_train dimension= ', y_test.shape)

# training the model using Ordinary Least Squares Algorithm
model = linear_model.LinearRegression()
model.fit(X_train,y_train) #this trains the data to the x_train and y_train data specified earlier using OLS

#printing the calculated coefficients 
print('coef= ', model.coef_) #expecting one for height and one for weight
print('intercept= ', model.intercept_)
print('score= ', model.score(X_test,y_test)) # returns the accuracy of the model

#predicting data in a column and comparing it
predictedWeight = pd.DataFrame(model.predict(X_test), columns=['Predicted Weight']) # Create new dataframe of column'Predicted Weight'
actualWeight = pd.DataFrame(y_test)
actualWeight = actualWeight.reset_index(drop=True) # Drop the index so that we can concat it, to create new dataframe
df_actual_vs_predicted = pd.concat([actualWeight,predictedWeight],axis =1)
print(df_actual_vs_predicted.T)

#visualizing predicted vs actual in a scatterplot
plt.scatter(y_test, model.predict(X_test)) 
plt.xlabel('Weight From Test Data')
plt.ylabel('Weight Predicted By Model')
plt.rcParams["figure.figsize"] = (10,6) 
plt.title("Weight From test Data Vs Weight Predicted By Model")
plt.show()

plt.scatter(X_test['Height'], y_test, color='red', label = 'Actual Weight')
plt.scatter(X_test['Height'], model.predict(X_test), color='green', label = 'Predicted Weight')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.rcParams["figure.figsize"] = (10,6) 
plt.title('Actual Vs Predicted Weight for Test Data')
plt.legend()
plt.show()




















