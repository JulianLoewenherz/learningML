import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('Google_Stock_Price_Train.csv')

# Set the display option to show all columns
pd.set_option('display.max_columns', None)

# Function to clean and convert columns
def clean_column(column):
    if df[column].dtype == 'object':  # Check if the column is of object type (likely string)
        df[column] = df[column].str.replace(',', '').astype(float)
    else:
        df[column] = df[column].astype(float)

# Clean and convert relevant columns
columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
for column in columns_to_convert:
    clean_column(column)

# Print the first few rows of the DataFrame
print(df.head())

# High Low percentage
df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100
# Percent change
df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100
# Select relevant features
df = df[['Close', 'HL_PCT', 'PCT_change', 'Volume']]

# Print the first few rows of the updated DataFrame
print(df.head())
