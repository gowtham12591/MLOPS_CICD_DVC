# Preprocessing the data and making it suitable for model building
'''
- Here the dependent variable is 'pollutant_avg'
- The dataset has values mixed with integer and object types which has to be pre-processed
- Apart from that we can see a lot of missing values which are to be handled
'''

#Importing the libraries used
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Reading the dataset using pandas dataframe
df = pd.read_csv('air_quality_data.csv')
print(df.info())
print('-'*100)
print(df.sample(10))

# Check the missing values in the dataframe
print('-'*100)
print('Missing value check')
print(df.isnull().sum())

'''
- Out of all the entires 'pollutant_unit' has only null values. So this feature can be dropped
- Apart from this other features like pollutant_min, pollutant_max and pollutant_avg has missing values 
which can be replaced with suitable techniques
'''

# Replacing the missing values in pollutant_min and pollutant_max features
df['pollutant_min'] = df['pollutant_min'].fillna(df['pollutant_min'].median())
df['pollutant_max'] = df['pollutant_max'].fillna(df['pollutant_max'].median())
df.drop(columns=['pollutant_unit'], inplace=True)

# Checking for missing values
print('-'*100)
print('Missing value check')
print(df.isnull().sum())

'''
- The target feauture 'pollutant_avg' has some missing values which can be used as a target feature for our future validation
'''
test_df = df[pd.isnull(df['pollutant_avg'])]
print('-'*100)
print('Seperating train and test data')
print(test_df.info())

# Dropping the target feature from the test_dataframe
test_df.drop(columns=['pollutant_avg'], inplace=True)

# Creating a train dataframe
train_df = df.dropna()
print('-'*100)
print(train_df.info())

# Let us see the unique values of the entire train dataframe to understand the important features
print('-'*100)
print('Unique values of independent features: ')
print('id: ', train_df['id'].nunique())
print('country: ', train_df['country'].nunique())
print('state: ', train_df['state'].nunique())
print('city: ', train_df['city'].nunique())
print('station: ', train_df['station'].nunique())
print('last_update: ', train_df['last_update'].nunique())
print('pollutant_id: ', train_df['pollutant_id'].nunique())

'''
- Features like id, country and last_update has either all the rows as unique or only one unique feature in the entire dataframe
- So it is better to drop these features along with station feature which is too complex with each name
'''

# Dropping unnecessary columns
train_df.drop(columns=['id', 'country', 'station', 'last_update'], inplace=True)
print('-'*100)
print('Reduced Dataframe: ')
print(train_df.info())

'''
- Now we have to encode state, city and pollutant_id
- Let us encode those features with label encoder
'''

# Initialising label encoder
lb = LabelEncoder()
for i in train_df.columns:
    if train_df[i].dtypes == 'object':
        train_df[i] = lb.fit_transform(train_df[i])

print('-'*100)
print('Encoded Dataframe: ')
print(train_df.info())

'''
- Let us try to perform correlation matrix to undertand the relationship with each independent feature
'''
print('-'*100)
print('Correlation Matrix: ')
print(train_df.corr())

'''
- Only the pollutant_min and polutant_max has good correlation with the target feature which can help for better prediction
- Let us save the final dataframe to use it for further modelling
'''

# Saving the dataframe for model training
train_df.to_csv('data_processed.csv', index=False)

'''
Following the same pre-processing steps for test data
'''

# Dropping unnecessary columns
test_df.drop(columns=['id', 'country', 'station', 'last_update'], inplace=True)

# Initialising label encoder
lb = LabelEncoder()
for i in test_df.columns:
    if test_df[i].dtypes == 'object':
        test_df[i] = lb.fit_transform(test_df[i])

print('-'*100)
print('Encoded Dataframe: ')
print(test_df.info())


# Saving the test dataset for model validation
test_df.to_csv('test_data.csv', index=False)