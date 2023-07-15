#Importing the libraries
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score
import h2o
from h2o.automl import H2OAutoML

# Initialization the H2o Package
h2o.init()

# Load data
train_df = pd.read_csv('data_processed.csv')
test_df = h2o.import_file('test_data.csv')
print(train_df.sample(2))

# Converting it to H2o Dataframe
h2o_train_df = h2o.H2OFrame(train_df)
#h2o_test_df = h2o.H2OFrame(test_df)

# Extract X and y variables and values
x = h2o_train_df.columns
y = 'pollutant_avg'
x.remove(y)

# # Scaling the independent features
# z = x.values
# std_scaler = StandardScaler()
# z_scaled = std_scaler.fit_transform(z)
# x = pd.DataFrame(z_scaled)


aml = H2OAutoML(max_models=10, 
                seed=1, 
                exclude_algos =['DeepLearning'], 
                balance_classes = False, 
                )
aml.train(x=x, y=y, training_frame=h2o_train_df)

lb = aml.leaderboard
print('-'*100)
print('Performance Matrix of all the lead models: ')
print(lb.head(rows=lb.nrows))

# Make predictions
pred = aml.leader.predict(test_df)

# Merging the predicted feature to test dataframe
pred_df = test_df.cbind(pred)
pred_df.set_names(['state', 'city', 'pollutant_id', 'pollutant_min', 'pollutant_max', 'pred(pollutant_avg)'])
print('-'*100)
print('Renamed Dataframe: ')
print(pred_df.head(3))

# Saving the predicted dataframe & model
h2o.export_file(pred_df, path='Predicted_Data.csv', force=True)
h2o.save_model(aml.leader, path = "h2o_model")

# Get the top model of leaderboard
lead_model = aml.leader
 
# Get the metalearner model of top model
metalearner = h2o.get_model(lead_model.metalearner()['name'])
 
# list baselearner models :
print('-'*100)
print('Details of metal learner')
print(metalearner.varimp())

# Converting the model metrics to list
leader_models = h2o.as_list(lb)
print(leader_models)

# Coverting the predicted dataframe to list
pred_result = h2o.as_list(pred_df)

# - - - - - - - GENERATE METRICS FILE
with open("metrics.txt", 'w') as outfile:
        outfile.write(f'\n 1. Model_Metrics of Top Learners = \n {leader_models}, \n\n\n 2. Predicted_Dataframe =\n {pred_result}, \n\n\n 3. Meta Learners = \n {metalearner.varimp()}.')