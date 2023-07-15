# Air Quality Data
'''
- Reading air quality data from https://data.gov.in/. 
- All the data belongs to different states in India.
- Alternatively you can find the dataset from this link https://raw.githubusercontent.com/deepanshu88/Datasets/master/UploadedFiles/stations.csv
- Use pandas.read_csv command to read the data from the above link
'''

# Import necessary libraries
import requests
import json
import pandas as pd
import re
import datetime
import time
import base64
from itertools import product

#stationsData = pd.read_csv("https://raw.githubusercontent.com/deepanshu88/Datasets/master/UploadedFiles/stations.csv")

def getData(api, criteria):
    # url from where the data is copied + api is the api which is generated from our account + format for json data
    url1 = "https://api.data.gov.in/resource/3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69?api-key=" + api + "&format=json&limit=500"
    # Grouping each city name together, filtering the unwanted characters with space inbetween (%20 represents space in url)   
    criteriaAll = [[(k, re.sub(r'\s+', '%20', v)) for v in criteria[k]] for k in criteria] 
    #print(criteriaAll)  
    # Creating an url with all the city names and pollutant_id, combining it in a proper order  
    url2 = [url1 + ''.join(f'&filters[{ls}]={value}' for ls, value in p) for p in product(*criteriaAll)]  
    #print('-'*60)
    #print(url2)
    #print('-'*60)
    
    pollutionDfAll = pd.DataFrame()
    for i in url2:
        response = requests.get(i, verify=True)     # Reading the values from url2
        #print(response.text)
        response_dict = json.loads(response.text)   # Loading the data read from previous step in the form of dictionary
        #print(response_dict)
        # Fetching the records data alone from the above json data and converting it to dataframe
        pollutionDf = pd.DataFrame(response_dict['records']) 
        # Concatinating all the dataframes from the url   
        pollutionDfAll = pd.concat([pollutionDfAll, pollutionDf])
    
    return pollutionDfAll


# Sample key
api = "579b464db66ec23bdd00000133a563b99ca242b87dbd7cc8b98ec894"   # API key generated from 'https://data.gov.in/'
# Mention all the city names from where data has to be fetched and mention the different pollutant_id
# Ref this link for city names: https://raw.githubusercontent.com/deepanshu88/Datasets/master/UploadedFiles/stations.csv
criteria = {'city':['Greater Noida','Delhi', 'Chennai', 'Mumbai', 'Bengaluru', 'Thiruvananthapuram', 'Ernakulam', 'Hyderabad',   
                    'Kochi', 'Mandideep', 'Bhopal', 'Pune - IITM', 'Lucknow', 'Jaipur', 'Kolkata', 'Chandigarh'], 
            'pollutant_id': ['PM10', 'PM2.5', 'NO2', 'NH3', 'SO2', 'CO', 'OZONE']}
mydata = getData(api, criteria)

# Displaying 10 sample rows from the created dataframe
print('-'*100)
print(mydata.sample(10))

# Saving the data to the local drive for future use
mydata.to_csv('air_quality_data.csv', index=False)