# -*- coding: utf-8 -*-
"""
Kaggle Titanic UV Analytics tutorial
"""

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# read in the training and testing data into Pandas.DataFram objects
input_df = pd.read_csv('data/train.csv', header=0)
submit_df = pd.read_csv('data/test.csv', header=0)

'''
We combine the data from the two files into one for a simple reason: when we
perform feature engineering on the features, it’s often useful to know the full
range of possible values, as well as the distributions of all known values. 
This will require that we keep track of the training and test data during our 
processing, but it turns out to not be too difficult.
'''
# merge the two DataFrames into one
df = pd.concat([input_df, submit_df])

# re-number the combined data set so there aren't duplicate indexes
df.reset_index(inplace=True)

# reset_index() generates a new column that we don't want, 
# so let's get rid of it
df.drop('index', axis=1, inplace=True)

# the remaining columns need to be reindexed so we can access 
# the first column at '0' instead of '1'
df = df.reindex_axis(input_df.columns, axis=1)

'''Missing Values: 3 methods'''
'''1) Assign a value that indicates a missing value'''
# Replace missing values with "U0"
df['Cabin'][df.Cabin.isnull()] = 'U0'

'''2) Assign the average/most common value'''
# Take the median of all non-null Fares and use that for all missing values
df['Fare'][ np.isnan(df['Fare']) ] = df['Fare'].median()

# Replace missing values with most common port
df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

'''3) Use a model to predict the values of missing variables'''
### Populate missing ages using RandomForestClassifier
### requires derived features created later
### see setMissingAges

'''Variable Transformations'''
'''1) Dummy Variables'''    
# Create a dataframe of dummy variables for each distinct value of 'Embarked'
dummies_df = pd.get_dummies(df['Embarked'])

# Rename the columns from 'S', 'C', 'Q' to 'Embarked_S', 'Embarked_C', 'Embarked_Q'
dummies_df = dummies_df.rename(columns=lambda x: 'Embarked_'+str(x))

# Add the new variables back to the original data set
df = pd.concat([df,dummies_df], axis=1)

'''2) Factorizing'''
# create feature for the alphabetical part of the cabin number
df['CabinLetter'] = df['Cabin'].map( lambda x: re.compile("([a-zA-Z]+)").search(x).group())

# convert the distinct cabin letters with incremental integer values
df['CabinLetter'] = pd.factorize(df.CabinLetter)[0]

df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# continued from missing value imputation
def setMissingAges(df):
    
    # Grab all the features that can be included in a Random Forest Regressor
    age_df = df[['Age','Embarked_C','Embarked_Q','Embarked_S','Fare','Parch','SibSp','Title_Dr','Title_Lady','Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Rev','Title_Sir','Pclass','CabinLetter']]
    
    # Split into sets with known and unknown Age values
    knownAge = age_df.loc[ (df.Age.notnull()) ]
    unknownAge = age_df.loc[ (df.Age.isnull()) ]
    
    # All age values are stored in a target array
    y = knownAge.values[:,0]
    
       # All the other values are stored in the feature array
    X = knownAge.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(unknownAge.values[:,1::])
    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    
    return df.Age

df.Age = setMissingAges(df)
### end impute missing values

'''3) Scaling'''
# StandardScaler will subtract the mean from each value then scale to the unit variance
scaler = StandardScaler()
df['Age_scaled'] = scaler.fit_transform(df['Age'])

'''4) Binning'''
# Divide all fares into quartiles
df['Fare_bin'] = pd.qcut(df['Fare'], 4)

# qcut() creates a new variable that identifies the quartile range, but we can't use the string so either
# factorize or create dummies from the result
df['Fare_bin_id'] = pd.factorize(df.Fare_bin)[0]
df.drop('Fare_bin', axis=1, inplace=True)

'''Derived Variables'''
# What is each person's title? 
df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])

# Group low-occuring, related titles together
df['Title'][df.Title == 'Jonkheer'] = 'Master'
df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
df['Title'][df.Title == 'Mme'] = 'Mrs'
df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'

# Build binary features
df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)

# Create a feature for the deck
df['Deck'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
df['Deck'] = pd.factorize(df['Deck'])[0]

# Create binary features for each deck
decks = pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))
df = pd.concat([df, decks], axis=1)

# Create feature for the room number
def findRoom(c):
    res = re.compile("([0-9]+)").search(c)
    if res != None:
        return int(res.group(1)) + 1
    else:
        return 1
df['Room'] = df['Cabin'].map(findRoom)

def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'
 
def getTicketNumber(ticket):
    match = re.compile("([\d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'

def processTicket():
    global df
    
    # extract and massage the ticket prefix
    df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[\.?\/?]', '', x) )
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )
        
    # create binary features for each prefix
    prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
    df = pd.concat([df, prefixes], axis=1)
    
    # factorize the prefix to create a numerical categorical variable
    df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]
    
    # extract the ticket number
    df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )
    
    # create a feature for the number of digits in the ticket number
    df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)
    
    # create a feature for the starting number of the ticket number
    df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)
    
    # The prefix and (probably) number themselves aren't useful
    df.drop(['TicketPrefix', 'TicketNumber'], axis=1, inplace=True)
    
processTicket()

'''Automatically compute interaction variables'''
df['Fare_scaled'] = scaler.fit_transform(df['Fare'])
df['Pclass_scaled'] = scaler.fit_transform(df['Pclass'])
df['Parch_scaled'] = scaler.fit_transform(df['Parch'])
df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
df['Room_scaled'] = scaler.fit_transform(df['Room'])
df['Fare_bin_id_scaled'] = scaler.fit_transform(df['Fare_bin_id'])
numerics = df.loc[:, ['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled', 
                      'Room_scaled', 'Fare_bin_id_scaled']]
# for each pair of variables, determine which mathmatical operators to use based on redundancy
for i in range(0, numerics.columns.size-1):
    for j in range(0, numerics.columns.size-1):
        col1 = str(numerics.columns.values[i])
        col2 = str(numerics.columns.values[j])
        # multiply fields together (we allow values to be squared)
        if i <= j:
            name = col1 + "*" + col2
            df = pd.concat([df, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)
        # add fields together
        if i < j:
            name = col1 + "+" + col2
            df = pd.concat([df, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)
        # divide and subtract fields from each other
        if not i == j:
            name = col1 + "/" + col2
            df = pd.concat([df, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)
            name = col1 + "-" + col2
            df = pd.concat([df, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)

# calculate the correlation matrix (ignore survived and passenger id fields)
df_corr = df.drop(['Survived', 'PassengerId'],axis=1).corr(method='spearman')

# create a mask to ignore self-
mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
df_corr = mask * df_corr

drops = []
# loop through each variable
for col in df_corr.columns.values:
    # if we've already determined to drop the current variable, continue
    if np.in1d([col],drops):
        continue
    
    # find all the variables that are highly correlated with the current variable 
    # and add them to the drop list 
    corr = df_corr[abs(df_corr[col]) > 0.98].index
    drops = np.union1d(drops, corr)

print "\nDropping", drops.shape[0], "highly correlated features...\n", drops
df.drop(drops, axis=1, inplace=True)

''' Not using b/c using Random Forests
# Minimum percentage of variance we want to be described by the resulting transformed components
variance_pct = .99

# Create PCA object
pca = PCA(n_components=variance_pct)

# Transform the initial features
X_transformed = pca.fit_transform(X,y)

# Create a data frame from the PCA'd data
pcaDataFrame = pd.DataFrame(X_transformed)
'''

train = df.loc[(df.Survived.notnull())]
test = df.loc[(df.Survived.isnull())]
train = train.drop(['PassengerId'], axis=1)
test = test.drop(['PassengerId','Survived'], axis=1)
X_train = train.loc[:,'Pclass':]
y_train = train.loc[:,'Survived']
X_test = test.loc[:,'Pclass':]

# Fit a random forest with (mostly) default parameters to determine feature importance
forest = RandomForestClassifier(oob_score=True, n_estimators=10000)
forest.fit(X_train, y_train)
feature_importance = forest.feature_importances_
