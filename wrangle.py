#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Wrangles data from Zillow Database'''

##################################################Wrangle.py##########################

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from env import user, password, host

#**************************************************Acquire*****************************

def acquire_zillow():
    ''' Acquires data from Zillow using env imports and renames columns'''
    url = f'mysql+pymysql://{user}:{password}@{host}/zillow'
    query = '''
SELECT bedroomcnt
, bathroomcnt
, calculatedfinishedsquarefeet
, taxvaluedollarcnt
, yearbuilt
, taxamount
, fips
FROM properties_2017
LEFT JOIN propertylandusetype USING(propertylandusetypeid)
WHERE propertylandusedesc IN ("Single Family Residential"
                                , "Inferred Single Family Residential")'''
    # return dataframe of zillow data
    df = pd.read_sql(query, url)
    # rename columns
    df = df.rename(columns= {'bedroomcnt': 'bedrooms'
                         , 'bathroomcnt': 'bathrooms'
                         , 'calculatedfinishedsquarefeet': 'sqr_ft'
                         , 'taxvaluedollarcnt': 'tax_value'
                         , 'yearbuilt': 'year_built'
                         , 'taxamount': 'tax_amount'})
    return df

#**************************************************Remove Outliers*************************

def remove_outliers(df, k, col_list):
    ''' removes outliers from a list of columns in a dataframe
        and returns that dataframe'''
    for col in col_list:
        
        # get quartiles
        q1, q3 = df[col].quantile([.25, .75])
        
        # calculate interquartile range
        iqr = q3 - q1
        
        # get upper bound
        upper_bound = q3 + k * iqr
        
        # get lower bound
        lower_bound = q1 - k * iqr
        
        # return dataframe without outliers
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df

#**************************************************Distributions******************

def get_hist(df):
    ''' Returns histographs of acquires continuous variables'''
    plt.figure(figsize=(16, 3))
    
    # list of columns
    cols = [col for col in df.columns if col not in ['fips', 'year_built']]

    for i, col in enumerate(cols):
        
        # i starts at 0, but plots nos should start at 1
        plot_number = i + 1
        
        # create subplot
        plt.subplot(1, len(cols), plot_number)
        
        # title with column name
        plt.title(col)
        
        # display histogram for column
        df[col].hist(bins=5)
        
        # hide gridelines
        plt.grid(False)
        
        # turn off scientific notation
        plt.ticklabel_format(useOffset=False)
        
        # set proper spacing between plots
        plt.tight_layout()
        
    plt.show()
    
def get_box(df):
    ''' Gets boxplots of acquire continuous variables'''
    
    # List of columns
    cols = ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount']
    
    plt.figure(figsize=(16, 3))
    
    for i, col in enumerate(cols):
        
        # i starts at 0, but plots nos should start at 1
        plot_number = i + 1
        
        # create subplot
        plt.subplot(1, len(cols), plot_number)
        
        # title with column name
        plt.title(col)
        
        # display boxplot for column
        sns.boxplot(data=df[[col]])
        
        # hide gridelines
        plt.grid(False)
        
        # set proper spacing between plots
        plt.tight_layout()
    
    plt.show()
    
    def prepare_zillow(df):
        ''' Prepare zillow data for exploration'''

        # remove outliers
        df = remove_outliers(df, 1.5, ['bedrooms', 'bathrooms', 'area', 'tax_value', 'taxamount'])

        # get distributions of numeric data
        get_hist(df)
        get_box(df)

        # converting column datatypes
        df.fips = df.fips.astype(object)
        df.year_built = df.year_built.astype(object)

        #train, validate, test split
        train_validate, test = train_test_split(df, test_size=.2, random_state=123)
        train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        # impute year built using mode
        imputer = SimpleImputer(strategy='median')

        imputer.fit(train[['year_built']])

        train[['year_built']] = imputer.transform(train[['year_built']])
        validate[['year_built']] = imputer.transform(validate[['year_built']])
        test[['year_built']] = imputer.transform(test[['year_built']])

        return train, validate, test

#**************************************************Wrangle*******************************************************

def wrangle_zillow():
    ''' Acquire and prepare data from Zillow batabase for explore'''
    train, validate, test = prepare_zillow(acquire_zillow())
    
    return train, validate, test

