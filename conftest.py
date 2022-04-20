"""
This module contains fixture functions of functions defined
in churn_library.py for other testing modules to access. It
also contain variables of column information for all modules
to access.

Author: Dung Dao
Date: April 2022
"""

import pytest
import churn_library as cls

# Columns kept for feature engineering and training
keep_cols = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']

# List of categorical columns for encoding
cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


@pytest.fixture
def import_data():
    '''Fixture for import_data function'''
    return cls.import_data


@pytest.fixture
def perform_eda():
    '''Fixture for perform_eda function'''
    return cls.perform_eda


@pytest.fixture
def encoder_helper():
    '''Fixture for encoder_helper function'''
    return cls.encoder_helper


@pytest.fixture
def perform_feature_engineering():
    '''Fixture for perform_feature_engineering function'''
    return cls.perform_feature_engineering


@pytest.fixture
def train_models():
    '''Fixture for train_models function'''
    return cls.train_models
