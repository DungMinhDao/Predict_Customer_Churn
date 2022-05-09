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
