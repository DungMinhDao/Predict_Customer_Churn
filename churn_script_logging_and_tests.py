"""
This module is used for testing functions in the churn_library.py
module and logging the output of the test

Author: Dung Dao
Date: April 2022
"""

import os
import logging
import churn_library as cls
from config import keep_cols, cat_columns

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    # Define global variable for dataframe to reuse across all tests
    global df_bank

    # Test importing data from path
    try:
        df_bank = import_data('./data/bank_data.csv')
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    # Check for validity of imported dataframe (it cannot be empty)
    try:
        assert df_bank.shape[0] > 0
        assert df_bank.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        # Encode Attrition_Flag column as Churn with 0/1 values
        def attrition_filter(
            val): return 0 if val == "Existing Customer" else 1
        df_bank['Churn'] = df_bank['Attrition_Flag'].apply(attrition_filter)

        # Check for existence of previous EDA figures, delete them before
        # execution of EDA
        for feature in ['churn', 'customer_age', 'marital_status', 'total_transaction']:
            file_path = './images/eda/' + feature + '_distribution.png'
            if os.path.isfile(file_path):
                logging.warning('Testing perform_eda: %s exist, will be overwritten' % file_path)
                os.remove(file_path)
        file_path = './images/eda/heatmap.png'
        if os.path.isfile(file_path):
            logging.warning('Testing perform_eda: %s exist, will be overwritten' % file_path)
            os.remove(file_path)

        # Perform EDA and check for figues existence in EDA folder
        perform_eda(df_bank)
        for feature in ['churn', 'customer_age', 'marital_status', 'total_transaction']:
            assert os.path.isfile('./images/eda/%s_distribution.png' % feature)
        assert os.path.isfile('./images/eda/heatmap.png')
        logging.info("Testing perform_data: SUCCESS")

    # Catch exception when some figures are missing in EDA folder
    except AssertionError as err:
        logging.error(
            'Testing perform_eda: EDA completed, but some figure file(s) are missing')
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df_encoded = encoder_helper(df_bank, category_lst=cat_columns)
        # Check that new encoded columns are in the dataframe
        for column in cat_columns:
            assert column + '_Churn' in df_encoded.columns
        logging.info('Testing encoder_helper: SUCCESS')

    except AssertionError as err:
        logging.error('Testing encoder_helper: Missing column after encode')
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    # Define the necessary columns for feature engineering
    try:
        # Set global variable for data splits for reusing later
        global X_train, X_test, y_train, y_test
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df_bank[keep_cols + ['Churn']])

        # Check if the set of columns after feature engineering is preserved
        feature_engineered_col = keep_cols[:-5] + \
            [col + '_Churn' for col in keep_cols[-5:]]
        assert set(X_train.columns) == set(feature_engineered_col)
        assert set(X_test.columns) == set(feature_engineered_col)

        # Check the consistency of data dimension before and after splitting
        assert len(X_train) + len(X_test) == len(df_bank)
        assert len(y_train) + len(y_test) == len(df_bank)
        assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
        logging.info('Testing perform_feature_engineering: SUCCESS')

    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: Something wrong with spliting data')
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        # Check if 2 models file, ROC curve and classification reports are available
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('./images/results/roc_curve_result.png')
        assert os.path.isfile('./images/results/rf_results.png')
        assert os.path.isfile('./images/results/logistic_results.png')
        logging.info('Tesing train_models: SUCCESS')

    except AssertionError as err:
        logging.error('Tesing train_models: A model file, the ROC curve file, \
            or a classification report is missing')
        raise err


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
