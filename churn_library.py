# library doc string
"""
This module is a library of functions to find customers
who are likely to churn.

Author: Dung Dao
Date: April 2022
"""

# import libraries
# pylint: disable=wrong-import-position
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import ImageDraw, Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from conftest import keep_cols, cat_columns
# pylint: enable=wrong-import-position

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
plt.rcParams['axes.grid'] = False
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Histogram of churn distribution
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.title('Churn distribution', fontsize=20)
    plt.savefig('./images/eda/churn_distribution.png')
    plt.clf()

    # Histogram of customers' age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.title('Age distribution', fontsize=20)
    plt.savefig('./images/eda/customer_age_distribution.png')
    plt.clf()

    # Bar plot of customers' marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title('Marital status distribution', fontsize=20)
    plt.savefig('./images/eda/marital_status_distribution.png')
    plt.clf()

    # Histogram of customers` total transaction count distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title('Total transaction count distribution', fontsize=20)
    plt.savefig('./images/eda/total_transaction_distribution.png')
    plt.clf()

    # Heatmap of feature correlation
    plt.figure(figsize=(30, 20))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.title('Heatmap of feature correlation', fontsize=20)
    plt.savefig('./images/eda/heatmap.png')
    plt.clf()


def encoder_helper(df, category_lst, response="_Churn"):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name

    output:
            df: pandas dataframe with new columns for
    '''
    # Iterate through each column for encoding
    for cat_column in category_lst:

        # Create a list to store encoded value
        val_lst = []
        # Group the column by value and take the churn proportion for each
        # group
        val_groups = df.groupby(cat_column).mean()['Churn']

        # Append the churn proportion correspond to the column's values to the
        # list
        for val in df[cat_column]:
            val_lst.append(val_groups.loc[val])

        # Store the list value in a new column
        df[cat_column + response] = val_lst

    return df


def perform_feature_engineering(df, response='_Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Encode the categorical columns using the helper function
    df = encoder_helper(df, category_lst=cat_columns, response=response)
    # Set the 'Churn' as output
    y = df['Churn']
    # Drop the 'Churn' column from the input
    X = df.drop(columns=['Churn'] + cat_columns, axis=1)
    # Split the dataset into train and test sets with ratio 70%-30%
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Classification report for random forest
    img = Image.new('RGB', (400, 400))
    lr_img = ImageDraw.Draw(img)
    lr_img.text((40, 10), "Classification Report")
    lr_img.text((40, 50), 'random forest results')
    lr_img.text((40, 70), 'test results')
    lr_img.text((40, 80), classification_report(y_test, y_test_preds_rf))
    lr_img.text((40, 230), 'train results')
    lr_img.text((40, 240), classification_report(y_train, y_train_preds_rf))
    img.save('images/results/rf_results.png')

    # Classification report for logistic regression
    img = Image.new('RGB', (400, 400))
    lr_img = ImageDraw.Draw(img)
    lr_img.text((40, 10), "Classification Report")
    lr_img.text((40, 50), 'logistic regression results')
    lr_img.text((40, 70), 'test results')
    lr_img.text((40, 80), classification_report(y_test, y_test_preds_lr))
    lr_img.text((40, 230), 'train results')
    lr_img.text((40, 240), classification_report(y_train, y_train_preds_lr))
    img.save('images/results/logistic_results.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 16))

    # Create plot title and y-axis label
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels and save figure
    plt.xticks(range(X_data.shape[1]), names, rotation=60)
    plt.savefig(output_pth)
    plt.close('all')


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # Create a random forest classifier for grid search
    rfc = RandomForestClassifier(random_state=42)

    # Define necessary hyparameters for searching
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Grid search for best models using random forest and predefined
    # hyperparameters
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Train logistic regression models with lbfgs solver
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)

    # Make prediction using the models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Make ROC curve plot for models
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.close('all')

    # Make classificaion report image
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)


if __name__ == '__main__':
    # Read the data in and perform EDA
    df_bank = import_data('./data/bank_data.csv')
    attrition_filter = lambda val: 0 if val == "Existing Customer" else 1
    df_bank['Churn'] = df_bank['Attrition_Flag'].apply(attrition_filter)
    perform_eda(df_bank)

    # Split the data into train-test set and train
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_bank[keep_cols + ['Churn']])
    # train_models(X_train, X_test, y_train, y_test)

    # Analyze feature importance of random forest classifier
    rfc_model = joblib.load('./models/rfc_model.pkl')
    feature_importance_plot(
        model=rfc_model,
        X_data=X_train,
        output_pth="./images/results/feature_importances.png")
