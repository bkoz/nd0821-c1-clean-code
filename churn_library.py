# library doc string
'''
File: churn_library.py
Description: A collection of functions that train, test and run scikit-learn
models used to prodict customer churn from sample data.

Author: bkozdemba@gmail.com
Date: May 2022

Usage:
$ python churn_library.py --log-file=logs/churn_library.log --log-level=INFO
'''

# import libraries
from asyncio.log import logger
import os
import logging
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import plot_roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
import seaborn as sns

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s : %(levelname)s : %(message)s')


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    try:
        data_frame = pd.read_csv(pth)
        logging.info("%s : %s", "import_data", "SUCCESS")
        return data_frame
    except FileNotFoundError:
        logging.error("import_data: The file wasn't found")
        return None


def perform_eda(data_frame):
    '''
    perform eda on data_frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    # Make sure Attrition_Flag is either 0 or 1
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Plot a Churn histogram.
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig("./images/eda/churn_histogram.png")
    logging.info("%s : %s", "perform_eda: churn histogram", "SUCCESS")

    # Plot a Customer_Age histogram.
    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig("./images/eda/customer_age_histogram.png")
    logging.info("%s : %s", "perform_eda: customer_age histogram", "SUCCESS")

    # Plot the marital status counts.
    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("./images/eda/marital_status_counts.png")
    logging.info(
        "%s : %s",
        "perform_eda: plot marital status counts",
        "SUCCESS")

    # Plot the total transaction histogram.
    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("./images/eda/total_transaction_histogram.png")
    logging.info(
        "%s : %s",
        "perform_eda: plot total transaction histogram",
        "SUCCESS")

    # Plot the heat map.
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("./images/eda/heatmap.png")
    logging.info("%s : %s", "perform_eda: plot heatmap", "SUCCESS")


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for category in category_lst:
        lst = []
        groups = data_frame.groupby(category).mean()['Churn']

        for val in data_frame[category]:
            lst.append(groups.loc[val])

        data_frame[f'{category}_Churn'] = lst

    logging.info("encoder_helper: SUCCESS")
    return data_frame

def perform_feature_engineering(data_frame, response):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_churn = data_frame['Churn']
    x_data_frame = pd.DataFrame()
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
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_data_frame[keep_cols] = data_frame[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data_frame, y_churn, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images/results folder
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
    # Save the random forest classification report image.
    #
    # Clear the current figure.
    plt.close()
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/rf_classification_report.png")
    logging.info(
        "%s : %s",
        "classification_report_image: random forest",
        "SUCCESS")

    # Save the logistic regression classification report image.
    plt.close()
    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig("./images/results/lr_classification_report.png")
    logging.info(
        "%s : %s",
        "classification_report_image: logistic regression",
        "SUCCESS")


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances and plot
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.close()
    plt.clf()
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)
    logging.info(
        "%s : %s",
        "feature_importance_plot: saved feature report to ",
        output_pth)


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    logging.info("%s : %s", "train_models: GridSearch", "Running")
    start_time = time.time()
    cv_rfc.fit(x_train, y_train)
    logging.info(
        "%s : %s : %f secs",
        "train_models: GridSearch",
        "SUCCESS",
        time.time() -
        start_time)

    logging.info("%s : %s", "train_models: Logistic Regression", "Running")
    start_time = time.time()
    lrc.fit(x_train, y_train)
    logging.info(
        "%s : %s : %f secs",
        "train_models: Logistic Regression",
        "SUCCESS",
        time.time() - start_time)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # scores
    print('random forest results')
    print('test results')
    print(classification_report(y_test, y_test_preds_rf))
    print('train results')
    print(classification_report(y_train, y_train_preds_rf))

    print('logistic regression results')
    print('test results')
    print(classification_report(y_test, y_test_preds_lr))
    print('train results')
    print(classification_report(y_train, y_train_preds_lr))

    # Save the models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Save scores as images.
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    # Save the feature importance images.
    feature_importance_plot(
        cv_rfc,
        x_train,
        './images/results/cv_feature_importance.png')

    # Plot and save the ROC curves.    
    logging.info(
        "%s : %s",
        "model_train: save roc curve",
        "BEGIN")
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    # fpr, tpr, _ = roc_curve(y_test, y_test_preds_lr)
    # roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.savefig("./images/results/lrc_roc_curve.png")
    logging.info(
        "%s : %s",
        "model_train: save lrc roc curve",
        "SUCCESS")
    # Combine both lrc ans rfc roc plots
    plt.clf()
    plt.close()
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("./images/results/lrc_rfc_roc_curves.png")
    logging.info(
        "%s : %s",
        "model_train: save lrc and rfc roc curves",
        "SUCCESS")
    

if __name__ == "__main__":
    main_data_frame = import_data('./data/bank_data.csv')
    perform_eda(main_data_frame)
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
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
        'Avg_Utilization_Ratio'
    ]
    main_data_frame = encoder_helper(main_data_frame, cat_columns, None)
    main_x_train, main_x_test, main_y_train, main_y_test = perform_feature_engineering(
        main_data_frame, None)
    train_models(main_x_train, main_x_test, main_y_train, main_y_test)
