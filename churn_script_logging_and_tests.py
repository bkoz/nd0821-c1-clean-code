"""
File: churn_script_logging_and_tests.py
Description: Python script that define unit test for the churn_library.py functions.

Author: bkozdemba@gmail.com
Creation Date: May 17, 2022

Usage:
$ pytest churn_script_logging_and_tests.py --log-file=logs/churn_library.log --log-level=INFO
"""
import logging
from genericpath import exists
import pytest
import churn_library as cl

# Configure logging.
logging.basicConfig(
filename='./logs/churn_library.log',
level=logging.INFO,
filemode='w',
encoding='utf-8',
format="%(asctime)-15s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@pytest.fixture(scope="module")
def import_data():
    '''
    Fixture - The test function test_import() will
    use the return of import_data() as an argument
    '''
    return cl.import_data


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df_local = import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df_local.shape[0] > 0
        assert df_local.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope="module")
def perform_eda():
    '''
    Fixture - The test function test_eda() will
    use the return of eda() as an argument
    '''
    return cl.perform_eda


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    try:
        df_local = cl.import_data('./data/bank_data.csv')
        perform_eda(df_local)
        assert exists('./images/churn_histogram.png')
        assert exists('./images/heatmap.png')
        assert exists('./images/total_transaction_histogram.png')
        assert exists('./images/customer_age_histogram.png')
        assert exists('./images/marital_status_counts.png')
        logger.info("Testing perform_eda: SUCCESS")

    except AssertionError as err:
        logger.error(
            "Testing perform_eda: FAILED!")
        raise err

@pytest.fixture(scope="module")
def encoder_helper():
    '''
    Fixture - The test function test_eda() will
    use the return of eda() as an argument
    '''
    return cl.encoder_helper


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df_local = cl.import_data('./data/bank_data.csv')
    cl.perform_eda(df_local)

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

    try:
        df_local = encoder_helper(df_local, cat_columns, quant_columns)
        assert df_local.shape[0] == 10127
        logger.info("Testing encoder_helper: SUCCESS")

    except AssertionError as err:
        logger.error("Testing encoder_helper: FAILED!")
        raise err


@pytest.fixture(scope="module")
def perform_feature_engineering():
    '''
    Fixture - The test function test_perform_feature_engineering() will
    use the return of perform_feature_engineering() as an argument
    '''
    return cl.perform_feature_engineering


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df_local = cl.import_data('./data/bank_data.csv')
    cl.perform_eda(df_local)

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

    df_local = cl.encoder_helper(df_local, cat_columns, None)

    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(df_local, None)
        assert x_train.shape[0] and y_train.shape[0] == 7088 \
            and x_test.shape[0] and y_test.shape[0] == 3039
        logger.info("Testing perform_feature_engineering: SUCCESS")

    except AssertionError as err:
        logger.error("Testing perform_feature_engineering: FAILED!")
        raise err


@pytest.fixture(scope="module")
def train_models():
    '''
    Fixture - The test function test_perform_feature_engineering() will
    use the return of perform_feature_engineering() as an argument
    '''
    return cl.train_models


def test_train_models(train_models):
    '''
    test train_models
    '''
    df_local = cl.import_data('./data/bank_data.csv')
    cl.perform_eda(df_local)

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

    df_local = cl.encoder_helper(df_local, cat_columns, None)

    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(df_local, None)

    try:
        cl.train_models(x_train, x_test, y_train, y_test)
        assert exists('models/logistic_model.pkl')
        assert exists('models/rfc_model.pkl')
        logger.info("Testing train_models: SUCCESS")

    except AssertionError as err:
        logger.error("Testing train_models: FAILED!")
        raise err

# if __name__ == "__main__":
#     pass
